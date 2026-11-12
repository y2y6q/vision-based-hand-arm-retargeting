"""Live video of one collection attempt: the operator's window as shown, with the time and the result on top.

Each frame is the view window's ``last_frame``: the three views exactly as
the operator sees them, the webcam inset included (the RESETTING notice is
drawn only on the displayed copy, so it is not in the video). One frame per
20 Hz control tick, so the video plays in real time. A header strip above
the window shows the task, a large timer from the first arm move and, at
the end, SUCCESS / FAILED / QUIT held for ``RESULT_HOLD_S``. The second
before the first move is kept (``PREROLL_S``); an attempt that never
started leaves no file.

Encoding runs on this object's own thread and in ffmpeg's own process
(below-normal priority); ``push`` never blocks the control loop (a full
queue drops that frame and counts it in ``dropped``).
"""

from __future__ import annotations

from collections import deque
import itertools
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Optional

import numpy as np

# Tuning, all in one place.
VIDEO_WIDTH = 1920  # the window (e.g. 2544 x 1334) is scaled down to this width; never scaled up
VIDEO_CRF = 23  # libx264 quality: lower is better and bigger
VIDEO_PRESET = "veryfast"
HEADER_HEIGHT = 72  # at a 1920-wide video; scaled with the width
PREROLL_S = 1.0  # kept from before the first move
RESULT_HOLD_S = 2.0  # the last frame with the result, at the end
QUEUE_FRAMES = 20  # frames waiting for the encoder; more are dropped (counted)
FINISH_TIMEOUT_S = 60.0

HEADER_BG = (24, 24, 24)
WHITE = (240, 240, 240)
GREY = (170, 170, 170)
AMBER = (255, 200, 0)
# result -> (word in the header, colour of the header and the frame border); RGB
RESULT_STYLE = {
    "success": ("SUCCESS", (34, 160, 70)),
    "failed": ("FAILED", (200, 50, 50)),
    "quit": ("QUIT", (105, 105, 105)),
}

_ids = itertools.count(1)


def video_geometry(window_w: int, window_h: int, width: int = VIDEO_WIDTH) -> tuple[int, int, int]:
    """(video width, window height in the video, header height); width and total height are multiples of 16."""

    window_w, window_h = int(window_w), int(window_h)
    if window_w < 16 or window_h < 16:
        raise ValueError("window frame too small: {}x{}".format(window_w, window_h))
    out_w = max(16, min(int(width), window_w) // 16 * 16)
    body_h = max(2, int(round(window_h * out_w / float(window_w) / 2.0)) * 2)
    header_h = max(32, int(round(HEADER_HEIGHT * out_w / 1920.0)))
    header_h += (-(body_h + header_h)) % 16  # H.264 macro blocks: no rescaling by ffmpeg
    return out_w, body_h, header_h


def _fit_text(cv2: Any, image: np.ndarray, text: str, origin: tuple[int, int], scale: float,
              color: tuple[int, int, int], thickness: int, max_width: int) -> None:
    (width, _height), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if width > max_width > 0:
        scale *= max_width / float(width)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_header(cv2: Any, canvas: np.ndarray, header_h: int, *, title: str, subtitle: str,
                clock: str, clock_color: tuple[int, int, int], background: tuple[int, int, int],
                subtitle_color: tuple[int, int, int] = GREY) -> None:
    """Task on the left, the large clock (or the result) right-aligned."""

    width = canvas.shape[1]
    s = header_h / float(HEADER_HEIGHT)
    pad = int(round(16 * s))
    canvas[:header_h] = background
    big, big_t = 1.5 * s, max(2, int(round(3 * s)))
    (clock_w, clock_h), _baseline = cv2.getTextSize(clock, cv2.FONT_HERSHEY_SIMPLEX, big, big_t)
    cv2.putText(canvas, clock, (width - pad - clock_w, (header_h + clock_h) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                big, clock_color, big_t, cv2.LINE_AA)
    room = width - clock_w - 3 * pad
    _fit_text(cv2, canvas, title, (pad, int(round(30 * s))), 0.8 * s, WHITE, max(1, int(round(2 * s))), room)
    _fit_text(cv2, canvas, subtitle, (pad, int(round(60 * s))), 0.6 * s, subtitle_color, max(1, int(round(s))), room)


def compose_frame(cv2: Any, canvas: np.ndarray, body: np.ndarray, header_h: int, *, title: str, subtitle: str,
                  elapsed_s: Optional[float] = None, result: Optional[tuple[str, Optional[float], str]] = None) -> np.ndarray:
    """Header + the window frame into ``canvas`` (reused between frames); returns it."""

    canvas[header_h:header_h + body.shape[0]] = body
    if result is not None:
        kind, time_s, note = result
        word, color = RESULT_STYLE.get(kind, (kind.upper(), RESULT_STYLE["quit"][1]))
        clock = word if time_s is None else "{}  {:.1f} s".format(word, float(time_s))
        draw_header(cv2, canvas, header_h, title=title, subtitle=subtitle + ("  |  " + note if note else ""),
                    clock=clock, clock_color=WHITE, background=color, subtitle_color=WHITE)
        border = max(3, int(round(6 * header_h / float(HEADER_HEIGHT))))
        cv2.rectangle(canvas, (border // 2, header_h + border // 2),
                      (canvas.shape[1] - 1 - border // 2, canvas.shape[0] - 1 - border // 2), color, border)
    elif elapsed_s is None:
        draw_header(cv2, canvas, header_h, title=title, subtitle=subtitle, clock="READY", clock_color=AMBER,
                    background=HEADER_BG)
    else:
        draw_header(cv2, canvas, header_h, title=title, subtitle=subtitle, clock="{:.1f} s".format(float(elapsed_s)),
                    clock_color=WHITE, background=HEADER_BG)
    return canvas


class _FfmpegPipe:
    """Raw RGB frames into ffmpeg's stdin -> H.264 MP4; ffmpeg runs at below-normal priority."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float) -> None:
        import imageio_ffmpeg

        self.path = Path(path)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "{}x{}".format(*size), "-r", "{:.3f}".format(float(fps)), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self.path),
        ]
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=self._stderr, creationflags=flags)

    def write(self, frame: np.ndarray) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).data)

    def close(self, timeout_s: float = FINISH_TIMEOUT_S) -> None:
        """Finish the file; raises with ffmpeg's message when it failed."""

        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
            code = self._process.wait(timeout=timeout_s)
        except Exception:
            self._process.kill()
            raise
        finally:
            self._stderr.seek(0)
            message = self._stderr.read()[-400:].decode("utf-8", "replace").strip()
            self._stderr.close()
        if code != 0:
            raise RuntimeError("ffmpeg exit {}: {}".format(code, message or "no message"))

    def kill(self) -> None:
        try:
            self._process.kill()
            self._process.wait(timeout=5.0)
        except Exception:
            pass
        try:
            self._stderr.close()
        except Exception:
            pass


class AttemptVideo:
    """One attempt's video, written on its own thread; ``finish`` names the file, ``discard`` leaves none.

    ``push(frame, elapsed_s)`` once per control tick with the window's latest
    frame and the time since the first move (``None`` before it). Frames
    that are ``stale_frame`` (the previous layout's last picture, before the
    window has drawn the new one) are skipped.

    While recording, the file is in ``work_dir`` (default ``folder``); it
    appears in ``folder`` only once finished. Explorer shows a file that
    ffmpeg still holds as 0 KB, and on 2026-11-10 that looked like a missing
    video in the videos folder.
    """

    def __init__(self, folder: Path, *, fps: float, title: str, subtitle: str, width: int = VIDEO_WIDTH,
                 preroll_s: float = PREROLL_S, hold_s: float = RESULT_HOLD_S, queue_frames: int = QUEUE_FRAMES,
                 stale_frame: Optional[np.ndarray] = None, work_dir: Optional[Path] = None) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.folder = Path(folder)
        self.work_dir = Path(work_dir) if work_dir is not None else self.folder
        self.fps = float(fps)
        self.title, self.subtitle = str(title), str(subtitle)
        self.width = int(width)
        self._preroll: deque = deque(maxlen=max(0, int(round(preroll_s * self.fps))))
        self._hold_frames = max(1, int(round(hold_s * self.fps)))
        self._queue: "queue.Queue[tuple[Any, ...]]" = queue.Queue(maxsize=max(1, int(queue_frames)))
        self._stale = stale_frame
        self._closed = False
        self.started = False  # the first move reached the encoder (a file is being written)
        self.frames = 0  # frames written to the file
        self.dropped = 0  # frames the encoder could not take in time
        self.path: Optional[Path] = None  # the finished file
        self.error: Optional[str] = None
        self._temporary = self.work_dir / "_recording_{}_{}.mp4".format(os.getpid(), next(_ids))
        self._thread = threading.Thread(target=self._run, name="attempt-video", daemon=True)
        self._thread.start()

    # ----------------------------------------------------- control loop side
    def push(self, frame: Optional[np.ndarray], elapsed_s: Optional[float]) -> None:
        if self._closed or frame is None or frame is self._stale:
            return
        self._stale = None  # the window has drawn this attempt's scene
        try:
            self._queue.put_nowait(("frame", frame, None if elapsed_s is None else float(elapsed_s)))
        except queue.Full:
            self.dropped += 1

    def finish(self, result: str, time_s: Optional[float], note: str, file_name: str) -> None:
        """End with ``result`` (success / failed / quit) shown for ``RESULT_HOLD_S``; saved as ``file_name``."""

        if not self._closed:
            self._closed = True
            self._queue.put(("finish", str(result), time_s, str(note), str(file_name)))

    def discard(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(("discard",))

    def wait(self, timeout_s: Optional[float] = FINISH_TIMEOUT_S) -> Optional[Path]:
        self._thread.join(timeout_s)
        return self.path

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    # ------------------------------------------------------------ the thread
    def _run(self) -> None:
        import cv2

        pipe: Optional[_FfmpegPipe] = None
        geometry: Optional[tuple[int, int, int]] = None
        canvas: Optional[np.ndarray] = None
        last_body: Optional[np.ndarray] = None
        try:
            while True:
                item = self._queue.get()
                kind = item[0]
                if kind == "frame":
                    frame, elapsed = item[1], item[2]
                    if geometry is None:
                        geometry = video_geometry(frame.shape[1], frame.shape[0], self.width)
                        canvas = np.empty((geometry[1] + geometry[2], geometry[0], 3), dtype=np.uint8)
                    out_w, body_h, header_h = geometry
                    body = frame if frame.shape[:2] == (body_h, out_w) else cv2.resize(
                        frame, (out_w, body_h), interpolation=cv2.INTER_AREA)
                    last_body = body
                    if elapsed is None and pipe is None:
                        self._preroll.append(body)
                        continue
                    if pipe is None:
                        pipe = self._start(cv2, geometry, canvas)
                    self._write(cv2, pipe, canvas, body, header_h, 0.0 if elapsed is None else elapsed)
                elif kind == "finish":
                    _kind, result, time_s, note, name = item
                    if pipe is None and last_body is not None and geometry is not None and canvas is not None:
                        pipe = self._start(cv2, geometry, canvas)  # ended before a started frame got through
                    if pipe is not None and last_body is not None and canvas is not None and geometry is not None:
                        final = compose_frame(cv2, canvas, last_body, geometry[2], title=self.title,
                                              subtitle=self.subtitle, result=(result, time_s, note))
                        for _ in range(self._hold_frames):
                            pipe.write(final)
                            self.frames += 1
                        pipe.close()
                        pipe = None
                        self.folder.mkdir(parents=True, exist_ok=True)
                        target = self.folder / name
                        try:
                            os.replace(self._temporary, target)
                        except OSError:  # work_dir on another drive
                            shutil.move(str(self._temporary), str(target))
                        self.path = target
                    break
                else:  # discard
                    break
        except Exception as exc:
            self.error = "{}: {}".format(type(exc).__name__, exc)
        finally:
            self._preroll.clear()
            if pipe is not None:
                pipe.kill()
            if self.path is None:
                try:
                    self._temporary.unlink()
                except OSError:
                    pass

    def _start(self, cv2: Any, geometry: tuple[int, int, int], canvas: np.ndarray) -> _FfmpegPipe:
        """Open the file and write the kept second before the first move (READY in the header)."""

        out_w, body_h, header_h = geometry
        self.work_dir.mkdir(parents=True, exist_ok=True)
        pipe = _FfmpegPipe(self._temporary, (out_w, body_h + header_h), self.fps)
        self.started = True
        for old in self._preroll:
            self._write(cv2, pipe, canvas, old, header_h, None)
        self._preroll.clear()
        return pipe

    def _write(self, cv2: Any, pipe: _FfmpegPipe, canvas: np.ndarray, body: np.ndarray, header_h: int,
               elapsed: Optional[float]) -> None:
        pipe.write(compose_frame(cv2, canvas, body, header_h, title=self.title, subtitle=self.subtitle,
                                 elapsed_s=elapsed))
        self.frames += 1
