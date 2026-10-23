"""Small non-physical MuJoCo video recorder shared by interactive entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class SimVideoRecorder:
    """Record a rendered camera stream to MP4, with a PNG fallback.

    Recording never changes MuJoCo state.  It is deliberately optional so an
    unavailable OpenGL context or video codec cannot interrupt teleoperation.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        fps: float,
        camera_name: str = "frontview",
        width: int = 960,
        height: int = 720,
    ) -> None:
        self.requested_path = None if path is None else Path(path)
        self.fps = float(fps)
        self.camera_name = str(camera_name)
        self.width = int(width)
        self.height = int(height)
        self.mode = "disabled" if path is None else "pending"
        self.output_path: Path | None = None
        self.frame_count = 0
        self.error: str | None = None
        self._writer: Any = None

    def capture(self, env: Any) -> None:
        if self.requested_path is None or self.mode == "unavailable":
            return
        try:
            import cv2

            image = np.asarray(
                env.sim.render(
                    width=self.width,
                    height=self.height,
                    camera_name=self.camera_name,
                )
            )
            if image.shape[:2] != (self.height, self.width) or image.ndim != 3 or image.shape[2] < 3:
                raise RuntimeError(f"unexpected recording frame shape {image.shape}")
            bgr = np.ascontiguousarray(image[::-1, :, :3][:, :, ::-1])
            if self.mode == "pending":
                self.requested_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(self.requested_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (self.width, self.height),
                )
                if writer.isOpened():
                    self._writer = writer
                    self.output_path = self.requested_path
                    self.mode = "mp4"
                else:
                    writer.release()
                    self.output_path = self.requested_path.with_suffix("").with_name(
                        f"{self.requested_path.stem}_frames"
                    )
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    self.mode = "frames"
            if self.mode == "mp4":
                self._writer.write(bgr)
            elif self.mode == "frames":
                frame = self.output_path / f"{self.frame_count:06d}.png"
                if not cv2.imwrite(str(frame), bgr):
                    raise RuntimeError("cv2.imwrite returned false")
            self.frame_count += 1
        except Exception as exc:  # host codecs / OpenGL are external resources
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            self.error = f"{type(exc).__name__}: {exc}"
            self.mode = "unavailable"

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def summary(self) -> dict[str, Any]:
        return {
            "requested_path": None if self.requested_path is None else str(self.requested_path),
            "mode": self.mode,
            "output_path": None if self.output_path is None else str(self.output_path),
            "frame_count": self.frame_count,
            "error": self.error,
            "fps": self.fps,
            "camera_name": self.camera_name,
            "width": self.width,
            "height": self.height,
        }
