"""Guided SpaceMouse direction calibration (about 40 seconds, needs the device).

Hold the SpaceMouse the way you normally use it (logo facing you). The tool
asks for six gestures, measures which HID axis and sign each one produces,
and writes ``axis_order`` / ``axis_inversion`` into the shared config used by
every robosuite and LIBERO entry. The previous config is backed up first.

    .\\.venv-robosuite\\Scripts\\python.exe .\\scripts\\calibrate_spacemouse_axes.py
    .\\.venv-robosuite\\Scripts\\python.exe .\\scripts\\calibrate_spacemouse_axes.py --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.axis_calibration import AXIS_GESTURES, measure_gesture, solve_axis_mapping  # noqa: E402
from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.device import SpaceMouseHid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spacemouse_wrist.json")
    parser.add_argument("--hold-s", type=float, default=2.5, help="seconds to record each held gesture")
    parser.add_argument("--dry-run", action="store_true", help="measure and print, but do not write the config")
    return parser.parse_args()


def _wait_neutral(device: SpaceMouseHid, timeout_s: float = 15.0) -> None:
    print("请松开 SpaceMouse，保持静止 2 秒（零偏校准）……")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample = device.poll()
        if sample.armed:
            print("零偏校准完成。")
            return
        time.sleep(0.01)
    raise RuntimeError("没有完成零偏校准：请确认 SpaceMouse 已连接且松手静止")


def _record(device: SpaceMouseHid, hold_s: float) -> np.ndarray:
    samples = []
    end = time.monotonic() + hold_s
    while time.monotonic() < end:
        sample = device.poll()
        if sample.valid and sample.armed:
            samples.append(np.asarray(sample.control_axes, dtype=np.float64))
        time.sleep(0.01)
    return np.asarray(samples) if samples else np.zeros((1, 6))


def _wait_release(device: SpaceMouseHid) -> None:
    quiet_since = None
    while True:
        sample = device.poll()
        moving = sample.valid and np.max(np.abs(sample.control_axes)) > 30.0
        now = time.monotonic()
        if moving:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        elif now - quiet_since > 0.6:
            return
        time.sleep(0.01)


def main() -> int:
    args = parse_args()
    config = WristConfig.load(args.config)
    device = SpaceMouseHid(config)
    info = device.open()
    print("已连接:", info.get("product_string"), "| 当前 axis_order", list(config.axis_order),
          "axis_inversion", list(config.axis_inversion))
    measurements = []
    try:
        _wait_neutral(device)
        for index, gesture in enumerate(AXIS_GESTURES, start=1):
            while True:
                print(f"\n[{index}/6] {gesture.prompt}")
                for count in (3, 2, 1):
                    print(f"  {count}…", end="", flush=True)
                    time.sleep(0.6)
                print("  记录中")
                samples = _record(device, args.hold_s)
                print("  好，请松手")
                _wait_release(device)
                try:
                    item = measure_gesture(gesture, samples)
                except ValueError as exc:
                    print("  需要重做:", exc)
                    continue
                print(f"  -> HID 轴 {item.raw_axis} 符号 {item.raw_sign:+d}，纯度 {item.purity:.2f}")
                if item.purity < 0.75:
                    print("  动作混入了其他轴，请只做这一个方向，再来一次")
                    continue
                measurements.append(item)
                break
    finally:
        device.close()

    order, inversion = solve_axis_mapping(measurements)
    print("\n结果: axis_order =", list(order), " axis_inversion =", list(inversion))
    if args.dry_run:
        print("--dry-run：未写入配置。")
        return 0
    backup_dir = ROOT / "outputs" / "spacemouse_axis_calibration"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy2(args.config, backup_dir / f"{stamp}_before_{args.config.name}")
    data = json.loads(args.config.read_text(encoding="utf-8-sig"))
    data["axis_order"] = list(order)
    data["axis_inversion"] = list(inversion)
    WristConfig.from_dict(data)  # validate before writing
    args.config.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (backup_dir / f"{stamp}_measurements.json").write_text(json.dumps([
        {"gesture": m.gesture.key, "raw_axis": m.raw_axis, "raw_sign": m.raw_sign,
         "purity": m.purity, "peak": m.peak} for m in measurements
    ], indent=2), encoding="utf-8")
    print("已写入", args.config, "（备份在", backup_dir, "）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
