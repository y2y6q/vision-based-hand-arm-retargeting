"""Inspect, install, and restore conservative 3DxWare C62E profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.threedxware import (  # noqa: E402
    DEFAULT_TARGET_EXECUTABLES,
    ThreeDxWarePaths,
    build_c62e_profile,
    inspect_3dxware,
    install_profiles,
    restore_profiles,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manage user-local 3DxWare profiles for SpaceMouse Wireless 256F:C62E. "
            "The tool never stops or restarts 3DxWare services."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="read driver paths and active profile state")
    inspect.add_argument("--report", type=Path, help="optional JSON output path")
    inspect.add_argument(
        "--target-exe", action="append", dest="target_exes",
        help="executable basename to include in the read-only report",
    )
    render = subparsers.add_parser("render", help="print a profile without writing 3DxWare")
    render.add_argument("--target-exe", required=True, help="executable basename, for example python.exe")
    install = subparsers.add_parser("install", help="back up and install explicit user-local profiles")
    install.add_argument("--target-exe", action="append", dest="target_exes", required=True)
    install.add_argument(
        "--backup-root", type=Path, default=ROOT / "outputs" / "3dxware_profiles" / "backups",
        help="directory where the restore manifest and original profile are kept",
    )
    install.add_argument("--allow-overwrite", action="store_true", help="back up then replace a tool profile")
    restore = subparsers.add_parser("restore", help="restore a manifest made by install")
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--force", action="store_true", help="allow replacing a profile edited after install")
    return parser


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = ThreeDxWarePaths.discover()
    if args.command == "inspect":
        report = inspect_3dxware(paths, args.target_exes or DEFAULT_TARGET_EXECUTABLES)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.report:
            _write_json(args.report, report)
            print(f"Wrote read-only report: {args.report}")
        return 0
    if args.command == "render":
        print(build_c62e_profile(args.target_exe), end="")
        return 0
    if args.command == "install":
        manifest = install_profiles(
            args.target_exes,
            paths=paths,
            backup_root=args.backup_root,
            allow_overwrite=args.allow_overwrite,
        )
        print(f"Installed user-local 3DxWare profile(s). Restore manifest: {manifest}")
        print("Refocus the target application, then verify raw HID and virtual-input behavior manually.")
        return 0
    if args.command == "restore":
        restored = restore_profiles(args.manifest, paths=paths, force=args.force)
        for path in restored:
            print(f"Restored: {path}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
