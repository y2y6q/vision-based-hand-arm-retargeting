"""Read-only SHA-256 comparison for the frozen PyBullet / panda-gym subset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "outputs" / "spacemouse_implementation_20260907_144907" / "baseline_before.json"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _frozen_relative(path: Path) -> bool:
    return (
        path == Path("scripts/run_camera_panda_allegro_teleop.py")
        or path.is_relative_to(Path("src/teleop"))
        or path.is_relative_to(Path("third_party/panda-gym"))
    )


def compare(baseline_path: Path) -> dict[str, object]:
    entries = json.loads(baseline_path.read_text(encoding="utf-8"))
    files = []
    for entry in entries:
        absolute = Path(entry["Path"])
        relative = absolute.relative_to(ROOT)
        if not _frozen_relative(relative):
            continue
        actual = _sha256(absolute)
        expected = str(entry["Hash"]).upper()
        files.append({
            "path": relative.as_posix(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
        })
    changed = [entry for entry in files if not entry["match"]]
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": str(baseline_path),
        "scope": [
            "scripts/run_camera_panda_allegro_teleop.py",
            "src/teleop",
            "third_party/panda-gym",
        ],
        "excluded_authorized_change": "README.md is intentionally outside this frozen-code comparison.",
        "checked_files": len(files),
        "unchanged_files": len(files) - len(changed),
        "changed_files": len(changed),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "frozen_baseline_hash_comparison_current.json")
    args = parser.parse_args()
    report = compare(args.baseline)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("checked_files", "unchanged_files", "changed_files")}, indent=2))
    print(f"report={args.report}")
    return 0 if report["changed_files"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
