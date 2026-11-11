"""Make videos, LIBERO-format datasets and the summary from collected demos; copy them to a share folder.

Reads the raw demos the collection entry (scripts/run_libero_collect.py)
writes only with ``--save-demos``, in outputs/libero_collect_demos/<suite>/
(the default ``--output-root``); the entry itself no longer runs this. Run it
by hand when you need datasets, e.g. at another image size. Only missing videos
are made (``--force`` remakes all); a task's dataset is made once it has
``--demos-per-task`` demos (``--partial``: any task with demos) and remade
when its demos change. Images are rendered from the saved states; nothing is
re-simulated.

``--share-dir``: a folder your cloud client syncs (OneDrive, Google Drive
for desktop). Videos, SUMMARY.md and summary.csv are copied to
``<share-dir>/<suite>/``; share that folder once and post its link.

Run only with ``.venv-libero``. See docs/libero_data_collection_zh.md.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.libero_collection import DEMO_ROOT, DEMOS_PER_TASK, IMAGE_SIZE  # noqa: E402
from wrist_teleop.libero_dataset_build import build_suite  # noqa: E402
from wrist_teleop.libero_spatial_teleop import LIBERO_SUITES, configure_local_libero_paths  # noqa: E402


def parse_tasks(text: str) -> list[int] | None:
    if text.strip().lower() in ("", "all"):
        return None
    tasks: list[int] = []
    for part in text.split(","):
        first, dash, last = part.strip().partition("-")
        tasks.extend(range(int(first), int(last if dash else first) + 1))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=LIBERO_SUITES, default="libero_object")
    parser.add_argument("--output-root", type=Path, default=DEMO_ROOT,
                        help="where --save-demos wrote the raw demos (default outputs/libero_collect_demos)")
    parser.add_argument("--tasks", default="all", help="all (default), 3, 0-4 or 1,3,7-9")
    parser.add_argument("--demos-per-task", type=int, default=DEMOS_PER_TASK)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="dataset image size (default 256)")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--no-datasets", action="store_true")
    parser.add_argument("--partial", action="store_true", help="also make datasets of tasks that are not complete")
    parser.add_argument("--exclude-disturbed", action="store_true",
                        help="leave demos that moved / tipped other objects out of the datasets")
    parser.add_argument("--force", action="store_true", help="remake videos and datasets that already exist")
    parser.add_argument("--share-dir", type=Path, default=os.environ.get("LIBERO_SHARE_DIR") or None,
                        help="synced cloud folder (default: env LIBERO_SHARE_DIR)")
    parser.add_argument("--share-url", default=os.environ.get("LIBERO_SHARE_URL") or None,
                        help="its share link, put into the Slack message (default: env LIBERO_SHARE_URL)")
    parser.add_argument("--share-datasets", action="store_true", help="copy the datasets to the share folder too")
    args = parser.parse_args()
    if not 32 <= args.image_size <= 1024:
        parser.error("--image-size must be within 32-1024")
    try:
        tasks = parse_tasks(args.tasks)
    except ValueError:
        parser.error("--tasks takes all, N, N-M or a comma list")
    configure_local_libero_paths()  # MUJOCO_GL as in the teleop entries
    started = time.perf_counter()
    result = build_suite(
        args.output_root, args.suite, tasks=tasks, demos_per_task=args.demos_per_task, image_size=args.image_size,
        videos=not args.no_videos, datasets=not args.no_datasets, partial=args.partial,
        exclude_disturbed=args.exclude_disturbed, force=args.force, share_dir=args.share_dir,
        share_url=args.share_url, share_datasets=args.share_datasets,
    )
    print("made {} videos, {} datasets in {:.0f} s".format(
        len(result["videos"]), len(result["datasets"]), time.perf_counter() - started))
    print("summary: {}".format(result["suite_root"] / "SUMMARY.md"))
    if result["share_dir"] is not None:
        print("copied {} files to {} (the sync client uploads them)".format(len(result["shared"]), result["share_dir"]))
    print("\nSlack message:\n" + result["message"])
    if not args.share_url:
        print("(add the shared folder's link with --share-url or LIBERO_SHARE_URL to include it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
