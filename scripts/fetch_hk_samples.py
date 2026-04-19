"""Fetch HK Buildings Department sample defect photos for LOCAL demo use.

The HK Buildings Department publishes a small set of defect reference
photos on its "Building Defects" page. These images are **Crown
Copyright of the HKSAR government** — we are not allowed to
redistribute them in this repository, but a user is free to download
them to their own machine for **private use** such as an interview
demo or a classroom lecture.

This script formalises that: run it locally, it downloads the images
into ``sample_images/hk_*.jpg``, which are covered by a ``.gitignore``
entry so they never get pushed to the repo. No secret URL lists, no
scraping beyond the five publicly visible thumbnails on the BD page.

Usage::

    python -m scripts.fetch_hk_samples --accept-license

The ``--accept-license`` flag is required to confirm you have read the
Terms of Use notice printed by this script and will use the images
for private demo purposes only.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_images"

BD_PAGE_URL = (
    "https://www.bd.gov.hk/en/safety-inspection/building-safety/"
    "index_bsi_defects.html"
)

BASE_IMAGE_URL = (
    "https://www.bd.gov.hk/images/safety-and-inspection/building-safety"
)


@dataclass(frozen=True)
class BDSample:
    """A single public thumbnail advertised on the BD defects page."""

    remote: str
    local: str
    caption: str


BD_SAMPLES: List[BDSample] = [
    BDSample(
        remote="05defects_thumb01.jpg",
        local="hk_bd_overview.jpg",
        caption="HK BD — Building Defects landing photo",
    ),
    BDSample(
        remote="05defects_thumb02.jpg",
        local="hk_bd_ns_crack.jpg",
        caption="HK BD — Non-structural cracks",
    ),
    BDSample(
        remote="05defects_thumb03.jpg",
        local="hk_bd_spalling.jpg",
        caption="HK BD — Spalling of concrete",
    ),
    BDSample(
        remote="05defects_thumb04.jpg",
        local="hk_bd_struct_crack.jpg",
        caption="HK BD — Structural cracks",
    ),
    BDSample(
        remote="05defects_thumb05.jpg",
        local="hk_bd_wall_finish.jpg",
        caption="HK BD — Defective external wall finishes",
    ),
]


LICENSE_NOTICE = f"""
============================================================
HK Buildings Department — Building Defects reference photos
============================================================

Source:
    {BD_PAGE_URL}

Copyright:
    (c) Hong Kong SAR Government. All photographs on the BD
    website are protected by Crown Copyright. See the BD
    "Copyright Notice" and "Disclaimer" pages for the full
    terms (https://www.bd.gov.hk/en/footer/copyright-notice/).

What this script does:
    Downloads the 5 publicly-advertised defect thumbnails into
    ``sample_images/hk_bd_*.jpg``. These files are covered by a
    ``.gitignore`` entry so they never get committed or pushed.

What you MAY do with the downloaded copies:
    - Run the Streamlit demo on your own laptop during an
      interview, presentation, or classroom session.
    - Keep a local copy for your personal reference.

What you MUST NOT do:
    - Commit these files to git or push them to GitHub / any
      other public host.
    - Redistribute, sell, or include them in another dataset.
    - Remove the BD attribution when displaying them.

If those constraints are unacceptable, delete the files after
your demo. The segmenter + classifier weights are unchanged —
you do NOT need these images for the model to work; they are
only for demo realism during an HK-specific pitch.
============================================================
""".strip()


def _download_one(session: requests.Session, sample: BDSample, force: bool) -> str:
    dst = SAMPLE_DIR / sample.local
    if dst.exists() and not force:
        return f"skip (exists): {sample.local}"
    url = f"{BASE_IMAGE_URL}/{sample.remote}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    content = resp.content
    if not content.startswith(b"\xff\xd8"):
        raise RuntimeError(
            f"Refusing to write non-JPEG content from {url} "
            f"(got {content[:8]!r})"
        )
    dst.write_bytes(content)
    return f"saved: {sample.local} ({len(content):,} bytes) — {sample.caption}"


def _print_license() -> None:
    print(LICENSE_NOTICE)
    print()


def _print_list() -> None:
    print(f"{'local filename':30s}  caption")
    print("-" * 72)
    for s in BD_SAMPLES:
        print(f"{s.local:30s}  {s.caption}")
    print()
    print(f"Destination: {SAMPLE_DIR.relative_to(ROOT)}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download HK Buildings Department defect thumbnails into "
            "sample_images/ for LOCAL interview demo use only. Files "
            "are gitignored and will not be pushed."
        )
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help=(
            "Confirm you have read the BD copyright notice and will use "
            "the downloaded files for private demo only. Required."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a local copy already exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just print what would be downloaded and exit.",
    )
    args = parser.parse_args(argv)

    _print_license()

    if args.list:
        _print_list()
        return 0

    if not args.accept_license:
        print(
            "Refusing to download without --accept-license. Re-run with "
            "the flag if you accept the terms above.",
            file=sys.stderr,
        )
        return 2

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "crack-detection-demo/1.0 (+https://github.com/) "
                    "local-demo fetch script"
                )
            }
        )
        errors: List[str] = []
        for sample in BD_SAMPLES:
            try:
                print(_download_one(session, sample, force=args.force))
            except Exception as exc:  # pragma: no cover - network-facing
                errors.append(f"{sample.local}: {exc}")
                print(f"FAILED: {sample.local}: {exc}", file=sys.stderr)

    print()
    print(f"Done. {len(BD_SAMPLES) - len(errors)}/{len(BD_SAMPLES)} downloaded.")
    print(
        "Reminder: these files are gitignored. Do not add / commit them "
        "even if git status shows them as untracked."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
