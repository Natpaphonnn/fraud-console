"""Subset the Thai web font (run when web/src/i18n/th.json gains new characters).

    uv run python scripts/build_fonts.py

Source: web/fonts_src/ibm-plex-sans-thai-looped/ (IBM Plex Sans Thai Looped, SIL OFL 1.1).
Output: web/src/fonts/fraudthai-{400,500,600}.woff2 + OFL.txt (copied into web/site by build_web).

Kept glyphs: the Thai block, printable ASCII and every other character used in th.json.
The subset is a modified version, so under the OFL's Reserved Font Name rule it must not be
called "Plex": its family is renamed to "FraudThai"; the copyright notice is kept as required.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "web" / "fonts_src" / "ibm-plex-sans-thai-looped"
OUT = REPO / "web" / "src" / "fonts"
FAMILY = "FraudThai"
WEIGHTS = {400: "Regular", 500: "Medium", 600: "SemiBold"}
THAI = range(0x0E00, 0x0E80)
ASCII = range(0x20, 0x7F)


def wanted_codepoints() -> set[int]:
    th = json.loads((REPO / "web" / "src" / "i18n" / "th.json").read_text())
    used = {ord(ch) for value in th.values() for ch in value}
    return set(THAI) | set(ASCII) | used


def rename(font: TTFont, style: str) -> None:
    """Family/full/PostScript names -> FraudThai; copyright and licence records are kept."""
    names = {
        1: FAMILY,
        2: style if style == "Regular" else "Regular",
        3: f"{FAMILY}-{style};subset",
        4: f"{FAMILY} {style}",
        6: f"{FAMILY}-{style}",
        16: FAMILY,
        17: style,
    }
    table = font["name"]
    for rec in list(table.names):
        if rec.nameID in names or rec.nameID in (7, 8, 9, 10, 11, 12, 18, 21, 22, 25):
            table.removeNames(nameID=rec.nameID)
    for name_id, value in names.items():
        table.setName(value, name_id, 3, 1, 0x409)
    if "CFF " in font:
        font["CFF "].cff.fontNames = [f"{FAMILY}-{style}"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    unicodes = sorted(wanted_codepoints())
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["*"]
    opts.name_IDs = ["*"]
    opts.name_languages = ["*"]
    opts.drop_tables += ["DSIG"]
    for weight, style in WEIGHTS.items():
        font = TTFont(SRC / f"IBMPlexSansThaiLooped-{style}.ttf")
        sub = subset.Subsetter(opts)
        sub.populate(unicodes=unicodes)
        sub.subset(font)
        rename(font, style)
        font.flavor = "woff2"
        out = OUT / f"fraudthai-{weight}.woff2"
        font.save(out)
        print(f"{out.relative_to(REPO)}: {out.stat().st_size / 1024:.1f} KB")
    shutil.copy(SRC / "OFL.txt", OUT / "OFL.txt")
    print(f"{len(unicodes)} code points kept")


if __name__ == "__main__":
    main()
