"""Subset the self-hosted web fonts (run when the site gains new characters): `make fonts`.

Sources (git-ignored, downloaded from github.com/IBM/plex, SIL OFL 1.1):
    web/fonts_src/ibm-plex-sans-thai-looped/IBMPlexSansThaiLooped-{Regular,Medium,SemiBold}.ttf
    web/fonts_src/ibm-plex-mono/IBMPlexMono-{Regular,Medium,SemiBold}.ttf  (+ OFL.txt in each)
Output (committed, copied into web/site by build_web):
    web/src/fonts/fraudthai-{400,500,600}.woff2 + OFL.txt
    web/src/fonts/fraudmono-{400,500,600}.woff2 + OFL-FraudMono.txt

FraudThai keeps the Thai block, printable ASCII and every other character th.json uses.
FraudMono keeps printable ASCII and every non-Thai character the site's text and scripts use.
A subset is a modified version, so under the OFL's Reserved Font Name rule it must not be
called "Plex": the families are renamed; the copyright notice is kept as required.
"""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

REPO = Path(__file__).resolve().parents[1]
FONTS_SRC = REPO / "web" / "fonts_src"
WEB_SRC = REPO / "web" / "src"
OUT = WEB_SRC / "fonts"
WEIGHTS = {400: "Regular", 500: "Medium", 600: "SemiBold"}
THAI = range(0x0E00, 0x0E80)
ASCII = range(0x20, 0x7F)
SITE_TEXT = ["i18n/en.json", "i18n/th.json", "index.html.tmpl", "app.js", "viz.js", "style.css"]


@dataclass(frozen=True)
class Family:
    name: str  # family name in the subset (never "Plex")
    slug: str  # file prefix
    src_dir: str
    src_prefix: str
    licence: str  # licence file name next to the woff2 files


THAI_FAMILY = Family(
    "FraudThai", "fraudthai", "ibm-plex-sans-thai-looped", "IBMPlexSansThaiLooped", "OFL.txt"
)
MONO_FAMILY = Family("FraudMono", "fraudmono", "ibm-plex-mono", "IBMPlexMono", "OFL-FraudMono.txt")
FAMILIES = [THAI_FAMILY, MONO_FAMILY]


def chars(text: str) -> set[int]:
    """Code points as rendered: HTML entities such as &ge; count as their character."""
    return {ord(ch) for ch in html.unescape(text)}


def site_chars() -> set[int]:
    return set().union(*(chars((WEB_SRC / name).read_text()) for name in SITE_TEXT))


def wanted_codepoints(family: Family = THAI_FAMILY) -> set[int]:
    if family is THAI_FAMILY:
        th = json.loads((WEB_SRC / "i18n" / "th.json").read_text())
        return set(THAI) | set(ASCII) | set().union(*(chars(value) for value in th.values()))
    return set(ASCII) | {cp for cp in site_chars() if cp >= 0x20 and cp not in THAI}


def rename(font: TTFont, family: str, style: str) -> None:
    """Family/full/PostScript names -> the new family; copyright and licence records are kept."""
    names = {
        1: family,
        2: "Regular",
        3: f"{family}-{style};subset",
        4: f"{family} {style}",
        6: f"{family}-{style}",
        16: family,
        17: style,
    }
    table = font["name"]
    for rec in list(table.names):
        if rec.nameID in names or rec.nameID in (7, 8, 9, 10, 11, 12, 18, 21, 22, 25):
            table.removeNames(nameID=rec.nameID)
    for name_id, value in names.items():
        table.setName(value, name_id, 3, 1, 0x409)
    if "CFF " in font:
        font["CFF "].cff.fontNames = [f"{family}-{style}"]


def build(family: Family) -> None:
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["*"]
    opts.name_IDs = ["*"]
    opts.name_languages = ["*"]
    opts.drop_tables += ["DSIG"]
    src = FONTS_SRC / family.src_dir
    unicodes = sorted(wanted_codepoints(family))
    for weight, style in WEIGHTS.items():
        font = TTFont(src / f"{family.src_prefix}-{style}.ttf")
        sub = subset.Subsetter(opts)
        sub.populate(unicodes=unicodes)
        sub.subset(font)
        rename(font, family.name, style)
        font.flavor = "woff2"
        out = OUT / f"{family.slug}-{weight}.woff2"
        font.save(out)
        kept = len(font.getBestCmap())
        print(f"{out.relative_to(REPO)}: {out.stat().st_size / 1024:.1f} KB, {kept} code points")
    shutil.copy(src / "OFL.txt", OUT / family.licence)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        build(family)


if __name__ == "__main__":
    main()
