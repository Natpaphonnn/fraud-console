"""Website languages: same keys, no typed numbers, same numbers in both languages, font rules."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from fraud_e2e.summary import build_context
from fraud_e2e.web import LANGS, SITE, SRC, dictionaries, load_dict, localized_ctx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_fonts import wanted_codepoints  # noqa: E402

BUILD_PH = re.compile(r"\{\{\s*(\w+)\s*\}\}")
RUNTIME_PH = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")
# names and identifiers that contain digits but are not numbers we report
ALLOWED_DIGIT_TOKENS = [
    "Credit Card Fraud Detection 2026",
    "CC0 1.0",
    "CC0",
    "3-D Secure",
    "sha256",
    "r0",
    "Fraud-E2E",
]
NUMBER = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?%?")


@pytest.fixture(scope="module")
def src():
    return {lang: load_dict(lang) for lang in LANGS}


@pytest.fixture(scope="module")
def built():
    return dictionaries()


def test_both_languages_have_the_same_keys(src):
    assert set(src["en"]) == set(src["th"])


def test_no_hand_typed_numbers_in_dictionaries(src):
    for lang, d in src.items():
        for key, text in d.items():
            rest = BUILD_PH.sub("", RUNTIME_PH.sub("", text))
            for token in ALLOWED_DIGIT_TOKENS:
                rest = rest.replace(token, "")
            rest = re.sub(r"<[^>]+>", "", rest)  # markup (e.g. href="#...") is not prose
            assert not re.search(r"[0-9๐-๙]", rest), f"{lang}:{key} has a typed number: {rest!r}"


def test_placeholders_match_between_languages(src):
    for key in src["en"]:
        for ph in (BUILD_PH, RUNTIME_PH):
            assert sorted(ph.findall(src["en"][key])) == sorted(ph.findall(src["th"][key])), key


def test_numbers_are_identical_after_switching_language(built):
    for key, en in built["en"].items():
        nums = lambda s: Counter(NUMBER.findall(re.sub(r"<[^>]+>", " ", s)))  # noqa: E731
        en_n, th_n = nums(en), nums(built["th"][key])
        for token in ALLOWED_DIGIT_TOKENS:
            for n in NUMBER.findall(token):
                en_n[n] -= 1 if en_n[n] else 0
                th_n[n] -= 1 if th_n[n] else 0
        assert +en_n == +th_n, f"{key}: {en_n - th_n} / {th_n - en_n}"


def test_english_phrases_match_the_readme_context():
    base = build_context()
    for k, v in localized_ctx(load_dict("en")).items():
        if k in base:
            assert v == base[k], k


def test_every_key_used_by_the_page_and_scripts_exists(src):
    keys = set(src["en"])
    page = (SRC / "index.html.tmpl").read_text()
    used = set(re.findall(r"\[\[([\w.]+)\]\]", page)) | set(
        re.findall(r'data-i18n="([\w.]+)"', page)
    )
    used |= {
        k
        for pair in re.findall(r'data-i18n-attr="([^"]+)"', page)
        for k in re.findall(r":([\w.]+)", pair)
    }
    for js in ("app.js", "viz.js"):
        text = (SRC / js).read_text()
        used |= set(re.findall(r"""\bt\(\s*["']([\w.]+)["']""", text))
        used |= {f"js.pal.{k}" for k in re.findall(r'\["(?:section|report|action)", "(\w+)"', text)}
    used |= {f"js.pal.group.{g}" for g in ("section", "report", "action")}
    used |= {f"js.pg.group.{g}" for g in ("transaction", "card", "risk", "device", "other")}
    used |= {f"js.flow.{o}" for o in ("tp", "fp", "fn", "tn")} | {
        f"js.out.{o}" for o in ("tp", "fp", "fn", "tn")
    }
    assert not used - keys, sorted(used - keys)


def test_built_dictionaries_are_in_the_site():
    for lang in LANGS:
        assert json.loads((SITE / "i18n" / f"{lang}.json").read_text()) == dictionaries()[lang]


def test_caveat_labels_are_translated(built):
    for key in ("flags.data", "flags.cost"):
        assert built["th"][key] != built["en"][key] and re.search(r"[฀-๿]", built["th"][key])


@pytest.mark.parametrize("weight", [400, 500, 600])
def test_thai_font_subset(weight, src):
    path = SRC / "fonts" / f"fraudthai-{weight}.woff2"
    font = TTFont(path)
    for rec in font["name"].names:  # Reserved Font Name: the modified font is not called "Plex"
        if rec.nameID in (1, 3, 4, 6, 16, 17, 21):
            assert "plex" not in rec.toUnicode().lower(), (rec.nameID, rec.toUnicode())
    assert font["name"].getDebugName(1) == "FraudThai"
    cmap = font.getBestCmap()
    source = TTFont(
        REPO / "web/fonts_src/ibm-plex-sans-thai-looped/IBMPlexSansThaiLooped-Regular.ttf"
    )
    has = source.getBestCmap()  # characters the original font has (others fall back, e.g. ↵)
    chars = {ch for text in src["th"].values() for ch in text if ord(ch) >= 0x20 and ord(ch) in has}
    assert {ch for ch in chars if ord(ch) not in cmap} == set()
    assert set(cmap) <= wanted_codepoints()  # Thai block, ASCII and characters th.json uses
    assert "plex" not in path.name.lower()
    for folder in (SRC / "fonts", SITE / "fonts"):
        assert (folder / "OFL.txt").read_text().startswith("Copyright")
