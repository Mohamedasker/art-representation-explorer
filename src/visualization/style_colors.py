"""
A fixed colour for every art style, shared by all pages.

The colour encodes where the style sits on the art-history timeline: older styles are darker, newer ones lighter
(CIE L* grows with the approximate year the style emerged), and the hue steps by the golden angle through the
chronological order, so styles that are neighbours in time still get clearly different hues.
"""

from __future__ import annotations

import math

# (style, approximate year of emergence), chronological; names as used by ArtBench-10 and WikiArt
TIMELINE: list[tuple[str, int]] = [
    ("early_renaissance", 1430), ("northern_renaissance", 1450), ("renaissance", 1480), ("high_renaissance", 1500),
    ("mannerism_late_renaissance", 1530), ("baroque", 1600), ("rococo", 1730), ("ukiyo_e", 1750),
    ("romanticism", 1790), ("realism", 1840), ("impressionism", 1867), ("pointillism", 1886),
    ("post_impressionism", 1886), ("symbolism", 1886), ("art_nouveau", 1890), ("fauvism", 1905),
    ("expressionism", 1905), ("analytical_cubism", 1909), ("cubism", 1909), ("synthetic_cubism", 1912),
    ("naive_art_primitivism", 1915), ("surrealism", 1924), ("abstract_expressionism", 1943),
    ("action_painting", 1947), ("color_field_painting", 1950), ("new_realism", 1960), ("pop_art", 1956),
    ("minimalism", 1960), ("contemporary_realism", 1970),
]
_YEAR0, _YEAR1 = 1430, 1970
_L_OLD, _L_NEW = 48.0, 80.0         # darkest (oldest) .. lightest (newest); kept readable on the dark page background


def _lch_to_hex(L: float, C: float, h_deg: float, with_lab: bool = False):
    h = math.radians(h_deg)
    a, b = C * math.cos(h), C * math.sin(h)

    def to_srgb(c: float) -> float:
        c = max(0.0, c)
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    def rgb(Lv, av, bv):
        fy = (Lv + 16) / 116
        fx, fz = fy + av / 500, fy - bv / 200
        f = lambda t: t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
        x, y, z = 0.95047 * f(fx), 1.0 * f(fy), 1.08883 * f(fz)
        r = 3.2406 * x - 1.5372 * y - 0.4986 * z
        g = -0.9689 * x + 1.8758 * y + 0.0415 * z
        bl = 0.0557 * x - 0.2040 * y + 1.0570 * z
        return r, g, bl

    for _ in range(40):                      # reduce chroma until the colour fits into sRGB
        r, g, bl = rgb(L, a, b)
        if min(r, g, bl) >= -0.001 and max(r, g, bl) <= 1.001:
            break
        a, b = a * 0.93, b * 0.93
    hx = "#%02x%02x%02x" % tuple(int(round(min(1.0, to_srgb(v)) * 255)) for v in (r, g, bl))
    return (hx, (L, a, b)) if with_lab else hx


_ORDER = sorted(range(len(TIMELINE)), key=lambda i: (TIMELINE[i][1], i))
_ARTBENCH = {"art_nouveau", "baroque", "expressionism", "impressionism", "post_impressionism", "realism",
             "renaissance", "romanticism", "surrealism", "ukiyo_e"}


def _lightness(year: int) -> float:
    return _L_OLD + (_L_NEW - _L_OLD) * (year - _YEAR0) / (_YEAR1 - _YEAR0)


def _best_hue_rule(chroma: float = 62.0) -> tuple[float, float]:
    """Search for the hue rule (offset + rank * step) that keeps the ten ArtBench styles (then all 29) as far apart as
    possible AFTER gamut clipping.  The result is stored in _STEP / _OFFSET below (re-run by hand after editing
    TIMELINE or the lightness range)."""
    ranked = [TIMELINE[i] for i in _ORDER]

    def min_dist(pts):
        return min(math.dist(pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts)))

    best, best_score = (137.508, 0.0), -1.0
    for step in range(95, 180):
        for off in range(0, 360, 6):
            allp = [_lch_to_hex(_lightness(y), chroma, (off + r * step) % 360.0, with_lab=True)[1]
                    for r, (_, y) in enumerate(ranked)]
            ab = [pt for pt, (n, _) in zip(allp, ranked) if n in _ARTBENCH]
            score = min_dist(ab) + 0.5 * min_dist(allp)
            if score > best_score:
                best, best_score = (float(step), float(off)), score
    return best


_STEP, _OFFSET = 107.0, 96.0        # result of _best_hue_rule() for the current TIMELINE and lightness range
STYLE_COLOR: dict[str, str] = {}
STYLE_YEAR: dict[str, int] = {}
for rank, i in enumerate(_ORDER):
    name, year = TIMELINE[i]
    STYLE_COLOR[name] = _lch_to_hex(_lightness(year), 62.0, (_OFFSET + rank * _STEP) % 360.0)
    STYLE_YEAR[name] = year


def style_colors(names: list[str]) -> list[str]:
    """Colour per style name (a stable hash-based colour for names that are not in the table)."""
    out = []
    for n in names:
        if n in STYLE_COLOR:
            out.append(STYLE_COLOR[n])
        else:
            h = sum(ord(c) * (i + 1) for i, c in enumerate(n)) % 360
            out.append(_lch_to_hex(70.0, 50.0, h))
    return out


def style_years(names: list[str]) -> list[int | None]:
    return [STYLE_YEAR.get(n) for n in names]
