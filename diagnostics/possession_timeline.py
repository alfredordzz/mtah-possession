"""Possession timeline of a submission, as a local HTML page. No ground truth is read.

    python diagnostics/possession_timeline.py                          # evaluation/results/v12_standalone.csv
    python diagnostics/possession_timeline.py evaluation/results/v9_1.csv

Writes evaluation/reports/timeline_<file>.html (the reports folder is git-ignored, so the page
stays on this machine) and prints its path. The page shows, for each half:
    - one strip for the whole half
    - the same half in rows of 5 minutes, wide enough to see one-second possessions
    - the home share of live-ball time in a rolling 5-minute window
Hovering a block shows who has the ball, when and for how long. Open it in any browser; it needs
no internet connection.
"""

import html
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from games import GAME, half_video_blocks  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
COL = {0: "#2D5BD0", 1: "#D47A22", 2: "#B9C0BB", 3: "#FFFFFF"}
NAME = {0: "Local", 1: "Visitante", 2: "Balón muerto", 3: "Sin calificar"}
ROW_S = 300      # seconds per detailed row
W = 1100         # drawing width in pixels


def clock(sec):
    return "{:d}:{:02d}".format(int(sec // 60), int(sec % 60))


def runs(c):
    ch = np.flatnonzero(np.diff(c)) + 1
    starts = np.r_[0, ch]
    ends = np.r_[ch, len(c)]
    return [(int(a), int(b), int(c[a])) for a, b in zip(starts, ends)]


def strip(c, t0, t1, height, span_frames=None):
    """SVG rectangles for frames t0..t1 of one half (codes c); span_frames maps to W pixels."""
    span = (span_frames or (t1 - t0)) / FPS
    out = []
    for a, b, k in runs(c[t0:t1]):
        x = a / FPS / span * W
        w = max((b - a) / FPS / span * W, 0.6)
        tip = "{}: {} a {} ({:.1f} s)".format(NAME[k], clock((t0 + a) / FPS), clock((t0 + b) / FPS), (b - a) / FPS)
        out.append('<rect x="{:.2f}" y="0" width="{:.2f}" height="{}" fill="{}"><title>{}</title></rect>'.format(
            x, w, height, COL[k], html.escape(tip)))
    return out


def axis(t0, t1, y, step_s, span_frames=None):
    span = (span_frames or (t1 - t0)) / FPS
    out = []
    first = int(np.ceil(t0 / FPS / step_s)) * step_s
    for s in np.arange(first, t1 / FPS + 1e-6, step_s):
        x = (s - t0 / FPS) / span * W
        out.append('<line x1="{0:.1f}" x2="{0:.1f}" y1="{1}" y2="{2}" stroke="#9AA59F" stroke-width="0.6"/>'
                   '<text x="{0:.1f}" y="{3}" text-anchor="middle" class="ax">{4}</text>'.format(x, y, y + 4, y + 15, clock(s)))
    return out


def share_curve(c, win_s=300):
    alive = (c == 0) | (c == 1)
    home = (c == 0).astype(float)
    k = win_s * FPS
    cs_h = np.r_[0, np.cumsum(home)]
    cs_a = np.r_[0, np.cumsum(alive)]
    idx = np.arange(0, len(c), FPS * 10)
    lo = np.clip(idx - k // 2, 0, len(c))
    hi = np.clip(idx + k // 2, 0, len(c))
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = (cs_h[hi] - cs_h[lo]) / (cs_a[hi] - cs_a[lo])
    return idx, sh


def half_block(name, c):
    n = len(c)
    graded = c != 3
    spells = [r for r in runs(c) if r[2] in (0, 1)]
    dur = np.array([(b - a) / FPS for a, b, _ in spells]) if spells else np.array([0.0])
    alive = ((c == 0) | (c == 1)).sum()
    parts = ['<section><h2>Mitad {}</h2>'.format(html.escape(str(name)))]
    parts.append('<p class="sum">{:.1f} minutos · {} posesiones ({} local, {} visitante) · mediana {:.1f} s · '
                 'la más larga {:.0f} s · balón muerto {:.0%} · local {:.0%} del balón vivo</p>'.format(
                     graded.sum() / FPS / 60, len(spells), sum(1 for s in spells if s[2] == 0),
                     sum(1 for s in spells if s[2] == 1), np.median(dur), dur.max(),
                     (c == 2).sum() / max(1, graded.sum()), (c == 0).sum() / max(1, alive)))
    parts.append('<svg viewBox="-24 -2 {} 52" class="full"><g>{}</g>{}</svg>'.format(
        W + 48, "".join(strip(c, 0, n, 30)), "".join(axis(0, n, 30, 300))))
    idx, sh = share_curve(c)
    pts = " ".join("{:.1f},{:.1f}".format(i / n * W, 5 + (1 - s) * 70) for i, s in zip(idx, sh) if np.isfinite(s))
    parts.append('<h3>Local, del tiempo con balón vivo (ventana de 5 minutos)</h3>'
                 '<svg viewBox="-24 0 {3} 100" class="full"><line x1="0" x2="{0}" y1="40" y2="40" stroke="#9AA59F" '
                 'stroke-dasharray="4 4"/><text x="4" y="36" class="ax">50%</text><text x="4" y="14" class="ax">100%</text>'
                 '<polyline points="{1}" fill="none" stroke="#2D5BD0" stroke-width="1.6"/>{2}</svg>'.format(
                     W, pts, "".join(axis(0, n, 76, 300)), W + 48))
    parts.append('<h3>En detalle (filas de 5 minutos)</h3>')
    for t0 in range(0, n, ROW_S * FPS):
        t1 = min(n, t0 + ROW_S * FPS)
        span = ROW_S * FPS
        parts.append('<svg viewBox="-24 -2 {} 42" class="row"><g>{}</g>{}</svg>'.format(
            W + 48, "".join(strip(c, t0, t1, 22, span)), "".join(axis(t0, t1, 22, 30, span))))
    parts.append("</section>")
    return "\n".join(parts)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evaluation", "results", "v12_standalone.csv")
    codes = pm.label_codes(pm.read_label_csv(path))
    blocks = half_video_blocks(GAME)
    base = os.path.splitext(os.path.basename(path))[0]
    body = []
    for name, a, b in blocks:
        c = codes[a:b + 1]
        keep = np.flatnonzero(c != 3)
        c = c[keep[0]:keep[-1] + 1] if len(keep) else c
        body.append(half_block(name, c))
    legend = "".join('<span><i style="background:{}"></i>{}</span>'.format(COL[k], NAME[k]) for k in (0, 1, 2))
    page = """<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Timeline de posesión: {name}</title>
<style>
body{{font:14px/1.5 system-ui,Segoe UI,Arial,sans-serif;margin:24px auto;max-width:1160px;padding:0 16px;color:#17201B;background:#F6F7F5}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:20px;margin:28px 0 4px}} h3{{font-size:14px;margin:16px 0 4px;color:#5B665F}}
.sum{{color:#5B665F;margin:0 0 8px}} svg{{display:block;width:100%;height:auto;margin:2px 0}}
 .ax{{font-size:9px;fill:#5B665F}}
.legend span{{margin-right:16px}} .legend i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.note{{color:#5B665F;font-size:13px}}
</style></head><body>
<h1>Timeline de posesión</h1>
<p class="note">Archivo: {file}. Son las respuestas del modelo, no lo que pasó en realidad. Pasa el cursor sobre un bloque para ver quién tiene el balón, desde cuándo y cuánto dura. Tiempos desde el inicio de cada mitad.</p>
<p class="legend">{legend}</p>
{body}
</body></html>""".format(name=html.escape(base), file=html.escape(os.path.relpath(path, ROOT)), legend=legend, body="\n".join(body))
    out_dir = os.path.join(ROOT, "evaluation", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "timeline_{}.html".format(base))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print("escrito: {}".format(os.path.relpath(out, ROOT)))
    print("ábrelo con:  start {}".format(os.path.relpath(out, ROOT)))


if __name__ == "__main__":
    main()
