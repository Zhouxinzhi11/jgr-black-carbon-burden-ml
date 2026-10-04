# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
Figure 1 redesign v3 - Word-compatible SVG master

Purpose
-------
Generate a publication-style Figure 1 using a single standalone SVG source that can
be inserted directly into Microsoft Word, while also exporting matching PDF/PNG.

Outputs
-------
Figure_1_redesign_v3_word.svg   # insert directly into Word
Figure_1_redesign_v3.pdf        # vector submission copy
Figure_1_redesign_v3.png        # high-resolution preview

Why this SVG is Word-friendly
-----------------------------
- SVG 1.1 with an explicit viewBox.
- Only basic primitives: rect, line, polygon, circle, text, group.
- No external images, fonts, CSS, clipPaths, masks, foreignObject, markers, or <use>.
- Heatmaps are pure-vector grids, not embedded raster/base64 images.
- All styling is written inline on each SVG element.

Requirements
------------
pip install numpy matplotlib svgwrite cairosvg
"""

from pathlib import Path
import math
import numpy as np
import svgwrite
import cairosvg
from matplotlib import colormaps

# -----------------------------
# Output settings
# -----------------------------
OUTDIR = Path('.')
BASENAME = 'Figure_1_redesign_v3'
SVG_NAME = f'{BASENAME}_word.svg'
PDF_NAME = f'{BASENAME}.pdf'
PNG_NAME = f'{BASENAME}.png'

W, H = 1800, 800
FONT = 'Arial'
MATH_FONT = 'Times New Roman'
BLACK = '#111111'
DARK = '#292929'
MID = '#666666'
LIGHT = '#b8b8b8'
VERY_LIGHT = '#d7d7d7'

# -----------------------------
# SVG canvas
# -----------------------------
dwg = svgwrite.Drawing(
    filename=str(OUTDIR / SVG_NAME),
    size=(f'{W}px', f'{H}px'),
    viewBox=f'0 0 {W} {H}',
    profile='full',
)
dwg.attribs['xmlns'] = 'http://www.w3.org/2000/svg'
dwg.attribs['version'] = '1.1'
dwg.attribs['preserveAspectRatio'] = 'xMidYMid meet'
dwg.attribs['shape-rendering'] = 'geometricPrecision'
dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill='white'))

# -----------------------------
# Basic helpers
# -----------------------------
def txt(x, y, s, size=20, weight='normal', anchor='start',
        fill=BLACK, family=FONT, style='normal'):
    el = dwg.text(
        s,
        insert=(x, y),
        fill=fill,
        font_family=family,
        font_size=size,
        font_weight=weight,
        font_style=style,
        text_anchor=anchor,
    )
    dwg.add(el)
    return el


def seg(x1, y1, x2, y2, stroke=DARK, width=1.5):
    dwg.add(dwg.line(start=(x1, y1), end=(x2, y2),
                     stroke=stroke, stroke_width=width))


def arr(x1, y1, x2, y2, stroke=DARK, width=1.8, head=9):
    dx, dy = x2-x1, y2-y1
    L = math.hypot(dx, dy)
    if L <= 1e-9:
        return
    ux, uy = dx/L, dy/L
    bx, by = x2-ux*head*1.45, y2-uy*head*1.45
    seg(x1, y1, bx, by, stroke=stroke, width=width)
    px, py = -uy, ux
    pts = [
        (x2, y2),
        (bx + px*head*0.65, by + py*head*0.65),
        (bx - px*head*0.65, by - py*head*0.65),
    ]
    dwg.add(dwg.polygon(points=pts, fill=stroke, stroke=stroke, stroke_width=0.4))


def panel_title(x, letter, title):
    txt(x, 43, f'({letter})', size=27, weight='bold')
    txt(x+48, 43, title, size=29, weight='bold')


def rect_box(x, y, w, h, label=None, size=19, lw=1.4):
    dwg.add(dwg.rect(insert=(x, y), size=(w, h), fill='white',
                     stroke=DARK, stroke_width=lw))
    if label:
        txt(x+w/2, y+h/2+size*0.34, label, size=size, anchor='middle')


def model_block(x, y, w, h, label):
    pts = [(x, y), (x+0.84*w, y+0.13*h),
           (x+0.84*w, y+0.87*h), (x, y+h)]
    dwg.add(dwg.polygon(points=pts, fill='white', stroke=DARK, stroke_width=1.6))
    txt(x+0.43*w, y+h/2+9, label, size=27, anchor='middle',
        family=MATH_FONT, style='italic')


def stack_icon(x, y, w, h, n=4, dx=6, dy=-7):
    for i in range(n-1, -1, -1):
        dwg.add(dwg.rect(insert=(x+i*dx, y+i*dy), size=(w, h),
                         fill='white', stroke=DARK, stroke_width=1.25))


def math_label(x, y, base, sub='', size=34, anchor='middle', overbar=False):
    """
    Robust label: base and subscript are separate SVG text elements.
    This avoids baseline-shift/tspan rendering problems in Word/CairoSVG.
    x,y define the visual centre of the base symbol.
    """
    # crude width-aware subscript offset; adequate for single-letter scientific labels
    txt(x, y, base, size=size, anchor=anchor, family=MATH_FONT, style='italic')
    # Overbars are supplied as Unicode precomposed/combining characters (e.g. ȳ, M̄)
    # rather than drawn as separate lines, which is more robust in Word SVG rendering.
    if sub:
        txt(x+size*0.34, y+size*0.18, sub, size=size*0.58,
            anchor='start', family=MATH_FONT, style='italic')


def eq_math_label(x, y):
    """Draw y_d = ybar_m + y'_d with independent SVG text elements."""
    math_label(x-132, y, 'y', 'd', size=39)
    txt(x-72, y, '=', size=36, anchor='middle', family=MATH_FONT)
    math_label(x-10, y, 'ȳ', 'm', size=39)
    txt(x+58, y, '+', size=36, anchor='middle', family=MATH_FONT)
    math_label(x+118, y, "y′", 'd', size=39)

# -----------------------------
# Vector thumbnail fields
# -----------------------------
def field(kind='mean', n=11):
    u = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(u, u)
    if kind == 'daily':
        Z = (1.0*np.exp(-((X+0.34)**2+(Y-0.10)**2)/0.16)
             + 0.62*np.exp(-((X-0.18)**2+(Y+0.22)**2)/0.23))
        cmap = 'viridis'
    elif kind == 'mean':
        Z = (1.2*np.exp(-((X+0.02)**2+(Y-0.02)**2)/0.25)
             + 0.20*np.exp(-((X-0.32)**2+(Y+0.25)**2)/0.37))
        cmap = 'viridis'
    elif kind == 'meteo':
        Z = 0.72*np.cos(1.30*np.pi*X)*np.cos(1.05*np.pi*Y)
        Z += 0.28*np.exp(-((X-0.10)**2+(Y+0.18)**2)/0.25)
        cmap = 'cividis'
    elif kind in ('anomaly', 'pred_anomaly'):
        amp = 1.0 if kind == 'anomaly' else 0.85
        Z = amp*np.sin(2.25*np.pi*X)*np.cos(1.85*np.pi*Y)
        Z *= np.exp(-0.18*(X**2+Y**2))
        cmap = 'RdBu_r'
    elif kind == 'pred_mean':
        Z = (1.03*np.exp(-((X+0.03)**2+(Y-0.02)**2)/0.24)
             + 0.18*np.exp(-((X-0.28)**2+(Y+0.23)**2)/0.34))
        cmap = 'viridis'
    else:
        Z = np.zeros((n,n)); cmap='Greys'
    Z = (Z-Z.min())/(Z.max()-Z.min()+1e-12)
    return Z, colormaps[cmap]


def heatmap(x, y, w, h, kind, n=11):
    Z, cmap = field(kind, n)
    cw, ch = w/n, h/n
    for r in range(n):
        for c in range(n):
            rgba = cmap(float(Z[r,c]))
            rgb = '#%02x%02x%02x' % tuple(int(round(255*v)) for v in rgba[:3])
            dwg.add(dwg.rect(insert=(x+c*cw, y+(n-1-r)*ch),
                             size=(cw+0.25, ch+0.25), fill=rgb, stroke='none'))
    dwg.add(dwg.rect(insert=(x,y), size=(w,h), fill='none',
                     stroke=DARK, stroke_width=1.2))

# -----------------------------
# Diagnostic-control glyphs
# -----------------------------
def cyclic_glyph(cx, cy):
    dwg.add(dwg.rect(insert=(cx-59, cy-12), size=(36,24), fill='white',
                     stroke=MID, stroke_width=1.15))
    dwg.add(dwg.rect(insert=(cx+23, cy-12), size=(36,24), fill='white',
                     stroke=MID, stroke_width=1.15))
    arr(cx-14, cy, cx+14, cy, stroke=MID, width=1.1, head=6)


def support_glyph(cx, cy):
    dwg.add(dwg.rect(insert=(cx-54, cy-14), size=(108,28), fill='white',
                     stroke=MID, stroke_width=1.15))
    for px, py in [(cx-32,cy-6),(cx-16,cy+7),(cx+15,cy-7),(cx+35,cy+6)]:
        dwg.add(dwg.circle(center=(px,py), r=4, fill=MID))
    arr(cx-4, cy-11, cx+18, cy-11, stroke=MID, width=1.0, head=5)


def binary_glyph(cx, cy):
    cell=12; x0=cx-18; y0=cy-18
    pattern=np.array([[1,0,1],[0,1,0],[1,1,0]])
    for r in range(3):
        for c in range(3):
            dwg.add(dwg.rect(insert=(x0+c*cell,y0+r*cell), size=(cell,cell),
                             fill=(DARK if pattern[r,c] else 'white'),
                             stroke=MID, stroke_width=0.65))

# -----------------------------
# Panel geometry
# -----------------------------
A0, A1 = 10, 535
B0, B1 = 565, 1290
C0, C1 = 1320, 1790
seg(A1, 58, A1, 690, stroke=VERY_LIGHT, width=1.3)
seg(B1, 58, B1, 690, stroke=VERY_LIGHT, width=1.3)
panel_title(A0, 'a', 'Time-scale decomposition')
panel_title(B0, 'b', 'Forward diagnosis')
panel_title(C0, 'c', 'Constrained inverse')

# -----------------------------
# Panel (a)
# -----------------------------
img_y, iw, ih = 285, 82, 82
xd, xm, xa = 60, 245, 420
heatmap(xd, img_y, iw, ih, 'daily')
heatmap(xm, img_y, iw, ih, 'mean')
heatmap(xa, img_y, iw, ih, 'anomaly')
txt(180, img_y+53, '=', size=38, weight='bold', anchor='middle')
txt(365, img_y+53, '+', size=36, weight='bold', anchor='middle')
for x, lab in [(xd,'daily'),(xm,'mean'),(xa,'anomaly')]:
    txt(x+iw/2, 410, lab, size=19, anchor='middle', fill=DARK)
math_label(xd+iw/2, 458, 'y', 'd', size=36)
math_label(xm+iw/2, 458, 'ȳ', 'm', size=36)
math_label(xa+iw/2, 458, "y′", 'd', size=36)
eq_math_label(267, 590)

# -----------------------------
# Panel (b): mean state
# -----------------------------
txt(B0+40, 132, 'Mean state', size=27, weight='bold')
stack_x, stack_y = B0+65, 212
stack_icon(stack_x, stack_y, 50, 76)
math_label(stack_x+35, 327, 'E', 'm', size=30)

plus_x = B0+200
txt(plus_x, 264, '+', size=34, weight='bold', anchor='middle')
met_x = B0+245
heatmap(met_x, 214, 80, 80, 'meteo')
math_label(met_x+40, 327, 'M̄', 'm', size=30)

# The plus sign itself encodes the two-input combination; no connector line is drawn
# between E_m and Mbar_m, avoiding a false sequential-flow interpretation.
merge_y = 254

model_x = B0+415
model_block(model_x, 208, 83, 93, 'fθ')
pred_x = B0+560
heatmap(pred_x, 214, 80, 80, 'pred_mean')
math_label(pred_x+40, 327, 'ŷ', 'm', size=30)
arr(met_x+96, merge_y, model_x-16, merge_y)
arr(model_x+79, merge_y, pred_x-18, merge_y)

# Diagnostic controls
txt(B0+40, 373, 'Diagnostic controls', size=19, weight='bold', fill=DARK)
seg(B0+220, 366, B1-35, 366, stroke=LIGHT, width=1.2)
centers=[B0+160,B0+365,B0+575]
cyclic_glyph(centers[0],416)
support_glyph(centers[1],416)
binary_glyph(centers[2],416)
for cx,lab in zip(centers,['cyclic shift','support-preserving','binary support']):
    txt(cx,460,lab,size=17,anchor='middle',fill=DARK)

# Mean-state geographic holdout note -- positioned directly under mean-state controls
# to avoid implying that the daily-anomaly branch used the same spatial holdout.
seg(B0+40, 484, B1-40, 484, stroke=LIGHT, width=1.0)
txt(B0+45, 508, 'Mean-state geographic holdout', size=16.5, weight='bold', fill=DARK)
txt(B0+45, 532, 'East Asia core excluded before normalization, training, and checkpoint selection',
    size=15.2, fill=DARK)

# Daily anomaly
txt(B0+40, 582, 'Daily anomaly', size=27, weight='bold')
an_x=B0+70
heatmap(an_x, 612, 80, 80, 'anomaly')
math_label(an_x+40, 730, 'M′', 'd', size=30)
txt(B0+185, 660, '+', size=34, weight='bold', anchor='middle')
fire_x=B0+225
rect_box(fire_x, 618, 130, 68, 'Fire / AEF', size=20)
g_x=B0+405
model_block(g_x, 611, 83, 88, 'gϕ')
out_x=B0+555
heatmap(out_x,612,80,80,'pred_anomaly')
math_label(out_x+40,730,'ŷ′','d',size=30)
arr(fire_x+145,660,g_x-18,660)
arr(g_x+78,660,out_x-18,660)

# -----------------------------
# Panel (c): constrained inverse
# -----------------------------
stack_x=C0+45
stack_icon(stack_x,215,47,72)
math_label(stack_x+35,326,'E','train',size=29)

plus_x=C0+147
txt(plus_x,260,'+',size=34,weight='bold',anchor='middle')
delta_x=C0+185
rect_box(delta_x,222,72,66,'δ',size=34)
txt(delta_x+36,326,'‖δ‖∞ ≤ Δₘₐₓ',size=21,anchor='middle',family=MATH_FONT)
# E_train and delta are shown as additive inputs; no line is drawn through the plus sign.

fm_x=C0+300
model_block(fm_x,211,79,86,'fθ')
txt(fm_x+34,175,'frozen',size=18,style='italic',anchor='middle',fill=DARK)
p_x=C0+405
heatmap(p_x,225,58,58,'pred_mean')
math_label(p_x+29,326,'ŷ','m',size=29)
arr(delta_x+85,254,fm_x-16,254)
arr(fm_x+75,254,p_x-15,254)

# Trade-off plot -- kept safely below the inverse pipeline.
plot_x, plot_base = C0+105, 690
plot_w, plot_h = 275, 205
plot_top = plot_base-plot_h
seg(plot_x,plot_base,plot_x,plot_top,stroke=DARK,width=1.7)
seg(plot_x,plot_base,plot_x+plot_w,plot_base,stroke=DARK,width=1.7)
pts=[(plot_x+40,plot_base-45),(plot_x+132,plot_base-120),(plot_x+225,plot_base-168)]
for (x1,y1),(x2,y2) in zip(pts[:-1],pts[1:]):
    seg(x1,y1,x2,y2,stroke='#444444',width=2.0)
for px,py in pts:
    dwg.add(dwg.circle(center=(px,py),r=6,fill='#444444'))
txt(plot_x+plot_w/2,plot_base+40,'prior deviation',size=18,anchor='middle',fill=DARK)
ylab=dwg.text('reconstruction skill',insert=(plot_x-43,plot_top+plot_h/2),
              fill=DARK,font_family=FONT,font_size=18,text_anchor='middle',
              transform=f'rotate(-90 {plot_x-43} {plot_top+plot_h/2})')
dwg.add(ylab)
txt(plot_x+208,plot_top+12,'trade-off',size=18,anchor='middle',fill=DARK)

# -----------------------------
# Save / convert
# -----------------------------
svg_path=OUTDIR/SVG_NAME
dwg.save(pretty=True)

cairosvg.svg2pdf(url=str(svg_path),write_to=str(OUTDIR/PDF_NAME),
                 output_width=W,output_height=H)
cairosvg.svg2png(url=str(svg_path),write_to=str(OUTDIR/PNG_NAME),
                 output_width=W*4,output_height=H*4)

print('Saved:')
print((OUTDIR/SVG_NAME).resolve())
print((OUTDIR/PDF_NAME).resolve())
print((OUTDIR/PNG_NAME).resolve())
print('SVG_PROFILE=standalone SVG 1.1; pure vector; no external dependencies')
print('WORD=Insert > Pictures > This Device > select Figure_1_redesign_v3_word.svg')
