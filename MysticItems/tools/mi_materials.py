"""Procedural hp materials. Colours are linear-light values.

Every material reads the same three wear signals so items age consistently:
  ao    - self occlusion stored on the hp mesh (dirt, patina, grime collect here)
  edge  - Cycles pointiness (convex edges get handled, polished or rubbed)
  up    - world normal Z (dust and lichen settle on top faces)
"""
from mi_core import MB


def _edge(mb, lo=0.505, hi=0.56, breakup=140.0, coverage=0.45):
    """Convex-edge mask broken up so wear reads as handling, not as a uniform outline."""
    e = mb.smooth(mb.geo('Pointiness'), lo, hi)
    n = mb.noise(scale=breakup, detail=6, rough=0.65)
    brk = mb.smooth(n, coverage - 0.12, coverage + 0.12)
    return mb.math('MULTIPLY', e, brk, clamp=True)


def _cavity(mb, open_=0.92, closed=0.45):
    return mb.maprange(mb.ao(), open_, closed, 0.0, 1.0)


def _up(mb):
    return mb.sep(mb.geo('Normal'))[2]


def _mask_stack(mb, masks):
    """masks: [(image, projector_object, facing_vector or None)] -> combined 0..1 height."""
    acc = None
    for img, proj, facing in masks:
        uv = proj(mb) if callable(proj) else mb.projector(proj)
        v = mb.image(img, uv, extension='EXTEND' if callable(proj) else 'CLIP')
        v = mb.math('MULTIPLY', v, 1.0)
        if facing is not None:
            d = mb.vmath('DOT_PRODUCT', mb.geo('Normal'), facing, out='Value')
            v = mb.math('MULTIPLY', v, mb.smooth(d, 0.25, 0.6))
        acc = v if acc is None else mb.math('MAXIMUM', acc, v)
    return acc


def scratches(mb, scale=60.0, density=0.55):
    """Fine directional scratches: stretched noise lines in two crossing directions."""
    out = None
    for rot, sc in (((0.3, 0.7, 0.2), (1.0, 1.0, 22.0)), ((1.2, -0.4, 0.9), (22.0, 1.0, 1.0))):
        v = mb.mapping(mb.coord(), rot=rot, scale=sc)
        n = mb.noise(v, scale=scale, detail=2, rough=0.4)
        line = mb.smooth(mb.math('ABSOLUTE', mb.math('SUBTRACT', n, 0.5)), 0.012, 0.0)
        gate = mb.smooth(mb.noise(scale=scale * 0.15, detail=2), density, density + 0.1)
        line = mb.math('MULTIPLY', line, gate)
        out = line if out is None else mb.math('MAXIMUM', out, line)
    return out


# ---------------------------------------------------------------------------
# metals
# ---------------------------------------------------------------------------

def aged_metal(name, clean=(0.72, 0.47, 0.19), tarnish=(0.16, 0.095, 0.045),
               patina=(0.085, 0.24, 0.17), patina_amount=0.55, masks=(), mask_mode='engrave',
               mask_depth=0.0006, polish=0.8, hammered=0.0):
    mb = MB(name)
    p = mb.coord()
    cav = _cavity(mb)
    edge = _edge(mb, coverage=0.4)
    tone = mb.noise(p, scale=35, detail=4, rough=0.6)
    sc = scratches(mb, scale=90, density=0.5)

    rub = mb.math('MAXIMUM', mb.math('MULTIPLY', edge, polish),
                  mb.math('MULTIPLY', mb.smooth(tone, 0.62, 0.8), 0.25))
    metal_col = mb.mixc(rub, mb.mixc(mb.smooth(tone, 0.3, 0.7), tarnish,
                                     tuple(c * 1.6 for c in tarnish)), clean)
    metal_col = mb.mixc(mb.math('MULTIPLY', sc, 0.5), metal_col, clean)

    height = mb.math('MULTIPLY', mb.noise(p, scale=900, detail=3), 0.15)
    if hammered > 0:
        dents = mb.voronoi(p, scale=160, feature='F1', out='Distance')
        height = mb.math('ADD', height, mb.math('MULTIPLY', mb.math('POWER', dents, 0.5), hammered))

    groove = None
    if masks:
        groove = _mask_stack(mb, masks)
        sign = -1.0 if mask_mode == 'engrave' else 1.0
        height = mb.math('MULTIPLY_ADD', groove, sign * mask_depth / 0.0004, height)
        if mask_mode == 'engrave':
            cav = mb.math('MAXIMUM', cav, mb.math('MULTIPLY', groove, 0.9))
        else:
            # raised relief gets rubbed bright on its tops, dark in the background
            metal_col = mb.mixc(mb.math('MULTIPLY', groove, 0.6), metal_col, clean)

    pat_noise = mb.noise(p, scale=60, detail=8, rough=0.7)
    pat_break = mb.noise(p, scale=14, detail=3)
    pat = mb.math('MULTIPLY', mb.smooth(cav, 0.35, 0.85),
                  mb.smooth(pat_noise, 0.64 - patina_amount * 0.2, 0.7 - patina_amount * 0.2))
    pat = mb.math('MULTIPLY', pat, mb.smooth(pat_break, 0.35, 0.6))
    grime = mb.smooth(cav, 0.55, 1.0)

    col = mb.mixc(pat, metal_col, patina)
    col = mb.mixc(mb.math('MULTIPLY', grime, 0.85), col, (0.022, 0.018, 0.013))
    metal = mb.math('SUBTRACT', 1.0, mb.math('MAXIMUM', pat, mb.math('MULTIPLY', grime, 0.9)),
                    clamp=True)
    rough = mb.mixf(rub, 0.5, 0.24)
    rough = mb.mixf(mb.math('MULTIPLY', sc, 0.6), rough, 0.38)
    rough = mb.mixf(pat, rough, 0.88)
    rough = mb.mixf(grime, rough, 0.8)
    pat_h = mb.math('MULTIPLY', pat, mb.noise(p, scale=1500, detail=2))
    height = mb.math('ADD', height, mb.math('MULTIPLY', pat_h, 0.6))
    nrm = mb.bump(height, 1.0, 0.0004)
    nrm = mb.bump(mb.math('MULTIPLY', sc, -1.0), 0.25, 0.00005, nrm)
    return mb.set(base=col, rough=rough, metal=metal, normal=nrm)


def brass(name, **kw):
    return aged_metal(name, **kw)


def bronze(name, **kw):
    kw.setdefault('clean', (0.52, 0.30, 0.13))
    kw.setdefault('tarnish', (0.11, 0.06, 0.03))
    kw.setdefault('patina_amount', 0.85)
    return aged_metal(name, **kw)


def dark_iron(name, **kw):
    kw.setdefault('clean', (0.42, 0.40, 0.37))
    kw.setdefault('tarnish', (0.05, 0.045, 0.04))
    kw.setdefault('patina', (0.20, 0.07, 0.025))  # rust instead of verdigris
    kw.setdefault('patina_amount', 0.7)
    return aged_metal(name, **kw)


# ---------------------------------------------------------------------------
# organics
# ---------------------------------------------------------------------------

def wood(name, species='walnut', grain_axis='Z', pith=(0.21, -0.37, 0.0), ring_scale=95.0,
         finish=0.46, masks=()):
    """Growth rings around an off-centre pith so turned parts show long flame figure."""
    palettes = {
        'walnut': [(0.2, (0.026, 0.013, 0.007)), (0.45, (0.050, 0.027, 0.014)),
                   (0.62, (0.075, 0.043, 0.023)), (0.8, (0.098, 0.060, 0.032))],
        'oak': [(0.2, (0.085, 0.050, 0.022)), (0.45, (0.150, 0.092, 0.042)),
                (0.62, (0.205, 0.130, 0.062)), (0.8, (0.245, 0.160, 0.080))],
        'ebonised': [(0.0, (0.012, 0.009, 0.007)), (0.6, (0.026, 0.018, 0.012)),
                     (1.0, (0.045, 0.030, 0.019))],
    }
    mb = MB(name)
    p = mb.coord()
    stretch = {'Z': (1.0, 1.0, 0.08), 'X': (0.08, 1.0, 1.0), 'Y': (1.0, 0.08, 1.0)}[grain_axis]
    pg = mb.mapping(p, loc=pith, scale=stretch)
    warp = mb.noise(pg, scale=18, detail=6, rough=0.6, out='Color')
    pw = mb.vmath('ADD', pg, mb.vmath('SCALE', mb.vmath('SUBTRACT', warp, (0.5, 0.5, 0.5)), scale=0.012))
    rings = mb.wave(pw, scale=ring_scale, kind='RINGS', axis=grain_axis, distortion=2.5, detail=3,
                    profile='SAW')
    late = mb.smooth(rings, 0.78, 0.97)  # thin, dark latewood lines rather than zebra bands
    streak = mb.noise(pg, scale=22, detail=7, rough=0.62)
    fig = mb.noise(pg, scale=90, detail=4)
    t = mb.math('MULTIPLY_ADD', streak, 0.75, mb.math('MULTIPLY', fig, 0.15))
    t = mb.math('SUBTRACT', t, mb.math('MULTIPLY', late, 0.18))
    col = mb.ramp(t, palettes[species])
    ppos = mb.mapping(p, scale=tuple(s * 6 for s in stretch))
    pores = mb.smooth(mb.noise(ppos, scale=420, detail=2), 0.62, 0.72)
    col = mb.mixc(mb.math('MULTIPLY', pores, 0.55), col, (0.012, 0.007, 0.004))

    cav = _cavity(mb)
    edge = _edge(mb, coverage=0.5)
    # finish wears through on edges: lighter, drier wood
    col = mb.mixc(mb.math('MULTIPLY', edge, 0.55), col, mb.hsv(col, 0.5, 0.75, 1.9))
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(cav, 0.3, 1.0), 0.8), col, (0.012, 0.009, 0.006))
    dust = mb.math('MULTIPLY', mb.smooth(_up(mb), 0.6, 0.95),
                   mb.smooth(mb.noise(p, scale=80, detail=5), 0.5, 0.7))
    col = mb.mixc(mb.math('MULTIPLY', dust, 0.3), col, (0.16, 0.14, 0.12))
    rough = mb.mixf(edge, finish, 0.62)
    rough = mb.mixf(mb.math('MULTIPLY', mb.smooth(_up(mb), 0.7, 0.95), 0.6), rough, 0.85)  # dull, dusty tops
    rough = mb.math('MULTIPLY_ADD', mb.noise(p, scale=30, detail=3), 0.12, rough)
    rough = mb.mixf(mb.math('MULTIPLY', pores, 0.5), rough, 0.7)
    height = mb.math('MULTIPLY', pores, -0.3)
    height = mb.math('MULTIPLY_ADD', late, -0.12, height)
    if masks:
        g = _mask_stack(mb, masks)
        height = mb.math('MULTIPLY_ADD', g, -2.0, height)
        col = mb.mixc(mb.math('MULTIPLY', g, 0.6), col, (0.015, 0.01, 0.006))
    nrm = mb.bump(height, 1.0, 0.00025)
    return mb.set(base=col, rough=rough, metal=0.0, normal=nrm, coat=0.0)


def leather(name, base=(0.060, 0.022, 0.012), dark=(0.026, 0.010, 0.006),
            worn=(0.20, 0.105, 0.05), grain_scale=380.0, masks=(), mask_depth=0.0005,
            crease=1.0):
    mb = MB(name)
    p = mb.coord()
    cav = _cavity(mb)
    edge = _edge(mb, lo=0.5, hi=0.545, breakup=90, coverage=0.55)
    mott = mb.noise(p, scale=28, detail=6, rough=0.65)
    col = mb.mixc(mb.smooth(mott, 0.35, 0.68), dark, base)
    pebble = mb.voronoi(p, scale=grain_scale, out='Distance')
    pebble2 = mb.voronoi(p, scale=grain_scale * 2.3, out='Distance')
    wr = mb.noise(mb.mapping(p, scale=(1, 3.5, 1)), scale=55, detail=10, rough=0.7)
    wrinkle = mb.smooth(mb.math('ABSOLUTE', mb.math('SUBTRACT', wr, 0.5)), 0.045, 0.0)
    scuff_n = mb.noise(p, scale=45, detail=8, rough=0.7)
    scuff = mb.math('MULTIPLY', mb.smooth(scuff_n, 0.66, 0.74),
                    mb.math('SUBTRACT', 1.0, mb.smooth(cav, 0.2, 0.6)))
    rub = mb.math('MAXIMUM', edge, mb.math('MULTIPLY', scuff, 0.8))
    col = mb.mixc(rub, col, worn)
    col = mb.mixc(mb.math('MULTIPLY', wrinkle, 0.35 * crease), col, dark)
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(cav, 0.25, 1.0), 0.75), col, (0.012, 0.007, 0.005))
    height = mb.math('ADD', mb.math('MULTIPLY', pebble, 0.6), mb.math('MULTIPLY', pebble2, 0.25))
    height = mb.math('SUBTRACT', height, mb.math('MULTIPLY', wrinkle, 1.2 * crease))
    rough = mb.mixf(rub, 0.62, 0.38)
    rough = mb.math('MULTIPLY_ADD', pebble, 0.15, rough)
    if masks:
        g = _mask_stack(mb, masks)
        height = mb.math('MULTIPLY_ADD', g, -mask_depth / 0.00025, height)
        col = mb.mixc(mb.math('MULTIPLY', g, 0.7), col, mb.hsv(dark, 0.5, 1.0, 0.7))
        rough = mb.mixf(g, rough, 0.34)  # tooling burnishes the leather
    nrm = mb.bump(height, 1.0, 0.00025)
    return mb.set(base=col, rough=rough, metal=0.0, normal=nrm, sheen=0.2)


def stone(name, masks=(), lichen=1.0, pigment=(0.17, 0.045, 0.022)):
    """Slate-grey greywacke with mica flecks, crustose lichen and red ochre left in carvings."""
    mb = MB(name)
    p = mb.coord()
    cav = _cavity(mb, 0.95, 0.5)
    edge = _edge(mb, lo=0.5, hi=0.535, breakup=60, coverage=0.5)
    up = _up(mb)
    big = mb.noise(p, scale=9, detail=5, rough=0.6)
    mid = mb.noise(p, scale=40, detail=3, rough=0.55)
    layers = mb.wave(mb.mapping(p, rot=(0.15, 0.3, 0.0)), scale=22, axis='Z', distortion=6,
                     detail=4, detail_scale=2)
    col = mb.ramp(big, [(0.25, (0.050, 0.054, 0.052)), (0.55, (0.090, 0.092, 0.085)),
                        (0.8, (0.120, 0.115, 0.100))])
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(layers, 0.55, 0.9), 0.35), col, (0.06, 0.05, 0.04))
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(mid, 0.45, 0.75), 0.22), col, (0.14, 0.14, 0.13))
    iron = mb.smooth(mb.noise(p, scale=6, detail=6, rough=0.7), 0.58, 0.72)
    col = mb.mixc(mb.math('MULTIPLY', iron, 0.3), col, (0.13, 0.075, 0.035))
    fleck = mb.smooth(mb.voronoi(p, scale=120, out='Distance'), 0.1, 0.04)
    col = mb.mixc(mb.math('MULTIPLY', fleck, 0.35), col, (0.26, 0.25, 0.23))
    # handled edges are slightly polished and paler
    col = mb.mixc(mb.math('MULTIPLY', edge, 0.4), col, (0.17, 0.165, 0.15))
    rough = mb.mixf(edge, 0.82, 0.6)
    rough = mb.mixf(mb.math('MULTIPLY', fleck, 0.8), rough, 0.35)

    # grain-scale relief stays gentle: strong fine bump reads as camouflage speckle under a key light
    height = mb.math('ADD', mb.math('MULTIPLY', mid, 0.35), mb.math('MULTIPLY', mb.noise(p, scale=220, detail=1.5), 0.12))
    height = mb.math('MULTIPLY_ADD', mb.voronoi(p, scale=90, feature='SMOOTH_F1', out='Distance'), 0.25, height)

    if masks:
        g = _mask_stack(mb, masks)
        height = mb.math('MULTIPLY_ADD', g, -6.0, height)
        cav = mb.math('MAXIMUM', cav, mb.math('MULTIPLY', g, 0.5))
        pig_mask = mb.math('MULTIPLY', mb.smooth(g, 0.4, 0.8),
                           mb.smooth(mb.noise(p, scale=70, detail=8, rough=0.75), 0.5, 0.62))
        pig_mask = mb.math('MULTIPLY', pig_mask, mb.smooth(mb.noise(p, scale=600, detail=2), 0.3, 0.55))
        col = mb.mixc(mb.math('MULTIPLY', pig_mask, 0.75), col, pigment)
        rough = mb.mixf(pig_mask, rough, 0.92)
        # the freshly cut walls of the grooves are a touch lighter than the weathered face
        col = mb.mixc(mb.math('MULTIPLY', mb.smooth(g, 0.05, 0.4), 0.25), col, (0.13, 0.13, 0.12))

    # only real crevices collect dirt; micro pits would read as camouflage speckle
    dirt = mb.smooth(cav, 0.65, 1.0)
    col = mb.mixc(mb.math('MULTIPLY', dirt, 0.55), col, (0.03, 0.024, 0.017))
    if lichen > 0:
        ln = mb.noise(p, scale=26, detail=5, rough=0.65)
        lmask = mb.math('MULTIPLY', mb.smooth(ln, 0.6 - 0.04 * lichen, 0.63 - 0.04 * lichen),
                        mb.smooth(up, -0.3, 0.6))
        lmask = mb.math('MULTIPLY', lmask, mb.math('SUBTRACT', 1.0, mb.smooth(cav, 0.3, 0.8)))
        kind = mb.noise(p, scale=8, detail=2)
        lcol = mb.mixc(mb.smooth(kind, 0.62, 0.7), (0.22, 0.24, 0.17), (0.36, 0.16, 0.03))
        speck = mb.voronoi(p, scale=320, out='Distance')
        lcol = mb.mixc(mb.math('MULTIPLY', mb.smooth(speck, 0.3, 0.05), 0.5), lcol, (0.04, 0.04, 0.03))
        col = mb.mixc(lmask, col, lcol)
        rough = mb.mixf(lmask, rough, 0.95)
        height = mb.math('MULTIPLY_ADD', lmask, 0.6, height)
        height = mb.math('MULTIPLY_ADD', mb.math('MULTIPLY', lmask, speck), 0.4, height)
    nrm = mb.bump(height, 1.0, 0.00025)
    return mb.set(base=col, rough=rough, metal=0.0, normal=nrm)


def cork(name):
    mb = MB(name)
    p = mb.coord()
    pores = mb.voronoi(p, scale=260, out='Distance', randomness=1.0)
    big = mb.voronoi(p, scale=70, out='Distance')
    n = mb.noise(p, scale=120, detail=6)
    col = mb.ramp(n, [(0.3, (0.27, 0.15, 0.07)), (0.7, (0.45, 0.28, 0.14))])
    pm = mb.smooth(pores, 0.12, 0.02)
    bm_ = mb.smooth(big, 0.08, 0.0)
    col = mb.mixc(mb.math('MAXIMUM', pm, bm_), col, (0.05, 0.025, 0.012))
    cav = _cavity(mb)
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(cav, 0.2, 1.0), 0.7), col, (0.03, 0.018, 0.01))
    height = mb.math('SUBTRACT', n, mb.math('MAXIMUM', pm, mb.math('MULTIPLY', bm_, 1.5)))
    return mb.set(base=col, rough=0.88, metal=0.0, normal=mb.bump(height, 1.0, 0.0004))


def wax(name, color=(0.075, 0.009, 0.007), masks=()):
    mb = MB(name)
    p = mb.coord()
    n = mb.noise(p, scale=90, detail=5)
    col = mb.mixc(mb.smooth(n, 0.3, 0.7), color, tuple(c * 1.7 for c in color))
    cav = _cavity(mb)
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(cav, 0.3, 1.0), 0.6), col, (0.03, 0.004, 0.003))
    edge = _edge(mb, coverage=0.6)
    col = mb.mixc(mb.math('MULTIPLY', edge, 0.4), col, tuple(c * 2.4 for c in color))
    rough = mb.mixf(mb.smooth(mb.noise(p, scale=40, detail=4), 0.4, 0.6), 0.36, 0.55)
    height = mb.math('MULTIPLY', mb.noise(p, scale=400, detail=3), 0.4)
    if masks:
        g = _mask_stack(mb, masks)
        height = mb.math('MULTIPLY_ADD', g, -1.6, height)
        col = mb.mixc(mb.math('MULTIPLY', g, 0.5), col, tuple(c * 0.5 for c in color))
    return mb.set(base=col, rough=rough, metal=0.0, normal=mb.bump(height, 1.0, 0.0002), sss=0.25)


def twine(name, color=(0.28, 0.19, 0.095), wrap_pitch=0.0):
    mb = MB(name)
    p = mb.coord()
    fib = mb.noise(mb.mapping(p, scale=(1, 1, 1)), scale=2400, detail=3, rough=0.6)
    if wrap_pitch:
        turns = mb.wave(p, scale=0.31416 / wrap_pitch, axis='Z', distortion=0.3, detail=1)
        fib = mb.math('ADD', mb.math('MULTIPLY', fib, 0.4), turns)
    n = mb.noise(p, scale=200, detail=4)
    col = mb.mixc(mb.smooth(n, 0.35, 0.7), tuple(c * 0.6 for c in color), color)
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(fib, 0.6, 0.75), 0.5), col, tuple(c * 1.5 for c in color))
    cav = _cavity(mb, 0.95, 0.5)
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(cav, 0.2, 1.0), 0.8), col, (0.03, 0.02, 0.01))
    return mb.set(base=col, rough=0.95, metal=0.0, normal=mb.bump(fib, 1.0, 0.00015), sheen=0.5)


def parchment(name, masks=(), ink=(0.018, 0.012, 0.008)):
    mb = MB(name)
    p = mb.coord()
    n = mb.noise(p, scale=45, detail=7, rough=0.65)
    col = mb.ramp(n, [(0.3, (0.40, 0.30, 0.17)), (0.6, (0.56, 0.46, 0.30)), (0.85, (0.62, 0.53, 0.37))])
    stain = mb.smooth(mb.noise(p, scale=14, detail=4), 0.62, 0.68)
    col = mb.mixc(mb.math('MULTIPLY', stain, 0.35), col, (0.25, 0.15, 0.06))
    fox = mb.smooth(mb.voronoi(p, scale=160, out='Distance'), 0.06, 0.0)
    fox = mb.math('MULTIPLY', fox, mb.smooth(mb.noise(p, scale=30), 0.5, 0.6))
    col = mb.mixc(mb.math('MULTIPLY', fox, 0.6), col, (0.22, 0.10, 0.03))
    edge = _edge(mb, lo=0.5, hi=0.55, coverage=0.3)
    cav = _cavity(mb)
    col = mb.mixc(mb.math('MULTIPLY', mb.math('MAXIMUM', edge, mb.smooth(cav, 0.3, 1.0)), 0.6),
                  col, (0.12, 0.07, 0.03))
    fibres = mb.noise(p, scale=1800, detail=3)
    rough = 0.85
    if masks:
        g = _mask_stack(mb, masks)
        g = mb.math('MULTIPLY', mb.smooth(g, 0.15, 0.45),
                    mb.smooth(mb.noise(p, scale=500, detail=3), 0.2, 0.45))
        col = mb.mixc(g, col, ink)
        rough = mb.mixf(g, 0.85, 0.55)
    return mb.set(base=col, rough=rough, metal=0.0, normal=mb.bump(fibres, 0.6, 0.0001))


def page_edges(name):
    """Text-block edges: stacked leaves (tone varies leaf to leaf), grime toward the outside, foxing."""
    mb = MB(name)
    p = mb.coord()
    across = mb.mapping(p, scale=(0.03, 0.03, 1.0))  # changes from leaf to leaf, barely along a leaf
    sheet = mb.noise(across, scale=350, detail=0)
    # ~1.2 mm leaf rhythm: anything finer than a few texels aliases into blotches when baked
    leaves = mb.wave(p, scale=260, axis='Z', distortion=0.4, detail=0, detail_scale=2)
    lines = mb.smooth(leaves, 0.8, 1.0)
    col = mb.ramp(mb.noise(p, scale=30, detail=4), [(0.3, (0.40, 0.32, 0.20)), (0.7, (0.50, 0.42, 0.27))])
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(sheet, 0.35, 0.7), 0.3), col, (0.30, 0.23, 0.14))
    col = mb.mixc(mb.math('MULTIPLY', lines, 0.3), col, (0.16, 0.11, 0.06))
    fox = mb.smooth(mb.voronoi(p, scale=90, out='Distance'), 0.08, 0.0)
    fox = mb.math('MULTIPLY', fox, mb.smooth(mb.noise(p, scale=12), 0.45, 0.6))
    col = mb.mixc(mb.math('MULTIPLY', fox, 0.5), col, (0.2, 0.09, 0.03))
    cav = _cavity(mb, 0.95, 0.55)
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(cav, 0.3, 1.0), 0.6), col, (0.05, 0.035, 0.02))
    height = mb.math('ADD', lines, mb.math('MULTIPLY', sheet, 0.3))
    return mb.set(base=col, rough=0.9, metal=0.0, normal=mb.bump(height, 1.0, 0.00012))


def felt(name, color=(0.012, 0.035, 0.02)):
    mb = MB(name)
    p = mb.coord()
    fuzz = mb.noise(p, scale=3000, detail=4, rough=0.7)
    col = mb.mixc(mb.math('MULTIPLY', fuzz, 0.5), color, tuple(c * 1.8 for c in color))
    return mb.set(base=col, rough=1.0, metal=0.0, normal=mb.bump(fuzz, 0.4, 0.0002), sheen=1.0)


def silk(name, color=(0.20, 0.022, 0.025)):
    mb = MB(name)
    p = mb.coord()
    weave = mb.wave(p, scale=1400, axis='X', detail=0)
    n = mb.noise(p, scale=60, detail=4)
    col = mb.mixc(mb.smooth(n, 0.3, 0.75), tuple(c * 0.6 for c in color), color)
    edge = _edge(mb, coverage=0.4)
    col = mb.mixc(mb.math('MULTIPLY', edge, 0.6), col, (0.30, 0.18, 0.14))
    return mb.set(base=col, rough=0.45, metal=0.0, normal=mb.bump(weave, 0.3, 0.00005), sheen=1.0)


def gem(name, color=(0.055, 0.0015, 0.005), glow=(0.30, 0.018, 0.012)):
    mb = MB(name)
    p = mb.coord()
    inc = mb.noise(p, scale=500, detail=3)
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(inc, 0.6, 0.7), 0.3), color, tuple(c * 2.2 for c in color))
    silk_ = mb.smooth(mb.noise(mb.mapping(p, scale=(1, 1, 30)), scale=300, detail=2), 0.7, 0.75)
    col = mb.mixc(mb.math('MULTIPLY', silk_, 0.25), col, (0.4, 0.15, 0.15))
    rough = mb.mixf(mb.smooth(mb.noise(p, scale=800), 0.6, 0.8), 0.04, 0.12)
    cav = _cavity(mb, 0.95, 0.6)
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(cav, 0.0, 1.0), 0.8), col, (0.012, 0.0005, 0.002))
    # emission is a bake-only mask (strength ~0 here); the game material decides the glow
    return mb.set(base=col, rough=rough, metal=0.0, spec=0.75, emit=glow, emit_strength=0.001)


def sand(name):
    mb = MB(name)
    p = mb.coord()
    g = mb.voronoi(p, scale=2500, out='Color')
    gc = mb.ramp(mb.sep(g)[0], [(0.0, (0.32, 0.22, 0.11)), (0.5, (0.48, 0.36, 0.20)),
                               (0.85, (0.60, 0.50, 0.33)), (1.0, (0.12, 0.09, 0.06))])
    n = mb.noise(p, scale=200, detail=3)
    col = mb.mixc(mb.math('MULTIPLY', n, 0.3), gc, (0.40, 0.29, 0.15))
    h = mb.voronoi(p, scale=2500, feature='SMOOTH_F1', out='Distance')
    # a few grains carry a faint glint: bake-only emission mask for the game to animate
    glint = mb.smooth(mb.sep(mb.voronoi(p, scale=900, out='Color'))[1], 0.975, 0.99)
    em = mb.mixc(glint, (0, 0, 0), (0.55, 0.40, 0.16))
    return mb.set(base=col, rough=0.92, metal=0.0, normal=mb.bump(h, 0.6, 0.0001), emit=em,
                  emit_strength=0.001)


def liquid(name, deep=(0.0025, 0.012, 0.012), shallow=(0.012, 0.055, 0.042), glow=(0.03, 0.42, 0.26),
           z0=0.0, z1=0.05):
    mb = MB(name)
    z = mb.sep(mb.coord())[2]
    t = mb.maprange(z, z0, z1)
    col = mb.mixc(t, deep, shallow)
    motes = mb.smooth(mb.voronoi(mb.coord(), scale=300, out='Distance'), 0.05, 0.0)
    em = mb.mixc(mb.math('MULTIPLY', t, 0.7), mb.mixc(motes, (0, 0, 0), glow), glow)
    em = mb.mixc(0.5, (0, 0, 0), em)
    murk = mb.noise(mb.coord(), scale=60, detail=5, rough=0.6)
    col = mb.mixc(mb.math('MULTIPLY', mb.smooth(murk, 0.45, 0.75), 0.6), col, (0.02, 0.03, 0.012))
    col = mb.mixc(mb.math('MULTIPLY', motes, 0.6), col, (0.10, 0.16, 0.10))
    return mb.set(base=col, rough=0.06, metal=0.0, spec=0.6, emit=em, emit_strength=0.001)


def glass(name, tint=(0.88, 0.93, 0.89), rough=0.02):
    mb = MB(name)
    n = mb.noise(scale=40, detail=2)
    r = mb.maprange(n, 0, 1, rough, rough + 0.06)
    mb.set(base=tint, rough=r, metal=0.0, transmission=1.0, ior=1.5)
    if hasattr(mb.mat, 'surface_render_method'):
        mb.mat.surface_render_method = 'BLENDED'
    return mb.mat
