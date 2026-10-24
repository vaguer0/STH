"""
Lightning Katana -- SeekThenHide
================================

Second sword of the set.  Reuses the toolkit in ../mystic_katana/build_mystic_katana.py
(blade geometry, brush painter, handle wrap, bakes, outline, export) and adds what is
specific to this one: an icy blade with an inazuma (lightning) temper line and a
mitsudomoe crest, a gold guard with raised bolt reliefs, a gold silk wrap, and
hand-drawn lightning that strikes, branches and flickers on an irregular beat.

    blender --background --python build_lightning_katana.py
    blender --background --python build_lightning_katana.py -- --fast
"""

import math
import os
import sys

import bpy  # noqa: E402  (bpy must load before bmesh when run as a module)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mystic_katana"))
import build_mystic_katana as K  # noqa: E402

K.HERE = HERE
K.TEX_DIR = os.path.join(HERE, "textures")
K.EXPORT_DIR = os.path.join(HERE, "export")
K.RENDER_DIR = os.path.join(HERE, "renders")

# a flat guard bar sits higher than the mystic sword's, so the handle starts higher
K.Z_FUCHI_TOP = -0.0095
K.Z_WRAP_TOP = K.Z_FUCHI_TOP - 0.0120
K.GRIP_CENTRE = 0.5 * (K.Z_WRAP_TOP + K.Z_WRAP_BOT)
K.ITO_Z0 = K.Z_WRAP_TOP - 0.0045

hexc, PAL = K.hexc, K.PAL
PAL.update({
    # the shared "silver" painter becomes gold for this sword
    "silver_hi": hexc("#ffeaa0"), "silver_mid": hexc("#e0a82a"), "silver_lo": hexc("#8a5a10"),
    "silver_cav": hexc("#3a2408"),
    "ink": hexc("#0d1424"),
})
_world_and_lights = K.setup_world_and_lights


def storm_lights(col):
    """Same preview rig as the mystic sword, but with an icy rim light."""
    _world_and_lights(col)
    rim = bpy.data.objects.get("Rim")
    if rim:
        rim.data.color = tuple(K.srgb_to_linear(hexc("#8fd8ff")))


K.setup_world_and_lights = storm_lights

ICE = {
    "deep": hexc("#2d3850"), "steel": hexc("#7f93b3"), "ji": hexc("#b4c6de"), "ha": hexc("#e3f3ff"),
    "storm": hexc("#5b6f95"), "bolt": hexc("#47d4ff"), "core": hexc("#f4feff"), "deep_glow": hexc("#1f7dff"),
}


# ---------------------------------------------------------------------------
# blade texture: icy steel, inazuma temper line, mitsudomoe crest
# ---------------------------------------------------------------------------

def zigzag_hamon(rng, length):
    """Lightning temper line: sharp alternating peaks in (along, f) space."""
    a, pts, hi = 0.028, [], True
    while a < length:
        f = rng.uniform(0.79, 0.9) if hi else rng.uniform(0.46, 0.62)
        if rng.random() < 0.18:                       # a small stutter step
            f = 0.5 * (f + (0.7 if hi else 0.75))
        pts.append((a, f))
        a += rng.uniform(0.006, 0.016)
        hi = not hi
    return np.array(pts)


def paint_lightning_blade(res=(512, 4096)):
    W, H = res
    bl = K.BLADE
    cv = K.StripCanvas(W, H)
    F, A, TAU, SIDE, WID = cv.F, cv.A, cv.TAU, cv.SIDE, cv.WID
    nz, nz2 = K.Noise2(41), K.Noise2(42)
    z = lambda: np.zeros((H, W), np.float32)
    bolt, branch, crest, scratch = z(), z(), z(), z()
    hamon_f = np.zeros((H, W))
    for side in (1, -1):
        rng = np.random.default_rng(500 + side)
        zz = zigzag_hamon(rng, bl.arc(bl.TY) + 0.004)
        a_s = np.linspace(0, bl.length, 6000)
        fh = np.interp(a_s, zz[:, 0], zz[:, 1])
        s = np.clip((bl.tau_of_arc(a_s) - bl.TY) / (1 - bl.TY), 0, 1)
        fh = fh * (1 - K.smoothstep(0, 0.3, s)) + (0.78 - 0.5 * K.smoothstep(0.55, 1, s)) * K.smoothstep(0, 0.3, s)
        m = SIDE == side
        hamon_f[m] = np.interp(A[m], a_s, fh)

        def to_ad(a, f):
            d_b, d_e = bl.bounds(bl.tau_of_arc(a))
            return np.column_stack([a, d_b + f * (d_e - d_b)])

        # the bolt itself: one sharp stroke, thick at the corners like an ink drawing
        ad = to_ad(zz[:, 0], zz[:, 1])
        rad = 0.00034 + 0.00016 * rng.random(len(zz))
        cv.stamp(bolt, ad, rad, side)
        # forks splitting off some corners toward the edge or the ridge
        for i in range(1, len(zz) - 1):
            if rng.random() > 0.42:
                continue
            a0, f0 = zz[i]
            toward_edge = zz[i, 1] > 0.7
            pts, a, f = [(a0, f0)], a0, f0
            for _ in range(int(rng.integers(2, 5))):
                a += rng.uniform(0.0015, 0.005)
                f += (rng.uniform(0.03, 0.08) if toward_edge else -rng.uniform(0.04, 0.1)) * rng.choice([1, 1, -0.4])
                f = float(np.clip(f, 0.32, 0.985))
                pts.append((a, f))
            pts = np.array(pts)
            cv.stamp(branch, to_ad(pts[:, 0], pts[:, 1]), np.linspace(0.00024, 0.00005, len(pts)), side)

        # mitsudomoe crest: three tapering commas chasing each other
        a_c = 0.046
        d_b, d_e = bl.bounds(bl.tau_of_arc(a_c))
        d_c = float(d_b + 0.56 * (d_e - d_b))
        for k in range(3):
            t = np.linspace(0, 1, 30)
            th = k * 2 * math.pi / 3 + 2.6 * t
            r = 0.0006 + 0.0043 * t
            path = np.column_stack([a_c + r * np.sin(th), d_c + r * np.cos(th) * side])
            cv.stamp(crest, path, 0.0016 * (1 - t) ** 1.3 + 0.00012, side)
        ring = np.linspace(0, 2 * math.pi, 80)
        cv.stamp(crest, np.column_stack([a_c + 0.0059 * np.sin(ring), d_c + 0.0059 * np.cos(ring)]),
                 np.full(80, 0.00022), side, opacity=0.9)
        for _ in range(55):
            sa = rng.uniform(0.04, bl.length * 0.97)
            d_b, d_e = bl.bounds(bl.tau_of_arc(sa))
            sd = float(d_b + rng.uniform(0.05, 0.95) * (d_e - d_b))
            ln, th = rng.uniform(0.003, 0.012), rng.uniform(-0.3, 0.3)
            cv.stamp(scratch, np.array([[sa, sd], [sa + ln * math.cos(th), sd + ln * math.sin(th)]]),
                     np.array([0.00004, 0.00001]), side, opacity=rng.uniform(0.2, 0.6))

    dist_h = (F - hamon_f) * WID
    ha = K.smoothstep(-0.0003, 0.0003, dist_h)
    storm = nz.fbm(A / 0.03, F * 6.0, octaves=4)
    streak = nz2.fbm(cv.D / 0.0008, A / 0.05, octaves=3)
    g = np.vectorize(bl.groove)(TAU[:, 0])[:, None] * np.ones((1, W))
    hw = bl.GROOVE_HW * (0.35 + 0.65 * g)
    groove = K.smoothstep(0, 0.25, g) * (1 - K.smoothstep(hw * 0.85, hw * 1.02, np.abs(F - bl.GROOVE_C)))

    def mixc(base, col, t):
        col = col if col.ndim == 3 else col[None, None, :]
        return base * (1 - t[..., None]) + col * t[..., None]

    alb = mixc(np.broadcast_to(ICE["deep"], (H, W, 3)).copy(), ICE["steel"], K.smoothstep(0.03, 0.3, F))
    alb = mixc(alb, ICE["ji"], K.smoothstep(0.28, 0.5, F) * (1 - ha))
    alb = mixc(alb, ICE["storm"], K.smoothstep(0.52, 0.75, storm) * 0.55 * (F > bl.F_SHINOGI) * (1 - ha))
    alb = mixc(alb, ICE["ha"], ha)
    alb = mixc(alb, ICE["deep"] * 0.8, groove * 0.85)
    alb = alb * (0.92 + 0.14 * streak[..., None])
    alb = mixc(alb, hexc("#eaf4ff"),
               np.clip(np.exp(-((F - bl.F_SHINOGI) * WID / 0.00028) ** 2) * 0.8, 0, 1))
    alb = mixc(alb, hexc("#1b2233"), np.clip(np.exp(-(F * WID / 0.0004) ** 2), 0, 1))   # dark back line
    alb = alb + 0.12 * scratch[..., None]
    halo = K.gauss_blur(bolt, 10, 8) + 0.6 * K.gauss_blur(branch, 6, 5) + K.gauss_blur(crest, 7, 6)
    alb = mixc(alb, ICE["bolt"], np.clip(halo * 0.5, 0, 0.5))
    alb = mixc(alb, ICE["core"], np.clip(bolt + branch * 0.8 + crest, 0, 1))
    alb = np.clip(alb, 0, 1)

    emi = np.zeros((H, W, 3))
    for col, amt in ((ICE["bolt"], halo * 0.75), (ICE["core"], bolt * 0.95 + branch * 0.6 + crest * 0.9),
                     (ICE["bolt"], K.smoothstep(0.975, 1.0, F) * 0.8),
                     (ICE["deep_glow"], groove * (1 - K.smoothstep(0, hw * 0.4, np.abs(F - bl.GROOVE_C))) * 0.3)):
        emi = emi + col[None, None, :] * np.clip(amt, 0, None)[..., None]
    emi = np.clip(emi, 0, 1)

    height = K.gauss_blur((-0.00026 * np.clip(bolt + branch + crest, 0, 1) - 0.00003 * scratch).astype(np.float32), 1.2)
    gx = np.gradient(height, axis=1) / ((bl.U1 - bl.U0) / W * WID)
    gy = np.gradient(height, axis=0) / cv.px_v
    nrm = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    nrm = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True) * 0.5 + 0.5

    out = {k: os.path.join(K.TEX_DIR, f"katana_blade_{k}.png") for k in ("albedo", "emission", "normal")}
    K.save_rgba(out["albedo"], alb, True, "katana_blade_albedo")
    K.save_rgba(out["emission"], emi, True, "katana_blade_emission")
    K.save_rgba(out["normal"], nrm, False, "katana_blade_normal")
    return out


# ---------------------------------------------------------------------------
# hilt: gold bolt guard, habaki, gold silk wrap, bolt menuki
# ---------------------------------------------------------------------------

BOLT_ICON = [(-0.10, 0.50), (0.32, 0.50), (0.10, 0.10), (0.36, 0.10), (-0.22, -0.50), (-0.02, -0.06),
             (-0.30, -0.06)]


def bolt_outline(n=6):
    """Classic hand-drawn lightning bolt, slightly rounded at the corners."""
    P = np.array(BOLT_ICON)
    P = P - P.mean(axis=0)
    loop = np.vstack([P, P[:1]])
    out = []
    for i in range(len(P)):          # corner rounding: keep the bolt crisp but not CAD-sharp
        a, b, c = loop[i - 1] if i else P[-1], loop[i], loop[i + 1]
        out += [b + (a - b) * 0.06, b, b + (c - b) * 0.06]
    o = K.resample_polyline(np.vstack([out, out[:1]]), 8 * len(P) + 1)[:-1]
    return 0.5 * o + 0.25 * (np.roll(o, 1, 0) + np.roll(o, -1, 0))


def blade_outline_xy(tau=0.0, grow=0.0, edge_grow=None):
    """Convex outline of the blade section (x, y) grown outward, for the habaki."""
    bl = K.BLADE
    prof = bl.profile(tau)
    keep = [0, 1, 2, 8] + list(range(9, len(prof)))
    pts = [tuple(bl.pos(tau, prof[i][0], prof[i][1])[:2]) for i in keep]
    pts += [tuple(bl.pos(tau, prof[i][0], -prof[i][1])[:2]) for i in reversed(keep[1:-1])]
    pts = np.array(pts)
    pts = K.resample_polyline(np.vstack([pts, pts[:1]]), 41)[:-1]
    c = np.array([0.0, float(np.mean(pts[:, 1]))])
    d = pts - c
    r = np.linalg.norm(d, axis=1, keepdims=True)
    eg = grow if edge_grow is None else edge_grow
    g = grow + (eg - grow) * K.smoothstep(0.0, -0.016, pts[:, 1:2] - c[1])
    sm = c + d * (r + g) / r
    for _ in range(3):
        sm = 0.5 * sm + 0.25 * (np.roll(sm, 1, 0) + np.roll(sm, -1, 0))
    return sm


def build_habaki(col):
    mb = K.MeshBuilder()
    Hh = 0.030
    rings = [(Hh, 0.0004, 0.0001, 0), (Hh - 0.0004, 0.0013, 0.0006, 0), (Hh - 0.0016, 0.0019, 0.0011, 0),
             (Hh * 0.62, 0.0021, 0.0013, 0), (0.006, 0.0023, 0.0015, 0), (0.0, 0.0024, 0.0016, 0)]
    idx = K.loft_z(mb, rings, lambda g, ge, _: blade_outline_xy(0.0, g, ge)[::-1], cap_top='flat', cap_bottom='flat')
    for ri, r in enumerate(rings):            # classic slant toward the edge
        if r[0] > 0.01:
            for j in range(40):
                x, y, zz = mb.verts[idx[ri] + j]
                mb.verts[idx[ri] + j] = (x, y, zz - 0.0045 * K.sstep(0.004, -0.017, y) * (zz / Hh))
    return mb.to_object("Hilt_Habaki", col)


GUARD_HALF = 0.0035
GUARD_ZC = -GUARD_HALF


def guard_outline():
    """Plan view: a long bar along the blade width with four filed V-notches."""
    th = np.linspace(0, 2 * math.pi, 160, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    e = 3.6
    x = 0.0255 * np.sign(c) * np.abs(c) ** (2 / e)
    y = 0.0415 * np.sign(s) * np.abs(s) ** (2 / e)
    notch = sum(np.exp(-((np.angle(np.exp(1j * (th - t0)))) / 0.06) ** 2) for t0 in (0.75, 2.39, 3.89, 5.53))
    k = 1 - 0.13 * notch
    return np.column_stack([x * k, y * k])


def build_bolt_guard(col):
    mb = K.MeshBuilder()
    X, Y, Z = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])

    def plaque(outline, half, slot, z, bevel, rings):
        sub = K.MeshBuilder()
        K.extrude_shape(sub, outline, 2 * half, bevel=bevel, bevel_rings=rings)
        sub.face_mat = [slot] * len(sub.faces)
        if slot == 2:   # flat ink, almost no atlas space
            sub.uv_phys = [[(u * 0.06, v * 0.06) for u, v in f] for f in sub.uv_phys]
        K.place_local(sub, np.array([0.0, 0.0, z]), X, Y, Z)
        mb.merge(sub)

    def grow(o, d):
        t = np.roll(o, -1, 0) - np.roll(o, 1, 0)
        area = np.sum(o[:, 0] * np.roll(o[:, 1], -1) - np.roll(o[:, 0], -1) * o[:, 1])
        n = np.column_stack([t[:, 1], -t[:, 0]]) * (1 if area > 0 else -1)
        return o + n / np.linalg.norm(n, axis=1, keepdims=True) * d

    base = guard_outline()
    plaque(grow(base, 0.0009), GUARD_HALF * 0.86, 2, GUARD_ZC, 0.0, 0)
    plaque(base, GUARD_HALF, 0, GUARD_ZC, 0.3, 2)
    bolt = bolt_outline()
    for face in (1, -1):                    # raised bolts on top and underside
        for sx in (1, -1):
            o = bolt * np.array([0.020, 0.036])
            o = np.column_stack([o[:, 0] * sx + sx * 0.0145, o[:, 1] * sx])
            zc = GUARD_ZC + face * GUARD_HALF
            plaque(grow(o, 0.0007), 0.0009, 2, zc, 0.0, 0)
            plaque(o, 0.0013, 1, zc, 0.35, 2)
    return mb.to_object("Hilt_BoltGuard", col)


def gold_silk_painter(name):
    k = K.NodeKit(bpy.data.materials.new(name))
    ao, edge, hl = k.painter_basics(ao_dist=0.0035, bevel_r=0.0003, edge_gain=8.0)
    u, v_ = k.uv("UVPhys")
    tw = k.math('SINE', k.mul(k.add(u, v_), 2 * math.pi / 0.0011))
    tw2 = k.math('SINE', k.mul(k.sub(u, v_), 2 * math.pi / 0.0019))
    weave = k.add(tw, k.mul(tw2, 0.4))
    v = k.add(k.add(k.mul(hl, 0.6), k.mul(k.math('POWER', ao, 1.5), 0.4)), k.mul(weave, 0.05))
    col = k.ramp(v, [(0.0, "#3b2604"), (0.3, "#a8780e"), (0.62, "#f2c22c"), (0.92, "#fff1a0")])
    col = k.mix(col, K.lin("#fffbe0"), k.mul(edge, 0.25))
    k.finish(col, None, (1.0, 0.8, 0.0), k.mul(weave, 0.00002))
    K.PAINTER_SOCKETS[k.mat.name] = k.sockets
    return k.mat


def build_hilt(col):
    K.menuki_outline = bolt_outline       # bolt-shaped menuki instead of the moon
    parts = {
        "guard": build_bolt_guard(col),
        "habaki": build_habaki(col),
        "seppa": K.plate(col, "Hilt_Seppa", GUARD_ZC - GUARD_HALF + 0.0003, K.Z_FUCHI_TOP, 0.0147, 0.0192),
        "tsuka": K.build_tsuka_core(col),
        "ito": K.build_ito(col),
        "menuki": K.build_menuki(col),
        "fuchi": K.build_fuchi(col),
        "kashira": K.build_kashira(col),
        "mekugi": K.build_mekugi(col),
    }
    gold = K.painter("silver", "P_Gold", tone=0.95)
    mats = {
        "guard": [K.painter("silver", "P_GuardGold", tone=0.9),
                  K.painter("enamel", "P_GuardBolt", "#7a4c08", "#f0c030", "#fff2a6", "#fffbe6", "#ffe066", 0.18),
                  K.painter("enamel", "P_GuardInk", "#05070d", PAL["ink"], "#1d2638", "#1d2638")],
        "habaki": K.painter("silver", "P_Habaki", file_marks=True, tone=1.0),
        "seppa": gold, "menuki": gold,
        "tsuka": K.painter("same", "P_Samegawa"),
        "ito": gold_silk_painter("P_GoldIto"),
        "fuchi": K.painter("iron", "P_Fuchi", wave_v=0.0036),
        "kashira": K.painter("iron", "P_Kashira", wave_v=0.0075),
        "mekugi": K.painter("iron", "P_Mekugi"),
    }
    for key, obj in parts.items():
        for m in (mats[key] if isinstance(mats[key], list) else [mats[key]]):
            obj.data.materials.append(m)
    return parts


# ---------------------------------------------------------------------------
# hand-drawn lightning
# ---------------------------------------------------------------------------
# Every bolt is a flat, angular stroke built from a midpoint-displaced polyline:
# thick and blunt where the charge is, thinning to a hair at the tips, split into
# forks, in three tones (cyan body, pale-blue mid, white core) like 2D effects
# animation.  Each drawing is a fresh strike; the beat between them is irregular.

BOLT_DRAWINGS = 8
LOOP_END = 72


def jagged(p0, p1, rng, rough=0.22, levels=4):
    pts = [np.asarray(p0, float), np.asarray(p1, float)]
    amp = rough
    for _ in range(levels):
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            n = np.array([-d[1], d[0]])
            out.append(0.5 * (a + b) + n * rng.uniform(-amp, amp))
            out.append(b)
        pts = out
        amp *= 0.55
    return np.array(pts)


def strip_poly(P, w):
    """Left/right outline of a stroke with per-point half widths (miter-limited)."""
    t = np.gradient(P, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return P + n * w[:, None], P - n * w[:, None]


def build_lightning(col, mats, seed, name):
    bl = K.BLADE
    mb = K.MeshBuilder()
    rng = np.random.default_rng(seed)
    dense = np.linspace(0.0, 1.0, 4001)
    Xw = np.array([1.0, 0.0, 0.0])

    def frames(f):
        E, T, O = K.blade_edge_frames(dense, f)
        a = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(E, axis=0), axis=1))])
        return a, E, T, O

    edge, back = frames(1.0), frames(0.0)

    def to3d(fr, a, y, z, roll):
        a_arr, E, T, O = fr
        a_c = float(np.clip(a, a_arr[0], a_arr[-1]))
        Ei, Ti, Oi = (np.array([np.interp(a_c, a_arr, M[:, k]) for k in range(3)]) for M in (E, T, O))
        Ti /= np.linalg.norm(Ti)
        Oi /= np.linalg.norm(Oi)
        Yv = Oi * math.cos(roll) + Xw * math.sin(roll)
        return Ei + Ti * max(0.0, a - a_arr[-1]) + Yv * y + np.cross(Ti, Yv) * z

    def stroke(fr, P, w, roll):
        """Three-tone ribbon of one bolt path."""
        for scale, off, mat in ((1.0, 0.0, 0), (0.55, 0.00018, 1), (0.26, K.CORE_OFFSET, 2)):
            L, R = strip_poly(P, w * scale)
            for zz, flip in ((off, False), (-off, True)) if off else ((0.0, False),):
                mb.cur_mat = mat
                vl = mb.add_verts([to3d(fr, a, y, zz, roll) for a, y in L])
                vr = mb.add_verts([to3d(fr, a, y, zz, roll) for a, y in R])
                isl = mb.new_island()
                for i in range(len(P) - 1):
                    q = [vl + i, vr + i, vr + i + 1, vl + i + 1]
                    mb.add_face(q[::-1] if flip else q, None, None, isl)
        mb.cur_mat = 0

    def bolt(fr, p0, p1, w0, roll, depth=0):
        P = jagged(p0, p1, rng, rough=rng.uniform(0.2, 0.32))
        P[:, 1] = np.abs(P[:, 1] - 0.0015) + 0.0015 if depth == 0 else P[:, 1]   # hug, never cut into the steel
        s = np.linspace(0, 1, len(P))
        w = w0 * (1 - 0.7 * s ** 1.4) * (0.8 + 0.45 * rng.random(len(P)) ** 3) + 0.00015
        stroke(fr, P, w, roll)
        if depth < 2:
            for _ in range(int(rng.integers(1, 3 - depth))):
                i = int(rng.integers(len(P) // 6, len(P) - 2))
                d = P[i + 1] - P[i]
                ang = math.atan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.4, 1.1)
                ln = np.linalg.norm(p1 - p0) * rng.uniform(0.18, 0.4)
                tip = P[i] + ln * np.array([math.cos(ang), abs(math.sin(ang)) + 0.2])
                bolt(fr, P[i], tip, w[i] * 0.55, roll, depth + 1)

    total = edge[0][-1]
    # main discharges crawling along the edge (and sometimes the back)
    for _ in range(int(rng.integers(1, 3))):
        fr = edge if rng.random() < 0.8 else back
        a0 = rng.uniform(0.06, 0.45) * total
        a1 = min(a0 + rng.uniform(0.18, 0.5) * total, total * 1.02)
        bolt(fr, np.array([a0, 0.002]), np.array([a1, rng.uniform(0.004, 0.02)]),
             rng.uniform(0.0034, 0.0044), rng.uniform(-0.5, 0.5))
    # short arcs jumping off the steel
    for _ in range(int(rng.integers(3, 7))):
        fr = edge if rng.random() < 0.7 else back
        a0 = rng.uniform(0.08, 0.97) * total
        ln = rng.uniform(0.015, 0.06)
        h = rng.uniform(0.012, 0.04)
        mid = np.array([a0 + ln * 0.5, h])
        roll = rng.uniform(-0.7, 0.7)
        for p, q in ((np.array([a0, 0.001]), mid), (mid, np.array([a0 + ln, 0.001]))):
            P = jagged(p, q, rng, rough=0.3, levels=3)
            s = np.linspace(0, 1, len(P))
            stroke(fr, P, 0.0022 * (0.45 + 0.55 * np.sin(np.pi * s)) + 0.0002, roll)
    # sparks
    for _ in range(int(rng.integers(6, 13))):
        fr = edge if rng.random() < 0.7 else back
        i = int(rng.uniform(0.15, 1.0) * 4000)
        E, T, O = fr[1][i], fr[2][i], fr[3][i]
        P = E + O * rng.uniform(0.006, 0.04) + Xw * rng.uniform(-0.012, 0.012)
        Zs = np.array([1.0, rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2)])
        Zs /= np.linalg.norm(Zs)
        Xs = np.cross(np.array([0.0, 0.0, 1.0]), Zs)
        Xs /= np.linalg.norm(Xs)
        Ys = np.cross(Zs, Xs)
        R = rng.uniform(0.0008, 0.0022)
        K.flat_shape(mb, K.spark_outline(R, 1.8), P, Xs, Ys, Zs, 0.0, mat=0)
        K.flat_shape(mb, K.spark_outline(R * 0.4, 1.6, 4), P, Xs, Ys, Zs, K.CORE_OFFSET, mat=2, both_sides=True)

    obj = mb.to_object(name, col, smooth=False)
    for m in mats:
        obj.data.materials.append(m)
    obj.visible_shadow = False
    return obj


def strike_schedule(seed=99):
    """Irregular beat: drawings held 1-3 frames, now and then a dark gap."""
    rng = np.random.default_rng(seed)
    sched, f, prev = [], 1, -1
    while f <= LOOP_END:
        if rng.random() < 0.18 and prev != -1:
            hold = int(rng.integers(1, 3))
            sched.append((f, hold, -1))
            prev = -1
        else:
            d = int(rng.integers(0, BOLT_DRAWINGS))
            while d == prev:
                d = int(rng.integers(0, BOLT_DRAWINGS))
            hold = int(rng.choice([1, 1, 2, 2, 2, 3]))
            sched.append((f, hold, d))
            prev = d
        f += hold
    return sched


def animate_strikes(drawings, sched, blade_mat):
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.frame_start, scene.frame_end = 1, LOOP_END
    for i, o in enumerate(drawings):
        last = None
        for f0, hold, d in sched:
            on = d == i
            if on != last:
                o.scale = (1.0, 1.0, 1.0) if on else (0.0, 0.0, 0.0)
                o.keyframe_insert("scale", frame=f0)
                last = on
        K.set_interpolation(o, 'CONSTANT')
    # the steel flashes with every strike and sinks in the gaps
    bsdf = next(n for n in blade_mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    sock = bsdf.inputs["Emission Strength"]
    base = sock.default_value
    rng = np.random.default_rng(7)
    for f0, hold, d in sched:
        for k in range(hold):
            sock.default_value = base * (0.55 if d < 0 else (rng.uniform(1.5, 2.1) if k == 0 else rng.uniform(1.0, 1.3)))
            sock.keyframe_insert("default_value", frame=f0 + k)
    sock.default_value = base
    K.set_interpolation(blade_mat.node_tree, 'CONSTANT')
    scene.frame_set(1)


def render_strike_video(col, drawings, sched, out_path):
    scene = bpy.context.scene
    o = -K.GRIP_CENTRE
    cam = K.add_camera(col, "Cam_loop", (1.62, 0.36, 0.30 + o), (0, 0.012, 0.25 + o), lens=50)
    res = (540, 960) if K.FAST else (720, 1280)
    tmp = os.path.join(K.RENDER_DIR, "_strike_frames")
    os.makedirs(tmp, exist_ok=True)
    stills = {}
    for f0, hold, d in sched:               # one still per drawing (and one dark gap)
        if d in stills:
            continue
        scene.frame_set(f0)
        path = os.path.join(tmp, f"strike_{d + 1:02d}.png")
        K.render_view(cam, path, res=res, samples=16 if K.FAST else 40)
        stills[d] = path
    scene.frame_set(1)
    vse = bpy.data.scenes.new("StrikeVideo")
    vse.render.fps = 24
    vse.render.resolution_x, vse.render.resolution_y = res
    vse.render.resolution_percentage = 100
    ed = vse.sequence_editor_create()
    coll = ed.strips if hasattr(ed, "strips") else ed.sequences
    f = 1
    for _ in range(3):
        for f0, hold, d in sched:
            st = coll.new_image(name=f"s{f}", filepath=stills[d], channel=1, frame_start=f)
            st.frame_final_duration = hold
            f += hold
    vse.frame_start, vse.frame_end = 1, f - 1
    if hasattr(vse.render.image_settings, "media_type"):
        vse.render.image_settings.media_type = 'VIDEO'
    vse.render.image_settings.file_format = 'FFMPEG'
    vse.render.ffmpeg.format = 'MPEG4'
    vse.render.ffmpeg.codec = 'H264'
    vse.render.ffmpeg.constant_rate_factor = 'HIGH'
    vse.render.filepath = out_path
    bpy.ops.render.render(animation=True, scene=vse.name)
    bpy.data.scenes.remove(vse)
    tiles = []
    for d in sorted(k for k in stills if k >= 0):
        img = bpy.data.images.load(stills[d])
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        tiles.append(px.reshape(h, w, 4)[h // 3:, :, :3])
        bpy.data.images.remove(img)
    for p in stills.values():
        os.remove(p)
    os.rmdir(tmp)
    K.save_rgba(out_path[:-4] + "_frames.png", np.concatenate(tiles, axis=1), name="strike_frames")


# ---------------------------------------------------------------------------

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = 'METRIC'
    col = K.get_collection("LightningKatana")

    blade = K.build_blade(col)
    paths = paint_lightning_blade((256, 2048) if K.FAST else (512, 4096))
    K.copy_uv(blade, "UVPhys")
    blade.data.materials.append(K.blade_material(paths))

    parts = build_hilt(col)
    fx_mats = [K.flat_emissive_material("M_Bolt_Outer", "#1fb6ff", 2.4),
               K.flat_emissive_material("M_Bolt_Mid", "#9eeaff", 2.8),
               K.flat_emissive_material("M_Bolt_Core", "#f6feff", 3.2)]
    drawings = [build_lightning(col, fx_mats, 3100 + i, f"FX_Lightning_F{i}") for i in range(BOLT_DRAWINGS)]
    K.shift_to_grip([blade] + drawings + list(parts.values()))

    hilt_objs = list(parts.values())
    res = 1024 if K.FAST else 2048
    K.pack_atlas(hilt_objs, res)
    hpaths = K.bake_hilt(hilt_objs, res, 8 if K.FAST else 48)
    hilt_mat = K.assign_final_hilt_material(parts, hpaths)

    ink = K.ink_material()
    outline = K.build_outline([parts[k] for k in ("guard", "habaki", "seppa", "fuchi", "kashira", "ito")], col, ink)

    root = bpy.data.objects.new("LightningKatana", None)
    col.objects.link(root)
    fx_group = bpy.data.objects.new("FX_Lightning", None)
    col.objects.link(fx_group)
    fx_group.parent = root
    for o in [blade, outline] + hilt_objs:
        o.parent = root
    for o in drawings:
        o.parent = fx_group
    sched = strike_schedule()
    animate_strikes(drawings, sched, blade.data.materials[0])

    if not K.NO_RENDER:
        K.render_previews(col)
        render_strike_video(col, drawings, sched, os.path.join(K.RENDER_DIR, "katana_lightning_loop.mp4"))

    os.makedirs(K.EXPORT_DIR, exist_ok=True)
    K.setup_live_viewport()
    K.tidy_preview_rig()
    blend_path = os.path.join(HERE, "lightning_katana.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True, relative_remap=True)
    for img in list(bpy.data.images):
        if img.users == 0:
            bpy.data.images.remove(img)
    for img in bpy.data.images:
        if img.source == 'FILE' and img.packed_file is None:
            img.pack()
    bpy.ops.wm.save_mainfile(compress=True)

    K.swap_ink_for_export(ink)
    stats = K.export_glb([blade, outline] + drawings + hilt_objs, os.path.join(K.EXPORT_DIR, "lightning_katana.glb"),
                         root_name="LightningKatana", clip_name="LightningLoop")
    print("[export] triangles:", stats, "total:", sum(stats.values()))


if __name__ == "__main__":
    main()
