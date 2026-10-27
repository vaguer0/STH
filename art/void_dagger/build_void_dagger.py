"""
Void Dagger -- SeekThenHide
===========================

The purple dagger, redesigned in the language of the reference daggers: a heavy,
crescent-curved blade with a convex cutting edge, backward hooks and a breaker notch
on the spine, a claw guard sweeping toward the blade and the grip, a glowing gem, a
purple-wrapped grip and a spiked pommel.  The blade carries glowing void cracks and a
hand-painted two-tone bevel, and hand-drawn shadow flames lick off the edge.

Reuses the toolkit in ../mystic_katana/build_mystic_katana.py.

    blender --background --python build_void_dagger.py
    blender --background --python build_void_dagger.py -- --fast
"""

import math
import os
import sys

import bpy  # noqa: E402  (bpy must load before bmesh when run as a module)
import numpy as np
from mathutils import Vector
from mathutils.geometry import delaunay_2d_cdt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mystic_katana"))
import build_mystic_katana as K  # noqa: E402

K.HERE = HERE
K.TEX_DIR = os.path.join(HERE, "textures")
K.EXPORT_DIR = os.path.join(HERE, "export")
K.RENDER_DIR = os.path.join(HERE, "renders")

# dagger grip: shorter and slimmer than the katanas'
K.Z_FUCHI_TOP = -0.0215
K.Z_WRAP_TOP = -0.0300
K.Z_WRAP_BOT = -0.1260
K.GRIP_CENTRE = 0.5 * (K.Z_WRAP_TOP + K.Z_WRAP_BOT)
K.ITO_PITCH = 0.031
K.ITO_W = 0.0058
K.ITO_T = 0.0008
K.ITO_Z0 = K.Z_WRAP_TOP - 0.004


def dagger_handle_axes(z):
    t = np.clip((K.Z_WRAP_TOP - z) / (K.Z_WRAP_TOP - K.Z_WRAP_BOT), 0, 1)
    waist = np.sin(np.pi * t) ** 1.2
    return 0.0068 - 0.0005 * waist, 0.0089 - 0.0007 * waist


K.handle_axes = dagger_handle_axes

hexc, PAL = K.hexc, K.PAL
PAL.update({
    "iron_dark": hexc("#140b1c"), "iron_mid": hexc("#3a2350"),
    "silver_lo": hexc("#6c4c96"), "silver_mid": hexc("#b896e6"), "silver_hi": hexc("#efe2ff"),
    "silver_cav": hexc("#120818"),
    "jade_hi": hexc("#fbf4ff"), "jade_mid": hexc("#b26bff"), "jade_lo": hexc("#3d0e6b"),
    "ink": hexc("#0c0612"),
})
VOID = {"deep": hexc("#140a1d"), "body": hexc("#2a1640"), "mottle": hexc("#47265f"),
        "bevel": hexc("#8a6fb3"), "edge": hexc("#e2d4f7"), "crack": hexc("#9a40ff"),
        "core": hexc("#dcbcff"), "ink": hexc("#0a0510")}

_world_and_lights = K.setup_world_and_lights


def void_lights(col):
    _world_and_lights(col)
    rim = bpy.data.objects.get("Rim")
    if rim:
        rim.data.color = tuple(K.srgb_to_linear(hexc("#b98aff")))


K.setup_world_and_lights = void_lights


# ---------------------------------------------------------------------------
# blade outline (blade plane: y across, z along; edge toward -y)
# ---------------------------------------------------------------------------
# (y, z, sharp) -- sharp points survive the smoothing, so hooks and the tip stay
# crisp while the long sweeps of the edge become fluid curves.

BLADE_PTS = [
    (0.0160, 0.000, 1), (-0.0160, 0.000, 1),                 # ricasso base
    (-0.0170, 0.018, 0), (-0.0235, 0.055, 0), (-0.0300, 0.105, 0), (-0.0300, 0.150, 0),
    (-0.0225, 0.200, 0), (-0.0070, 0.248, 0), (0.0150, 0.282, 0), (0.0345, 0.306, 1),   # tip
    (0.0230, 0.284, 0), (0.0110, 0.258, 0), (0.0055, 0.226, 0),
    (0.0085, 0.207, 1), (0.0330, 0.171, 1), (0.0140, 0.181, 1),                         # upper hook
    (0.0135, 0.155, 0), (0.0150, 0.134, 1), (0.0365, 0.099, 1), (0.0180, 0.110, 1),     # lower hook
    (0.0185, 0.085, 0), (0.0190, 0.064, 1), (0.0255, 0.055, 1), (0.0200, 0.046, 1),     # breaker tooth
    (0.0205, 0.030, 0), (0.0170, 0.014, 0),
]
BLADE_T = 0.0017      # half thickness of the flat plate
BLADE_D = 0.0045      # width of the ground edge bevel


def chaikin(pts, sharp, iters=4):
    """Corner-cutting that leaves the points flagged sharp exactly where they are."""
    P, S = np.asarray(pts, float), list(sharp)
    for _ in range(iters):
        out, so = [], []
        n = len(P)
        for i in range(n):
            a, b, sb = P[i], P[(i + 1) % n], S[(i + 1) % n]
            if S[i]:
                out.append(a)
                so.append(1)
            else:
                out.append(0.75 * a + 0.25 * b)
                so.append(0)
            if not sb:
                out.append(0.25 * a + 0.75 * b)
                so.append(0)
        P, S = np.array(out), so
    return P


def blade_outline(spacing=0.0011):
    pts = [(y, z) for y, z, _ in BLADE_PTS]
    sharp = [s for _, _, s in BLADE_PTS]
    P = chaikin(pts, sharp)
    L = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum()
    P = K.resample_polyline(np.vstack([P, P[:1]]), int(L / spacing) + 1)[:-1]
    area = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return P if area > 0 else P[::-1]


def dist_to_outline(Q, P, chunk=20000):
    """Distance from points Q (n,2) to the closed polyline P."""
    A, B = P, np.roll(P, -1, axis=0)
    AB = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-18)
    out = np.empty(len(Q))
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk, None, :]
        t = np.clip(np.einsum("qij,ij->qi", q - A[None], AB) / L2, 0, 1)
        d = np.linalg.norm(q - (A[None] + t[..., None] * AB[None]), axis=2)
        out[s:s + chunk] = d.min(axis=1)
    return out


def inside(Q, P):
    y, z = Q[:, 0], Q[:, 1]
    res = np.zeros(len(Q), bool)
    A, B = P, np.roll(P, -1, axis=0)
    for (y1, z1), (y2, z2) in zip(A, B):
        cond = ((z1 > z) != (z2 > z))
        xint = (y2 - y1) * (z - z1) / np.where(z2 - z1 == 0, 1e-12, z2 - z1) + y1
        res ^= cond & (y < xint)
    return res


def blade_thickness(d):
    """Flat plate with a narrow ground bevel at the edge -- no swollen middle."""
    return BLADE_T * np.clip(d / BLADE_D, 0, 1) ** 0.8


def build_dagger_blade(col):
    P = blade_outline()
    B = len(P)
    ymin, ymax = P[:, 0].min(), P[:, 0].max()
    g = 0.0024
    ys, zs = np.meshgrid(np.arange(ymin, ymax, g), np.arange(0.0, P[:, 1].max(), g))
    G = np.column_stack([ys.ravel(), zs.ravel()])
    G = G + (np.random.default_rng(3).random(G.shape) - 0.5) * g * 0.35
    keep = inside(G, P)
    G = G[keep]
    G = G[dist_to_outline(G, P) > 0.7 * g]
    pts = np.vstack([P, G])
    vco = [Vector((float(a), float(b))) for a, b in pts]
    res = delaunay_2d_cdt(vco, [(i, (i + 1) % B) for i in range(B)], [list(range(B))], 1, 1e-9)
    verts, _, faces, orig_verts = res[0], res[1], res[2], res[3]
    V = np.array([(v.x, v.y) for v in verts])
    is_edge = np.array([any(o < B for o in ov) for ov in orig_verts])
    d = dist_to_outline(V, P)
    d[is_edge] = 0.0
    t = blade_thickness(d)
    mb = K.MeshBuilder()
    Wy = ymax - ymin
    Hz = 0.31

    def uv(i, side):
        y, z = V[i]
        u = 0.5 * (y - ymin) / Wy if side > 0 else 0.5 + 0.5 * (ymax - y) / Wy
        return (u, z / Hz)

    top = mb.add_verts([(t[i], V[i, 0], V[i, 1]) for i in range(len(V))])
    bot = mb.add_verts([(-t[i], V[i, 0], V[i, 1]) for i in range(len(V))])
    bidx = lambda i: top + i if is_edge[i] else bot + i
    for f in faces:
        f = list(f)
        mb.add_face([top + i for i in f], [uv(i, 1) for i in f], [uv(i, 1) for i in f], 0)
        fr = f[::-1]
        mb.add_face([bidx(i) for i in fr], [uv(i, -1) for i in fr], [uv(i, -1) for i in fr], 0)
    for i in range(B):
        j = (i + 1) % B
        mb.sharp.add((top + i, top + j))
    obj = mb.to_object("Dagger_Blade", col)
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if bm.calc_volume(signed=True) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(obj.data)
    bm.free()
    obj["outline"] = P.ravel().tolist()
    return obj, P


# ---------------------------------------------------------------------------
# blade texture: obsidian, ground bevel, void cracks
# ---------------------------------------------------------------------------

def paint_dagger_blade(P, res=(1024, 2048)):
    W, H = res
    ymin, ymax = P[:, 0].min(), P[:, 0].max()
    Wy, Hz = ymax - ymin, 0.31
    half = W // 2
    yy = ymin + (np.arange(half) + 0.5) / half * Wy
    zz = (np.arange(H) + 0.5) / H * Hz
    Y, Z = np.meshgrid(yy, zz)
    Q = np.column_stack([Y.ravel(), Z.ravel()])
    print("[dagger] distance field ...")
    D = dist_to_outline(Q, P).reshape(H, half)
    IN = inside(Q, P).reshape(H, half)
    px = Wy / half
    aa = px * 1.2
    nz, nz2 = K.Noise2(71), K.Noise2(72)
    crack = np.zeros((H, half), np.float32)

    def stamp(pts, radii, layer, opacity=1.0):
        for (p0, p1, r0, r1) in zip(pts[:-1], pts[1:], radii[:-1], radii[1:]):
            rm = max(r0, r1) + 2 * aa
            c0 = int(max(0, (min(p0[0], p1[0]) - rm - ymin) / px - 1))
            c1 = int(min(half, (max(p0[0], p1[0]) + rm - ymin) / px + 2))
            r0_ = int(max(0, (min(p0[1], p1[1]) - rm) / Hz * H - 1))
            r1_ = int(min(H, (max(p0[1], p1[1]) + rm) / Hz * H + 2))
            if c1 <= c0 or r1_ <= r0_:
                continue
            y_, z_ = Y[r0_:r1_, c0:c1], Z[r0_:r1_, c0:c1]
            d = p1 - p0
            l2 = float(d @ d) or 1e-18
            tt = np.clip(((y_ - p0[0]) * d[0] + (z_ - p0[1]) * d[1]) / l2, 0, 1)
            dist = np.hypot(y_ - (p0[0] + tt * d[0]), z_ - (p0[1] + tt * d[1]))
            cov = np.clip((r0 + (r1 - r0) * tt - dist) / aa + 0.5, 0, 1) * opacity
            layer[r0_:r1_, c0:c1] = np.maximum(layer[r0_:r1_, c0:c1], cov)

    rng = np.random.default_rng(11)

    def d_at(p):
        return float(dist_to_outline(np.array([p]), P)[0]) if inside(np.array([p]), P)[0] else -1

    def walk(p, ang, steps, r, depth):
        pts = [p.copy()]
        for _ in range(steps):
            ang2 = ang + rng.uniform(-0.9, 0.9)
            q = pts[-1] + rng.uniform(0.004, 0.009) * np.array([math.sin(ang2), math.cos(ang2)])
            if d_at(q) < 0.0045:
                break
            pts.append(q)
            if depth < 2 and rng.random() < 0.22:
                walk(q, ang + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2), int(rng.integers(2, 5)), r * 0.6, depth + 1)
        if len(pts) > 1:
            pts = np.array(pts)
            stamp(pts, np.linspace(r, r * 0.25, len(pts)), crack)

    for k in range(12):
        for _try in range(40):
            p = np.array([rng.uniform(ymin, ymax), 0.012 + 0.23 * (k + rng.random()) / 12])
            if d_at(p) > 0.009:
                break
        walk(p, rng.uniform(-0.5, 0.5) + (0 if rng.random() < 0.7 else math.pi), int(rng.integers(5, 12)), 0.0006, 0)

    def mixc(base, col, t):
        return base * (1 - t[..., None]) + col[None, None, :] * t[..., None]

    mott = nz.fbm(Y / 0.012, Z / 0.03, octaves=4)
    streak = nz2.fbm(Y / 0.0012 + 2 * nz(Z / 0.05, 1.0), Z / 0.06, octaves=3)
    bevel = 1 - K.smoothstep(0.0, BLADE_D * 0.85, D)
    alb = mixc(np.broadcast_to(VOID["deep"], (H, half, 3)).copy(), VOID["body"], K.smoothstep(0.3, 0.7, mott))
    alb = mixc(alb, VOID["mottle"], K.smoothstep(0.62, 0.8, mott) * 0.6)
    alb = alb * (0.9 + 0.18 * streak[..., None])
    alb = mixc(alb, VOID["bevel"], bevel ** 1.4)
    alb = mixc(alb, VOID["edge"], K.smoothstep(0.0012, 0.0, D))
    bevel_line = np.exp(-((D - BLADE_D * 0.86) / 0.00045) ** 2)
    alb = mixc(alb, VOID["ink"], np.clip(bevel_line * 0.9, 0, 1))
    halo = K.gauss_blur(crack, 7)
    carve = np.clip(K.gauss_blur(crack, 2.0) * 2.2 - crack * 2, 0, 1)
    alb = mixc(alb, VOID["ink"], carve * 0.8)
    alb = mixc(alb, VOID["crack"], np.clip(halo * 0.6, 0, 0.6) * (1 - carve))
    alb = mixc(alb, VOID["core"], crack)
    alb = np.clip(alb, 0, 1)
    emi = (VOID["crack"][None, None] * np.clip(halo * 0.9, 0, 1)[..., None]
           + VOID["core"][None, None] * crack[..., None]
           + VOID["crack"][None, None] * (K.smoothstep(0.0009, 0.0, D) * 0.55)[..., None])
    emi = np.clip(emi, 0, 1)
    h = K.gauss_blur((-0.00025 * crack).astype(np.float32), 1.2)
    gx = np.gradient(h, axis=1) / px
    gy = np.gradient(h, axis=0) / (Hz / H)
    nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    nrm = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True) * 0.5 + 0.5
    full = lambda a: np.concatenate([a, a[:, ::-1]], axis=1)
    nflip = nrm[:, ::-1].copy()
    nflip[..., 0] = 1 - nflip[..., 0]
    out = {k: os.path.join(K.TEX_DIR, f"dagger_blade_{k}.png") for k in ("albedo", "emission", "normal")}
    K.save_rgba(out["albedo"], full(alb), True, "dagger_blade_albedo")
    K.save_rgba(out["emission"], full(emi), True, "dagger_blade_emission")
    K.save_rgba(out["normal"], np.concatenate([nrm, nflip], axis=1), False, "dagger_blade_normal")
    return out


# ---------------------------------------------------------------------------
# hilt: claw guard, gem, grip, spiked pommel
# ---------------------------------------------------------------------------

# Guard drawn in the blade plane (y across, z along), like the reference daggers:
# a thick shield-shaped boss and, on each side, a heavy sickle wing sweeping up
# toward the blade plus a shorter one hooking down toward the grip.  (y, z, sharp)
GUARD_BOSS = [(0.0, 0.0065, 1), (0.0150, 0.0040, 0), (0.0215, -0.0080, 0), (0.0140, -0.0215, 0),
              (0.0, -0.0270, 1), (-0.0140, -0.0215, 0), (-0.0215, -0.0080, 0), (-0.0150, 0.0040, 0)]
GUARD_WING_UP = [(0.0120, -0.0110, 0), (0.0300, -0.0130, 0), (0.0470, -0.0050, 0), (0.0575, 0.0110, 0),
                 (0.0560, 0.0330, 0), (0.0440, 0.0560, 1), (0.0470, 0.0330, 0), (0.0430, 0.0150, 0),
                 (0.0330, 0.0060, 1), (0.0300, 0.0110, 0), (0.0200, 0.0040, 0), (0.0120, 0.0030, 0)]
GUARD_WING_DOWN = [(0.0120, -0.0140, 0), (0.0270, -0.0175, 0), (0.0380, -0.0270, 0), (0.0430, -0.0480, 1),
                   (0.0335, -0.0330, 0), (0.0250, -0.0275, 1), (0.0230, -0.0225, 0), (0.0120, -0.0215, 0)]
GUARD_BOSS_HALF = 0.0105


def guard_shape(pts, sy=1.0, scale=1.0):
    P = chaikin([(y * sy * scale, z * scale) for y, z, _ in pts], [s for _, _, s in pts], iters=3)
    area = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return P if area > 0 else P[::-1]


def build_guard(col):
    mb = K.MeshBuilder()
    X, Y, Z = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])

    def plaque(outline, half, bevel=0.42):
        sub = K.MeshBuilder()
        K.extrude_shape(sub, outline, 2 * half, bevel=bevel, bevel_rings=3)
        K.place_local(sub, np.zeros(3), Y, Z, X)   # local x -> Y, local y -> Z, thickness -> X
        mb.merge(sub)

    plaque(guard_shape(GUARD_BOSS), GUARD_BOSS_HALF)
    for sy, sc in ((1.0, 1.0), (-1.0, 1.12)):      # the edge side (-y) wing is the bigger one
        plaque(guard_shape(GUARD_WING_UP, sy, sc), 0.0048)
        plaque(guard_shape(GUARD_WING_DOWN, sy, sc), 0.0040)
    return mb.to_object("Hilt_Guard", col)


def build_gem(col):
    mb = K.MeshBuilder()
    th = np.linspace(0, 2 * math.pi, 9, endpoint=False) + 0.2
    oct_ = np.column_stack([np.cos(th) * 0.0062, np.sin(th) * 0.0078])
    for sx in (1, -1):
        sub = K.MeshBuilder()
        K.extrude_shape(sub, oct_, 0.0042, bevel=0.55, bevel_rings=2)
        K.place_local(sub, np.array([sx * (GUARD_BOSS_HALF + 0.0011), 0.0, -0.0095]), np.array([0, sx * 1.0, 0]),
                      np.array([0, 0, 1.0]), np.array([sx * 1.0, 0, 0]))
        mb.merge(sub)
    return mb.to_object("Hilt_Gem", col)


def spike(mb, ctrl, r0, flat=0.55, twist=0.6):
    path = K.catmull_rom([np.array(c, float) for c in ctrl], 10)
    path = K.resample_polyline(path, 30)
    T, N, Bv = K.rotation_minimizing_frames(path, np.array([1.0, 0.0, 0.0]))
    s = np.linspace(0, 1, len(path))
    profs = []
    for k, sk in enumerate(s):
        r = r0 * (1 - sk) ** 0.85 + 0.00015
        ph = np.array([0, 1, 2, 3, 4, 5]) * math.pi / 3 + twist * sk
        rr = r * np.array([1.0, 0.62, 0.62, 1.0, 0.62, 0.62])
        profs.append(np.column_stack([rr * np.cos(ph), flat * rr * np.sin(ph)]))
    K.sweep(mb, path, (N, Bv), profs, cap_start='flat', cap_end=0.003, sharp_cols=(0, 3))


def build_pommel(col):
    mb = K.MeshBuilder()
    zb = K.Z_WRAP_BOT
    hx, hy = dagger_handle_axes(zb)

    def outline(sc, e):
        return K.superellipse(36, hx * sc, hy * sc, e)

    rings = [(zb + 0.001, 1.04, 2.3), (zb - 0.001, 1.22, 2.4), (zb - 0.005, 1.26, 2.5), (zb - 0.008, 1.05, 2.4),
             (zb - 0.010, 1.12, 2.4), (zb - 0.016, 0.86, 2.2), (zb - 0.026, 0.58, 2.1), (zb - 0.036, 0.32, 2.0),
             (zb - 0.042, 0.12, 2.0)]
    K.loft_z(mb, rings, outline, cap_top=None, cap_bottom=0.007, sharp_rings=(3, 4))
    for sy in (1, -1):
        spike(mb, [(0, sy * 0.010, zb - 0.006), (0, sy * 0.020, zb - 0.010), (0, sy * 0.025, zb - 0.022)], 0.0048,
              flat=0.45)
    return mb.to_object("Hilt_Pommel", col)


def violet_silk_painter(name):
    k = K.NodeKit(bpy.data.materials.new(name))
    ao, edge, hl = k.painter_basics(ao_dist=0.0035, bevel_r=0.0003, edge_gain=8.0)
    u, v_ = k.uv("UVPhys")
    weave = k.add(k.math('SINE', k.mul(k.add(u, v_), 2 * math.pi / 0.0011)),
                  k.mul(k.math('SINE', k.mul(k.sub(u, v_), 2 * math.pi / 0.0019)), 0.4))
    v = k.add(k.add(k.mul(hl, 0.6), k.mul(k.math('POWER', ao, 1.5), 0.4)), k.mul(weave, 0.05))
    col = k.ramp(v, [(0.0, "#16061f"), (0.3, "#4a1675"), (0.62, "#8d3fd6"), (0.92, "#d2a6ff")])
    col = k.mix(col, K.lin("#f1e2ff"), k.mul(edge, 0.25))
    k.finish(col, None, (1.0, 0.8, 0.0), k.mul(weave, 0.00002))
    K.PAINTER_SOCKETS[k.mat.name] = k.sockets
    return k.mat


def build_hilt(col):
    parts = {
        "guard": build_guard(col),
        "gem": build_gem(col),
        "tsuka": K.build_tsuka_core(col),
        "ito": K.build_ito(col),
        "fuchi": K.build_fuchi(col),
        "pommel": build_pommel(col),
    }
    metal = K.painter("iron", "P_VoidMetal")
    mats = {"guard": metal, "pommel": metal, "fuchi": K.painter("iron", "P_Fuchi", wave_v=0.0036),
            "gem": K.painter("jade", "P_VoidGem"), "tsuka": K.painter("same", "P_Samegawa"),
            "ito": violet_silk_painter("P_VioletIto")}
    for key, obj in parts.items():
        obj.data.materials.append(mats[key])
    return parts


# ---------------------------------------------------------------------------
# hand-drawn shadow flames along the cutting edge
# ---------------------------------------------------------------------------

def edge_curve(P):
    """The convex cutting edge (from the ricasso to the tip) as an (a, E, T, O) frame set."""
    n = len(P)
    i0 = int(np.argmin(np.hypot(P[:, 0] + 0.016, P[:, 1] - 0.0)))
    i1 = int(np.argmax(P[:, 1]))
    fwd = [(i0 + k) % n for k in range((i1 - i0) % n + 1)]
    bwd = [(i0 - k) % n for k in range((i0 - i1) % n + 1)]
    idx = fwd if P[fwd, 0].mean() < P[bwd, 0].mean() else bwd     # the edge is the -y side
    C = P[idx]
    a = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    t = np.gradient(C, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    o = np.column_stack([t[:, 1], -t[:, 0]])
    if np.mean(o[:, 0]) > 0:                                # make sure it points away from the blade (-y)
        o = -o
    E = np.column_stack([np.zeros(len(C)), C])
    T = np.column_stack([np.zeros(len(C)), t])
    O = np.column_stack([np.zeros(len(C)), o])
    return a, E, T, O


def build_shadow_fire(col, mats, P, phase, name):
    mb = K.MeshBuilder()
    a_arr, E, T, O = edge_curve(P)
    Xw = np.array([1.0, 0, 0])

    def to3d(poly, z, roll):
        out = []
        for a, y in poly:
            a_c = float(np.clip(a, a_arr[0], a_arr[-1]))
            Ei, Ti, Oi = (np.array([np.interp(a_c, a_arr, M[:, k]) for k in range(3)]) for M in (E, T, O))
            Ti /= np.linalg.norm(Ti)
            Oi /= np.linalg.norm(Oi)
            Yv = Oi * math.cos(roll) + Xw * math.sin(roll)
            out.append(Ei + Ti * max(0.0, a - a_arr[-1]) + Yv * y + np.cross(Ti, Yv) * z)
        return out

    crng = np.random.default_rng(909)
    total = a_arr[-1]
    a, k = total * 0.08, 0
    while a < total * 0.9:
        span = crng.uniform(0.025, 0.05)
        layers = K.flame_clump(crng, span, seed=950 + k, scale=0.55 + 0.25 * K.sstep(0.0, 0.6, a / total), phase=phase)
        roll = crng.uniform(-0.45, 0.45)
        for polys, off, mat in zip(layers, (0.0, 0.00018, K.CORE_OFFSET), (0, 1, 2)):
            for poly in polys:
                poly = poly + np.array([a, 0.0])
                tris = K.triangulate_polygon(poly)
                for zz, flip in ((off, False), (-off, True)) if off else ((0.0, False),):
                    mb.cur_mat = mat
                    base = mb.add_verts(to3d(poly, zz, roll))
                    isl = mb.new_island()
                    for t in tris:
                        mb.add_face([base + i for i in (t[::-1] if flip else t)], None, None, isl)
        mb.cur_mat = 0
        k += 1
        a += span * crng.uniform(0.6, 1.0) + (crng.uniform(0.008, 0.03) if crng.random() < 0.65 else 0.0)
    # wisps streaming back off the spine hooks
    for j, (y, z) in enumerate(((0.0330, 0.171), (0.0365, 0.099), (0.0345, 0.306))):
        fh = 0.012 * (1 + 0.25 * K.wave(phase, j * 0.31))
        o2, c2 = K.flame_outlines(fh, fh * 0.4, 0.9, 0.8 * (1 if j % 2 else -1), 0.0, 1200 + j)
        Tv = np.array([0.0, 0.35, -1.0]) if j < 2 else np.array([0.0, 0.6, 0.8])
        Tv /= np.linalg.norm(Tv)
        Yv = np.array([0.0, 0.9, 0.4]) if j < 2 else np.array([0.0, 0.8, -0.6])
        Yv = Yv - Tv * (Yv @ Tv)
        Yv /= np.linalg.norm(Yv)
        Pp = np.array([0.0, y, z])
        K.flat_shape(mb, o2, Pp, Tv, Yv, np.cross(Tv, Yv), 0.0, mat=0)
        K.flat_shape(mb, c2, Pp, Tv, Yv, np.cross(Tv, Yv), K.CORE_OFFSET, mat=2, both_sides=True)
    obj = mb.to_object(name, col, smooth=False)
    for m in mats:
        obj.data.materials.append(m)
    obj.visible_shadow = False
    return obj


# ---------------------------------------------------------------------------

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = 'METRIC'
    col = K.get_collection("VoidDagger")

    blade, P = build_dagger_blade(col)
    paths = paint_dagger_blade(P, (512, 1024) if K.FAST else (1024, 2048))
    K.copy_uv(blade, "UVPhys")
    blade.data.materials.append(K.pbr_material("M_Dagger_Blade", K.load_image(paths["albedo"]),
                                               K.load_image(paths["emission"]),
                                               K.load_image(paths["normal"], non_color=True),
                                               metallic=0.25, roughness=0.35, emission_strength=3.0))
    parts = build_hilt(col)
    fx_mats = [K.flat_emissive_material("M_Shadow_Outer", "#5a1fa6", 1.8),
               K.flat_emissive_material("M_Shadow_Mid", "#8f3dff", 2.4),
               K.flat_emissive_material("M_Shadow_Core", "#ecd9ff", 2.8)]
    frames = [build_shadow_fire(col, fx_mats, P, i / K.FX_FRAMES, f"FX_ShadowFlame_F{i}") for i in range(K.FX_FRAMES)]
    K.shift_to_grip([blade] + frames + list(parts.values()))

    hilt_objs = list(parts.values())
    res = 1024
    K.pack_atlas(hilt_objs, res)
    hpaths = K.bake_hilt(hilt_objs, res, 8 if K.FAST else 48)
    hilt_mat = K.assign_final_hilt_material(parts, hpaths)

    ink = K.ink_material()
    outline = K.build_outline([blade, parts["guard"], parts["pommel"], parts["fuchi"], parts["ito"]], col, ink,
                              base=0.00035)
    outline.name = "Dagger_InkOutline"

    root = bpy.data.objects.new("VoidDagger", None)
    col.objects.link(root)
    grp = bpy.data.objects.new("FX_ShadowFlame", None)
    col.objects.link(grp)
    grp.parent = root
    for o in [blade, outline] + hilt_objs:
        o.parent = root
    for o in frames:
        o.parent = grp
    K.animate_fire(frames)
    K.animate_glow([blade.data.materials[0], hilt_mat])

    o = -K.GRIP_CENTRE
    if not K.NO_RENDER:
        K.render_previews(col, shots=[
            ("hero", (0.95, 0.30, 0.16 + o), (0, 0.004, 0.07 + o), 50, None, (1100, 2000)),
            ("front", (0.95, 0.0, 0.07 + o), (0, 0.004, 0.07 + o), 50, None, (1100, 2000)),
            ("guard", (0.17, 0.12, 0.05 + o), (0, 0.0, -0.005 + o), 50, None, (1400, 1400)),
            ("blade", (0.30, -0.12, 0.20 + o), (0, 0.0, 0.17 + o), 50, None, (1400, 1400)),
            ("back", (-0.9, -0.35, 0.18 + o), (0, 0.0, 0.07 + o), 50, None, (1100, 2000)),
        ])
        K.render_fire_loop_video(col, os.path.join(K.RENDER_DIR, "_loop_frames"),
                                 os.path.join(K.RENDER_DIR, "dagger_shadow_loop.mp4"),
                                 cam_loc=(0.95, 0.30, 0.16 + o), cam_target=(0, 0.004, 0.07 + o))

    os.makedirs(K.EXPORT_DIR, exist_ok=True)
    K.setup_live_viewport()
    K.tidy_preview_rig()
    path = os.path.join(HERE, "void_dagger.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True, relative_remap=True)
    for img in list(bpy.data.images):
        if img.users == 0:
            bpy.data.images.remove(img)
    for img in bpy.data.images:
        if img.source == 'FILE' and img.packed_file is None:
            img.pack()
    bpy.ops.wm.save_mainfile(compress=True)

    K.swap_ink_for_export(ink)
    stats = K.export_glb([blade, outline] + frames + hilt_objs, os.path.join(K.EXPORT_DIR, "void_dagger.glb"),
                         root_name="VoidDagger", clip_name="ShadowFlameLoop", hilt_name="Dagger_Hilt")
    print("[export] triangles:", stats, "total:", sum(stats.values()))


if __name__ == "__main__":
    main()
