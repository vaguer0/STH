"""
Void Dagger -- SeekThenHide
===========================

The purple dagger, redesigned after Kamish's Wrath: a flame-shaped blade -- a dark,
gently S-curved core whose two edges break into uneven, upswept flame teeth -- with
a glowing vein down the spine that sends a branch into every tooth.  The guard is a
scaled draconic head with a glowing almond eye, from which thick, layered flame
tongues continue the blade's teeth up along the blade and down toward the grip.  The
grip is segmented like a tail of scales, and hand-drawn shadow flames burn off the
tooth tips.

Reuses the toolkit in ../mystic_katana/build_mystic_katana.py.

    blender --background --python build_void_dagger.py
    blender --background --python build_void_dagger.py -- --fast
"""

import math
import os
import sys

import bpy  # noqa: E402  (bpy must load before bmesh when run as a module)
import bmesh
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
K.Z_FUCHI_TOP = -0.0330
K.Z_WRAP_TOP = -0.0400
K.Z_WRAP_BOT = -0.1300
K.GRIP_CENTRE = 0.5 * (K.Z_WRAP_TOP + K.Z_WRAP_BOT)


def dagger_handle_axes(z):
    t = np.clip((K.Z_WRAP_TOP - z) / (K.Z_WRAP_TOP - K.Z_WRAP_BOT), 0, 1)
    waist = np.sin(np.pi * t) ** 1.2
    return 0.0070 - 0.0006 * waist, 0.0090 - 0.0008 * waist


K.handle_axes = dagger_handle_axes

hexc, PAL = K.hexc, K.PAL
PAL.update({
    "iron_dark": hexc("#140b1c"), "iron_mid": hexc("#3a2350"),
    "silver_lo": hexc("#6c4c96"), "silver_mid": hexc("#b896e6"), "silver_hi": hexc("#efe2ff"),
    "silver_cav": hexc("#120818"),
    "jade_hi": hexc("#fbf4ff"), "jade_mid": hexc("#b26bff"), "jade_lo": hexc("#3d0e6b"),
    "ink": hexc("#0c0612"),
})
VOID = {"core": hexc("#120a1b"), "core_lift": hexc("#2a1640"), "tooth": hexc("#5b22b0"),
        "tooth_tip": hexc("#c793ff"), "edge": hexc("#efe3ff"), "vein": hexc("#9a40ff"),
        "vein_core": hexc("#e6ccff"), "ink": hexc("#07030b")}

_world_and_lights = K.setup_world_and_lights


def void_lights(col):
    _world_and_lights(col)
    rim = bpy.data.objects.get("Rim")
    if rim:
        rim.data.color = tuple(K.srgb_to_linear(hexc("#b98aff")))


K.setup_world_and_lights = void_lights


# ---------------------------------------------------------------------------
# blade outline (blade plane: y across, z along; cutting edge toward -y)
# ---------------------------------------------------------------------------
# The outline is grown procedurally: a core band along an S-curved spine, and on
# each side a run of flame teeth of uneven size and spacing that lean toward the
# tip.  Teeth on the cutting edge (-y) are bigger.  Valleys and tips are flagged
# sharp so they survive the smoothing; everything else turns into fluid curves.

BLADE_L = 0.288
BLADE_T = 0.0018      # half thickness of the flat plate
BLADE_D = 0.0040      # width of the ground edge bevel
BLADE_TEETH = []      # (side, root (y, z), tip (y, z), height), filled by blade_outline()


def spine_y(z):
    return 0.0048 * np.sin(np.pi * np.asarray(z, dtype=float) / BLADE_L * 1.15)


def core_w(z):
    return 0.0135 * np.clip(1 - (np.asarray(z, dtype=float) / BLADE_L) ** 1.8, 0, 1) ** 0.75 + 0.0006


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


def flame_teeth(rng, scale, span_range, calm=0.0):
    """calm = chance of a low, almost smooth stretch instead of a tooth."""
    z, teeth = 0.018 + rng.uniform(0, 0.006), []
    while z < BLADE_L * 0.8:
        span = rng.uniform(*span_range)
        z1 = min(z + span, BLADE_L * 0.83)
        t = 0.5 * (z + z1) / BLADE_L
        env = np.sin(np.pi * np.clip((t + 0.05) / 0.95, 0, 1)) ** 0.6
        h = scale * (0.004 + 0.019 * env * rng.uniform(0.45, 1.4))
        if rng.random() < calm:
            h = 0.0016
        teeth.append((z, z1, h, rng.uniform(0.7, 1.35)))
        z = z1
    return teeth


def side_points(side, teeth):
    """Outline points of one side, walking up the blade."""
    yc = lambda z, off: float(spine_y(z) + side * (core_w(z) + off))
    pts = [(yc(z, 0.0), float(z), 0) for z in np.linspace(0.0, teeth[0][0], 3)[1:-1]]
    for z0, z1, h, lean in teeth:
        zt = z1 + 0.55 * lean * h
        tip = (yc(z1, h), zt)
        pts += [(yc(z0, 0.0), z0, 1),
                (yc(z0 + 0.45 * (z1 - z0), 0.42 * h), z0 + 0.45 * (z1 - z0) + 0.12 * h, 0),
                (yc(z1, 0.82 * h), zt - 0.42 * lean * h, 0),
                (tip[0], tip[1], 1),
                (yc(z1, 0.36 * h), z1 + 0.2 * lean * h, 0)]
        BLADE_TEETH.append((side, (float(spine_y(0.5 * (z0 + z1))), 0.5 * (z0 + z1)), tip, h))
    z_end = teeth[-1][1]
    pts.append((yc(z_end, 0.0), z_end, 1))
    pts += [(yc(z, 0.0), float(z), 0) for z in np.linspace(z_end, BLADE_L * 0.97, 5)[1:]]
    return pts


def self_intersects(P):
    A, B = P, np.roll(P, -1, axis=0)
    n = len(P)
    for i in range(n):
        j = np.arange(i + 2, n if i > 0 else n - 1)
        if len(j) == 0:
            continue
        d, f = B[i] - A[i], B[j] - A[j]
        den = d[0] * f[:, 1] - d[1] * f[:, 0]
        ok = np.abs(den) > 1e-18
        w = A[j] - A[i]
        t = (w[:, 0] * f[:, 1] - w[:, 1] * f[:, 0]) / np.where(ok, den, 1)
        u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, den, 1)
        if np.any(ok & (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)):
            return True
    return False


def blade_outline(spacing=0.0010):
    for seed in range(40, 120):
        rng = np.random.default_rng(seed)
        BLADE_TEETH.clear()
        w0, y0 = float(core_w(0.0)), float(spine_y(0.0))
        right = side_points(1, flame_teeth(rng, 0.42, (0.024, 0.042), calm=0.4))    # spine: small, sparse
        left = side_points(-1, flame_teeth(rng, 1.45, (0.020, 0.040)))             # cutting edge: big flames
        tip = (float(spine_y(BLADE_L)) + 0.004, BLADE_L + 0.014, 1)
        raw = [(y0 - w0, 0.0, 1), (y0 + w0, 0.0, 1)] + right + [tip] + left[::-1]
        P = chaikin([(y, z) for y, z, _ in raw], [sh for _, _, sh in raw])
        L = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum()
        P = K.resample_polyline(np.vstack([P, P[:1]]), int(L / spacing) + 1)[:-1]
        if not self_intersects(P):
            break
    print(f"[dagger] outline seed {seed}, {len(BLADE_TEETH)} teeth, {len(P)} points")
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


HZ = 0.31     # texture height of the blade strip (m)


def build_dagger_blade(col):
    P = blade_outline()
    B = len(P)
    ymin, ymax = P[:, 0].min(), P[:, 0].max()
    g = 0.0022
    ys, zs = np.meshgrid(np.arange(ymin, ymax, g), np.arange(0.0, P[:, 1].max(), g))
    G = np.column_stack([ys.ravel(), zs.ravel()])
    G = G + (np.random.default_rng(3).random(G.shape) - 0.5) * g * 0.35
    G = G[inside(G, P)]
    G = G[dist_to_outline(G, P) > 0.7 * g]
    pts = np.vstack([P, G])
    vco = [Vector((float(a), float(b))) for a, b in pts]
    res = delaunay_2d_cdt(vco, [(i, (i + 1) % B) for i in range(B)], [list(range(B))], 1, 1e-9)
    verts, faces, orig_verts = res[0], res[2], res[3]
    V = np.array([(v.x, v.y) for v in verts])
    is_edge = np.array([any(o < B for o in ov) for ov in orig_verts])
    d = dist_to_outline(V, P)
    d[is_edge] = 0.0
    t = blade_thickness(d)
    mb = K.MeshBuilder()
    Wy = ymax - ymin

    def uv(i, side):
        y, z = V[i]
        u = 0.5 * (y - ymin) / Wy if side > 0 else 0.5 + 0.5 * (ymax - y) / Wy
        return (u, z / HZ)

    top = mb.add_verts([(t[i], V[i, 0], V[i, 1]) for i in range(len(V))])
    bot = mb.add_verts([(-t[i], V[i, 0], V[i, 1]) for i in range(len(V))])
    bidx = lambda i: top + i if is_edge[i] else bot + i
    for f in faces:
        f = list(f)
        mb.add_face([top + i for i in f], [uv(i, 1) for i in f], [uv(i, 1) for i in f], 0)
        fr = f[::-1]
        mb.add_face([bidx(i) for i in fr], [uv(i, -1) for i in fr], [uv(i, -1) for i in fr], 0)
    obj = mb.to_object("Dagger_Blade", col)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if bm.calc_volume(signed=True) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(obj.data)
    bm.free()
    return obj, P


# ---------------------------------------------------------------------------
# blade texture: dark core, flame teeth, glowing vein
# ---------------------------------------------------------------------------

def paint_dagger_blade(P, res=(1024, 2048)):
    W, H = res
    ymin, ymax = P[:, 0].min(), P[:, 0].max()
    Wy = ymax - ymin
    half = W // 2
    yy = ymin + (np.arange(half) + 0.5) / half * Wy
    zz = (np.arange(H) + 0.5) / H * HZ
    Y, Z = np.meshgrid(yy, zz)
    Q = np.column_stack([Y.ravel(), Z.ravel()])
    print("[dagger] distance field ...")
    D = dist_to_outline(Q, P).reshape(H, half)
    px = Wy / half
    aa = px * 1.2
    nz, nz2 = K.Noise2(71), K.Noise2(72)
    vein = np.zeros((H, half), np.float32)

    def stamp(pts, radii, layer):
        for (p0, p1, r0, r1) in zip(pts[:-1], pts[1:], radii[:-1], radii[1:]):
            rm = max(r0, r1) + 2 * aa
            c0 = int(max(0, (min(p0[0], p1[0]) - rm - ymin) / px - 1))
            c1 = int(min(half, (max(p0[0], p1[0]) + rm - ymin) / px + 2))
            r0_ = int(max(0, (min(p0[1], p1[1]) - rm) / HZ * H - 1))
            r1_ = int(min(H, (max(p0[1], p1[1]) + rm) / HZ * H + 2))
            if c1 <= c0 or r1_ <= r0_:
                continue
            y_, z_ = Y[r0_:r1_, c0:c1], Z[r0_:r1_, c0:c1]
            d = p1 - p0
            l2 = float(d @ d) or 1e-18
            tt = np.clip(((y_ - p0[0]) * d[0] + (z_ - p0[1]) * d[1]) / l2, 0, 1)
            dist = np.hypot(y_ - (p0[0] + tt * d[0]), z_ - (p0[1] + tt * d[1]))
            cov = np.clip((r0 + (r1 - r0) * tt - dist) / aa + 0.5, 0, 1)
            layer[r0_:r1_, c0:c1] = np.maximum(layer[r0_:r1_, c0:c1], cov)

    rng = np.random.default_rng(11)
    # main vein: a jittery line down the spine, thick at the guard, thinning out
    zs = np.arange(0.006, BLADE_L * 0.96, 0.0045)
    spine = np.column_stack([spine_y(zs) + rng.uniform(-1, 1, len(zs)) * 0.0011 * (1 - zs / BLADE_L), zs])
    stamp(spine, np.linspace(0.00085, 0.00018, len(zs)), vein)
    # a branch into every tooth, stopping short of its tip
    for side, root, tip, h in BLADE_TEETH:
        if h < 0.006 or rng.random() < 0.4:             # only some teeth get a branch
            continue
        a, b = np.array(root), np.array(tip)
        n = int(rng.integers(3, 6))
        pts = [a]
        for k in range(1, n + 1):
            q = a + (b - a) * (0.72 * k / n)
            nrm = np.array([-(b - a)[1], (b - a)[0]]) / (np.linalg.norm(b - a) + 1e-12)
            pts.append(q + nrm * rng.uniform(-1, 1) * 0.0012)
        pts = np.array(pts)
        stamp(pts, np.linspace(0.0005, 0.00012, len(pts)), vein)
        if rng.random() < 0.5:                           # little fork
            k = int(rng.integers(1, len(pts) - 1))
            fork = np.array([pts[k], pts[k] + (b - a) * 0.18 + rng.uniform(-1, 1, 2) * 0.002])
            stamp(fork, np.array([0.0003, 0.0001]), vein)

    def mixc(base, col, t):
        return base * (1 - t[..., None]) + col[None, None, :] * t[..., None]

    off = np.abs(Y - spine_y(Z)) - core_w(Z)        # <0 inside the core band, >0 out in the teeth
    in_core = K.smoothstep(0.0006, -0.0006, off)
    reach = K.smoothstep(0.0, 0.022, off)
    mott = nz.fbm(Y / 0.012, Z / 0.03, octaves=4)
    streak = nz2.fbm(Y / 0.0012 + 2 * nz(Z / 0.05, 1.0), Z / 0.06, octaves=3)
    alb = mixc(np.broadcast_to(VOID["tooth"], (H, half, 3)).copy(), VOID["tooth_tip"], reach ** 0.9)
    alb = mixc(alb, VOID["core"], in_core)
    alb = mixc(alb, VOID["core_lift"], in_core * K.smoothstep(0.55, 0.8, mott) * 0.7)
    alb = alb * (0.9 + 0.18 * streak[..., None])
    alb = mixc(alb, VOID["ink"], np.exp(-(off / 0.00045) ** 2) * 0.95)            # ink line round the core
    bevel = 1 - K.smoothstep(0.0, BLADE_D * 0.85, D)
    alb = mixc(alb, VOID["tooth_tip"] * 1.1, bevel ** 1.6 * 0.8)
    alb = mixc(alb, VOID["edge"], K.smoothstep(0.0011, 0.0, D))
    alb = mixc(alb, VOID["ink"], np.exp(-((D - BLADE_D * 0.88) / 0.0004) ** 2) * 0.8)
    halo = K.gauss_blur(vein, 7)
    carve = np.clip(K.gauss_blur(vein, 2.0) * 2.2 - vein * 2, 0, 1)
    alb = mixc(alb, VOID["ink"], carve * 0.8)
    alb = mixc(alb, VOID["vein"], np.clip(halo * 0.6, 0, 0.6) * (1 - carve))
    alb = mixc(alb, VOID["vein_core"], vein)
    alb = np.clip(alb, 0, 1)
    emi = (VOID["vein"][None, None] * np.clip(halo * 0.9, 0, 1)[..., None]
           + VOID["vein_core"][None, None] * vein[..., None]
           + VOID["vein"][None, None] * (K.smoothstep(0.006, 0.026, off) * 0.45)[..., None]
           + VOID["vein"][None, None] * (K.smoothstep(0.0009, 0.0, D) * 0.45)[..., None])
    emi = np.clip(emi, 0, 1)
    h = K.gauss_blur((-0.00025 * vein).astype(np.float32), 1.2)
    gx = np.gradient(h, axis=1) / px
    gy = np.gradient(h, axis=0) / (HZ / H)
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
# hilt: scaled draconic head, flame-tongue guard, eye gem, scaled grip, tail pommel
# ---------------------------------------------------------------------------

AX_X, AX_Y, AX_Z = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])

# head seen in the blade plane (y across, z along); (y, z, sharp)
HEAD = [(0.0, 0.0045, 1), (0.0095, 0.0060, 0), (0.0185, 0.0020, 1), (0.0160, -0.0050, 0),
        (0.0205, -0.0120, 1), (0.0130, -0.0190, 0), (0.0080, -0.0285, 0), (0.0, -0.0330, 1),
        (-0.0080, -0.0285, 0), (-0.0130, -0.0190, 0), (-0.0205, -0.0120, 1), (-0.0160, -0.0050, 0),
        (-0.0185, 0.0020, 1), (-0.0095, 0.0060, 0)]
HEAD_HALF = 0.0095

# flame tongues: (root y, root z, up?, height, base width, lean, curl, x offset, half thickness, seed)
TONGUES = [
    (0.0120, -0.0030, True, 0.058, 0.017, 0.45, 0.75, 0.0000, 0.0040, 11),
    (0.0160, -0.0080, True, 0.040, 0.013, 0.85, -0.6, 0.0040, 0.0032, 12),
    (0.0180, -0.0130, True, 0.027, 0.010, 1.15, 0.8, -0.0040, 0.0028, 13),
    (0.0120, -0.0230, False, 0.030, 0.011, 0.70, -0.7, 0.0020, 0.0030, 14),
    (0.0080, -0.0290, False, 0.019, 0.009, 0.95, 0.55, -0.0025, 0.0026, 15),
]


def plaque(mb, outline, half, origin=(0.0, 0.0, 0.0), mat=0, bevel=0.38, rings=3):
    """Extrude a 2D outline drawn in the blade plane into a rounded plate."""
    sub = K.MeshBuilder()
    K.extrude_shape(sub, outline, 2 * half, bevel=bevel, bevel_rings=rings)
    sub.face_mat = [mat] * len(sub.faces)
    K.place_local(sub, np.asarray(origin, float), AX_Y, AX_Z, AX_X)   # local x->Y, y->Z, thickness->X
    mb.merge(sub)


def tongue_outline(h, b, lean, curl, seed, side, up):
    outer, _ = K.flame_outlines(h, b, lean, curl, 0.12, seed)
    o = np.array(outer)
    o[:, 0] *= side
    if not up:
        o[:, 1] *= -1
    return o


def build_guard(col):
    mb = K.MeshBuilder()
    head = chaikin([(y, z) for y, z, _ in HEAD], [s for _, _, s in HEAD], iters=3)
    plaque(mb, head, HEAD_HALF, mat=0, bevel=0.45)
    for side, sc in ((1.0, 1.0), (-1.0, 1.15)):      # the cutting-edge side flames are bigger
        for (ry, rz, up, h, b, lean, curl, xo, half, seed) in TONGUES:
            o = tongue_outline(h * sc, b * sc, lean, curl, seed + (0 if side > 0 else 50), side, up)
            plaque(mb, o, half, origin=(xo, side * ry, rz), mat=1, bevel=0.35, rings=2)
    return mb.to_object("Hilt_Guard", col)


def build_gem(col):
    """Almond eye on both faces of the head."""
    mb = K.MeshBuilder()
    t = np.linspace(-1, 1, 14)
    top = np.column_stack([0.0085 * t, 0.0042 * (1 - t ** 2)])
    eye = np.vstack([top, (top * [1, -1])[::-1][1:-1]])
    rot = math.radians(-12)
    eye = eye @ np.array([[math.cos(rot), math.sin(rot)], [-math.sin(rot), math.cos(rot)]])
    for sx in (1, -1):
        sub = K.MeshBuilder()
        K.extrude_shape(sub, eye, 0.0036, bevel=0.55, bevel_rings=2)
        K.place_local(sub, np.array([sx * (HEAD_HALF - 0.0004), 0.0, -0.0115]), AX_Y * sx, AX_Z, AX_X * sx)
        mb.merge(sub)
    return mb.to_object("Hilt_Gem", col)


def build_scaled_grip(col):
    """Grip of overlapping scale rings, each ring flaring then tucking under the next."""
    mb = K.MeshBuilder()
    n = 9
    z_top, z_bot = K.Z_FUCHI_TOP - 0.002, K.Z_WRAP_BOT
    seg = (z_top - z_bot) / n
    rings = []
    for k in range(n):
        zt = z_top - k * seg
        ph = 0.5 * k
        rings += [(zt - 0.0002, 0.93, ph), (zt - 0.25 * seg, 1.0, ph), (zt - 0.8 * seg, 1.1, ph),
                  (zt - seg + 0.0004, 1.12, ph)]

    def outline(z, sc, ph):
        hx, hy = dagger_handle_axes(z)
        th = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        c, s = np.cos(th), np.sin(th)
        e = 2.3
        x = hx * np.sign(c) * np.abs(c) ** (2 / e)
        y = hy * np.sign(s) * np.abs(s) ** (2 / e)
        lobe = 1 + 0.05 * np.abs(np.cos(3 * th + ph)) ** 0.5 * (sc - 0.9) / 0.2
        return np.column_stack([x, y]) * (sc * lobe)[:, None]

    K.loft_z(mb, [(r[0], r[0], r[1], r[2]) for r in rings], outline, cap_top='flat', cap_bottom='flat',
             sharp_rings=tuple(range(3, len(rings) - 1, 4)))
    return mb.to_object("Hilt_Grip", col)


def build_pommel(col):
    mb = K.MeshBuilder()
    zb = K.Z_WRAP_BOT
    hx, hy = dagger_handle_axes(zb)

    def outline(sc, e):
        return K.superellipse(36, hx * sc, hy * sc, e)

    rings = [(zb + 0.001, 1.10, 2.3), (zb - 0.001, 1.24, 2.4), (zb - 0.005, 1.26, 2.5), (zb - 0.008, 1.05, 2.4),
             (zb - 0.010, 1.12, 2.4), (zb - 0.016, 0.86, 2.2), (zb - 0.026, 0.58, 2.1), (zb - 0.036, 0.32, 2.0),
             (zb - 0.042, 0.12, 2.0)]
    K.loft_z(mb, rings, outline, cap_top=None, cap_bottom=0.007, sharp_rings=(3, 4))
    for side in (1.0, -1.0):          # small flame tongues licking down off the tail
        o = tongue_outline(0.022, 0.012, 0.8, 0.6 * side, 21 if side > 0 else 22, side, False)
        plaque(mb, o, 0.0028, origin=(0.0, side * 0.007, zb - 0.004), bevel=0.35, rings=2)
    return mb.to_object("Hilt_Pommel", col)


def scale_painter(name):
    """Dark dragon scales: each Voronoi cell lit in the middle, inked at the rim."""
    k = K.NodeKit(bpy.data.materials.new(name))
    ao, edge, hl = k.painter_basics(ao_dist=0.004)
    cell = k.voronoi(1 / 0.0028)
    rim = k.sstep(0.22, 0.5, cell)
    nz = k.noise(140.0, 3.0)
    v = k.add(k.add(k.mul(hl, 0.5), k.mul(ao, 0.5)), k.add(k.mul(rim, -0.28), k.mul(k.sub(nz, 0.5), 0.08)))
    col = k.ramp(v, [(0.0, "#07030b"), (0.35, "#1f0f30"), (0.7, "#4b2a73"), (1.0, "#a27ad6")])
    col = k.mix(col, K.lin("#d9c2ff"), k.mul(edge, 0.55))
    k.finish(col, None, (1.0, 0.42, 0.35), k.mul(k.sub(1.0, rim), 0.00006))
    K.PAINTER_SOCKETS[k.mat.name] = k.sockets
    return k.mat


def build_hilt(col):
    parts = {
        "guard": build_guard(col),
        "gem": build_gem(col),
        "grip": build_scaled_grip(col),
        "fuchi": K.build_fuchi(col),
        "pommel": build_pommel(col),
    }
    scales = scale_painter("P_DragonScale")
    flames = K.painter("enamel", "P_GuardFlame", "#14061f", "#3f1670", "#9a5ce0", "#b98aff", "#8a30ff", 0.12)
    mats = {"guard": [scales, flames], "pommel": [flames], "grip": scales,
            "fuchi": K.painter("iron", "P_Fuchi", wave_v=0.0036), "gem": K.painter("jade", "P_VoidGem")}
    for key, obj in parts.items():
        for m in (mats[key] if isinstance(mats[key], list) else [mats[key]]):
            obj.data.materials.append(m)
    return parts


# ---------------------------------------------------------------------------
# hand-drawn shadow flames burning off the tooth tips
# ---------------------------------------------------------------------------

def build_shadow_fire(col, mats, phase, name):
    mb = K.MeshBuilder()
    rng = np.random.default_rng(909)

    def flame(origin, up, size, width, lean, curl, roll, seed):
        up = up / np.linalg.norm(up)
        lat = np.cross(AX_X, up)
        lat = lat * math.cos(roll) + AX_X * math.sin(roll)
        nrm = np.cross(lat, up)
        outer, core = K.flame_outlines(size, width, lean, curl, 0.0, seed)
        mid = np.array(outer) * 0.66
        K.flat_shape(mb, outer, origin, lat, up, nrm, 0.0, mat=0)
        K.flat_shape(mb, mid, origin + up * size * 0.04, lat, up, nrm, 0.00018, mat=1, both_sides=True)
        K.flat_shape(mb, core, origin, lat, up, nrm, K.CORE_OFFSET, mat=2, both_sides=True)

    for j, (side, root, tip, h) in enumerate(BLADE_TEETH):
        if rng.random() < 0.25:            # leave some teeth bare: fire is never even
            continue
        r, t = np.array([0.0, *root]), np.array([0.0, *tip])
        d = (t - r) / np.linalg.norm(t - r)
        up = d + AX_Z * 0.9                  # fire bends upward off the tooth
        size = (0.006 + 0.85 * h) * (1 + 0.22 * K.wave(phase, rng.random(), 1 + j % 2)) * rng.uniform(0.75, 1.2)
        flame(t - d * 0.0015, up, size, size * 0.42, rng.uniform(0.4, 0.9) * side,
              (0.6 + 0.3 * K.wave(phase, rng.random())) * -side, rng.uniform(-0.5, 0.5), 1500 + j)
    tip = np.array([0.0, float(spine_y(BLADE_L)) + 0.004, BLADE_L + 0.012])
    for k in range(1):
        size = 0.015 * (1 + 0.2 * K.wave(phase, 0.3 * k + 0.1))
        flame(tip, np.array([0.0, 0.25, 1.0]), size, size * 0.4, 0.6, 0.8 * (1 if k else -1),
              (0.6 if k else -0.4), 1600 + k)
    # embers rising off the blade and burning out
    for j in range(14):
        rise = (phase + j * 0.618) % 1.0
        side, root, tip_, h = BLADE_TEETH[int(rng.integers(0, len(BLADE_TEETH)))]
        p = np.array([rng.uniform(-0.01, 0.01), tip_[0], tip_[1]]) + AX_Z * (0.012 + 0.05 * rise) \
            + AX_Y * side * rng.uniform(0.0, 0.012)
        fh = rng.uniform(0.004, 0.0075) * (1 - 0.6 * rise)
        o2, c2 = K.flame_outlines(fh, fh * 0.5, rng.uniform(0.5, 0.9), float(rng.choice([-1, 1])), 0.0, 1700 + j)
        roll = rng.uniform(-0.6, 0.6)
        lat = AX_Y * math.cos(roll) + AX_X * math.sin(roll)
        K.flat_shape(mb, o2, p, lat, AX_Z, np.cross(lat, AX_Z), 0.0, mat=1)
        K.flat_shape(mb, c2, p, lat, AX_Z, np.cross(lat, AX_Z), K.CORE_OFFSET, mat=2, both_sides=True)
    for j in range(10):
        side, root, tip_, h = BLADE_TEETH[int(rng.integers(0, len(BLADE_TEETH)))]
        p = np.array([rng.uniform(-0.012, 0.012), tip_[0] + side * rng.uniform(0.004, 0.02),
                      tip_[1] + rng.uniform(0.0, 0.03) + 0.03 * ((phase + j * 0.41) % 1.0)])
        R = rng.uniform(0.0008, 0.002) * (0.5 + 0.5 * abs(K.wave(phase, j * 0.29, 2)))
        Zs = np.array([1.0, rng.uniform(-0.3, 0.3), 0.0])
        Zs /= np.linalg.norm(Zs)
        Xs = np.cross(AX_Z, Zs)
        Xs /= np.linalg.norm(Xs)
        K.flat_shape(mb, K.spark_outline(R), p, Xs, np.cross(Zs, Xs), Zs, 0.0, mat=1)
        K.flat_shape(mb, K.spark_outline(R * 0.42, 1.3, 4), p, Xs, np.cross(Zs, Xs), Zs, K.CORE_OFFSET, mat=2,
                     both_sides=True)
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
                                               metallic=0.08, roughness=0.42, emission_strength=3.0))
    parts = build_hilt(col)
    fx_mats = [K.flat_emissive_material("M_Shadow_Outer", "#5a1fa6", 1.8),
               K.flat_emissive_material("M_Shadow_Mid", "#8f3dff", 2.4),
               K.flat_emissive_material("M_Shadow_Core", "#ecd9ff", 2.8)]
    frames = [build_shadow_fire(col, fx_mats, i / K.FX_FRAMES, f"FX_ShadowFlame_F{i}") for i in range(K.FX_FRAMES)]
    K.shift_to_grip([blade] + frames + list(parts.values()))

    hilt_objs = list(parts.values())
    K.pack_atlas(hilt_objs, 1024)
    hpaths = K.bake_hilt(hilt_objs, 1024, 8 if K.FAST else 48)
    hilt_mat = K.assign_final_hilt_material(parts, hpaths)

    ink = K.ink_material()
    outline = K.build_outline([blade, parts["guard"], parts["pommel"], parts["fuchi"], parts["grip"]], col, ink,
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
            ("hero", (1.0, 0.32, 0.16 + o), (0, 0.0, 0.07 + o), 50, None, (1100, 2000)),
            ("front", (1.0, 0.0, 0.07 + o), (0, 0.0, 0.07 + o), 50, None, (1100, 2000)),
            ("guard", (0.19, 0.13, 0.04 + o), (0, 0.0, -0.01 + o), 50, None, (1400, 1400)),
            ("blade", (0.32, -0.13, 0.20 + o), (0, 0.0, 0.17 + o), 50, None, (1400, 1400)),
            ("back", (-0.95, -0.35, 0.18 + o), (0, 0.0, 0.07 + o), 50, None, (1100, 2000)),
        ])
        K.render_fire_loop_video(col, os.path.join(K.RENDER_DIR, "_loop_frames"),
                                 os.path.join(K.RENDER_DIR, "dagger_shadow_loop.mp4"),
                                 cam_loc=(1.0, 0.32, 0.16 + o), cam_target=(0, 0.0, 0.07 + o))
        for f in os.listdir(K.RENDER_DIR):          # the shared helper names stills katana_*
            if f.startswith("katana_"):
                os.replace(os.path.join(K.RENDER_DIR, f), os.path.join(K.RENDER_DIR, "dagger_" + f[7:]))

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
