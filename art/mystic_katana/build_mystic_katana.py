"""
Mystic Katana -- SeekThenHide
=============================

Builds the mystic katana entirely from code inside Blender: geometry, hand-painted
textures, hand-drawn style FX, ink outline, texture bakes, .blend, .glb and preview
renders.

Run with Blender 4.2+ (tested on 5.2):

    blender --background --python build_mystic_katana.py
    blender --background --python build_mystic_katana.py -- --no-render
    blender --background --python build_mystic_katana.py -- --fast

Every visible shape is authored from hand-designed profiles and splines (no stock
cylinders / spheres / cones), and the effects are flat two-tone shapes in the style
of hand-drawn 2D VFX instead of glowing primitives.

Axes: blade points +Z, flats face +/-X, cutting edge faces -Y.  Origin is at the
centre of the grip so the item can be parented straight to a hand bone.
Units are metres.
"""

import math
import os
import sys

import bpy  # noqa: E402  (bpy must load before bmesh when run as a module)
import bmesh
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
TEX_DIR = os.path.join(HERE, "textures")
EXPORT_DIR = os.path.join(HERE, "export")
RENDER_DIR = os.path.join(HERE, "renders")

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
FAST = "--fast" in ARGS
NO_RENDER = "--no-render" in ARGS
NO_BAKE = "--no-bake" in ARGS


# ---------------------------------------------------------------------------
# palette (sRGB hex, painted like a 2D artist would pick them)
# ---------------------------------------------------------------------------

def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


PAL = {
    "steel_deep": hexc("#1a1526"),
    "steel_mid": hexc("#2d2542"),
    "steel_lift": hexc("#4b4068"),
    "steel_edge": hexc("#8e86a8"),
    "ha_pale": hexc("#a59cbd"),
    "ha_warm": hexc("#d7a3c4"),
    "glow_hot": hexc("#ff3d8b"),
    "glow_core": hexc("#ffe0ec"),
    "glow_deep": hexc("#c4185f"),
    "ink": hexc("#1a0b1e"),
    "silver_hi": hexc("#efe9f2"),
    "silver_mid": hexc("#b9b0c6"),
    "silver_lo": hexc("#5d5470"),
    "silver_cav": hexc("#241d30"),
    "ito_hi": hexc("#ff8db6"),
    "ito_mid": hexc("#ee4a86"),
    "ito_lo": hexc("#86184b"),
    "same_dark": hexc("#17131d"),
    "same_nod": hexc("#3a3346"),
    "iron_dark": hexc("#1c1824"),
    "iron_mid": hexc("#3b3449"),
    "jade_hi": hexc("#ffc2db"),
    "jade_mid": hexc("#f0538f"),
    "jade_lo": hexc("#6e0f3a"),
}


# ---------------------------------------------------------------------------
# small math helpers
# ---------------------------------------------------------------------------

def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def catmull_rom(points, samples_per_seg=12, closed=False):
    """Centripetal-ish Catmull-Rom through hand placed control points."""
    pts = [np.array(p, dtype=np.float64) for p in points]
    if closed:
        ext = [pts[-1]] + pts + [pts[0], pts[1]]
    else:
        ext = [pts[0] * 2 - pts[1]] + pts + [pts[-1] * 2 - pts[-2]]
    out = []
    nseg = len(pts) if closed else len(pts) - 1
    for i in range(nseg):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for j in range(samples_per_seg):
            t = j / samples_per_seg
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(pts[-1])
    return np.array(out)


def resample_polyline(pts, n):
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, cum[-1], n)
    out = np.empty((n, pts.shape[1]))
    for k in range(pts.shape[1]):
        out[:, k] = np.interp(target, cum, pts[:, k])
    return out


def polyline_tangents(pts):
    t = np.gradient(pts, axis=0)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def rotation_minimizing_frames(pts, n0):
    """Parallel-transport normals along a 3D polyline (double reflection method)."""
    T = polyline_tangents(pts)
    N = np.zeros_like(pts)
    n = np.asarray(n0, dtype=np.float64)
    n = n - T[0] * np.dot(n, T[0])
    N[0] = n / np.linalg.norm(n)
    for i in range(len(pts) - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = np.dot(v1, v1)
        if c1 < 1e-20:
            N[i + 1] = N[i]
            continue
        rL = N[i] - (2.0 / c1) * np.dot(v1, N[i]) * v1
        tL = T[i] - (2.0 / c1) * np.dot(v1, T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = np.dot(v2, v2)
        nn = rL - (2.0 / c2) * np.dot(v2, rL) * v2 if c2 > 1e-20 else rL
        N[i + 1] = nn / np.linalg.norm(nn)
    B = np.cross(T, N)
    return T, N, B


class Noise2:
    """Value noise on an arbitrary coordinate pair, vectorised with numpy."""

    def __init__(self, seed, size=512):
        rng = np.random.default_rng(seed)
        self.size = size
        self.grid = rng.random((size, size)).astype(np.float32)

    def __call__(self, x, y):
        s = self.size
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xi = np.floor(x).astype(np.int64)
        yi = np.floor(y).astype(np.int64)
        fx = x - xi
        fy = y - yi
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)
        x0, x1 = xi % s, (xi + 1) % s
        y0, y1 = yi % s, (yi + 1) % s
        g = self.grid
        a = g[y0, x0] * (1 - fx) + g[y0, x1] * fx
        b = g[y1, x0] * (1 - fx) + g[y1, x1] * fx
        return a * (1 - fy) + b * fy

    def fbm(self, x, y, octaves=4, lac=2.03, gain=0.5):
        amp, tot, norm = 1.0, 0.0, 0.0
        for o in range(octaves):
            tot = tot + amp * self(x * lac ** o + 17.3 * o, y * lac ** o - 9.1 * o)
            norm += amp
            amp *= gain
        return tot / norm


# ---------------------------------------------------------------------------
# mesh helpers
# ---------------------------------------------------------------------------

def get_collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(col)
    return col


class MeshBuilder:
    """Accumulates verts / faces / per-corner UV sets, then emits one object."""

    def __init__(self):
        self.verts = []
        self.faces = []
        self.uv_phys = []      # per face: list of (u, v) in metres -> atlas packing
        self.uv_param = []     # per face: list of (u, v) normalised params -> procedural shading
        self.face_island = []
        self.face_mat = []
        self.sharp = set()
        self.cur_mat = 0
        self.vert_attrs = {}
        self.island_count = 0

    def new_island(self):
        self.island_count += 1
        return self.island_count - 1

    def add_verts(self, pts):
        base = len(self.verts)
        self.verts.extend([tuple(map(float, p)) for p in pts])
        return base

    def add_face(self, idx, uvp=None, uvq=None, island=0):
        self.faces.append(tuple(idx))
        self.uv_phys.append(uvp or [(0.0, 0.0)] * len(idx))
        self.uv_param.append(uvq or [(0.0, 0.0)] * len(idx))
        self.face_island.append(island)
        self.face_mat.append(self.cur_mat)

    def merge(self, other):
        base = len(self.verts)
        ib = self.island_count
        self.verts.extend(other.verts)
        for f, a, b, isl, mi in zip(other.faces, other.uv_phys, other.uv_param, other.face_island, other.face_mat):
            self.cur_mat = mi
            self.add_face([i + base for i in f], a, b, isl + ib)
        self.cur_mat = 0
        self.sharp |= {(i + base, j + base) for i, j in other.sharp}
        self.island_count += other.island_count

    def to_object(self, name, collection, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.verts, [], self.faces)
        me.update()
        for layer_name, data in (("UVPhys", self.uv_phys), ("UVParam", self.uv_param)):
            uv = me.uv_layers.new(name=layer_name)
            flat = [c for face in data for uvc in face for c in uvc]
            uv.uv.foreach_set("vector", flat)
        isl = me.attributes.new("island", 'INT', 'FACE')
        isl.data.foreach_set("value", self.face_island)
        if any(self.face_mat):
            me.polygons.foreach_set("material_index", self.face_mat)
        for an, vals in self.vert_attrs.items():
            at = me.attributes.new(an, 'FLOAT', 'POINT')
            at.data.foreach_set("value", list(vals) + [0.0] * (len(self.verts) - len(vals)))
        if smooth:
            me.shade_smooth()
        if self.sharp:
            attr = me.attributes.get("sharp_edge") or me.attributes.new("sharp_edge", 'BOOLEAN', 'EDGE')
            vals = [False] * len(me.edges)
            for e in me.edges:
                a, b = e.vertices
                if (a, b) in self.sharp or (b, a) in self.sharp:
                    vals[e.index] = True
            attr.data.foreach_set("value", vals)
        obj = bpy.data.objects.new(name, me)
        collection.objects.link(obj)
        return obj


def sweep(mb, pts, frames, profiles, closed_profile=True, cap_start=None, cap_end=None,
          sharp_cols=(), island=None, u_scale=None):
    """Sweep per-ring 2D profiles (arrays of (n, b) offsets) along pts.

    frames = (N, B) arrays; profiles = list of (M, 2) arrays, one per ring.
    cap_* = None | 'point' (collapse to tip vertex) | 'flat' (fan cap).
    """
    N, B = frames
    rings = len(pts)
    M = len(profiles[0])
    island = mb.new_island() if island is None else island
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    vlen = np.concatenate([[0.0], np.cumsum(seg)])
    total = vlen[-1] if vlen[-1] > 0 else 1.0

    ring_idx = []
    perims = []
    us = []
    for r in range(rings):
        prof = profiles[r]
        world = pts[r] + np.outer(prof[:, 0], N[r]) + np.outer(prof[:, 1], B[r])
        ring_idx.append(mb.add_verts(world))
        loop = np.vstack([prof, prof[:1]]) if closed_profile else prof
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
        perims.append(d[-1])
        us.append(d / max(d[-1], 1e-12))
    ref_perim = u_scale if u_scale is not None else float(np.max(perims))

    ncols = M if closed_profile else M - 1
    for r in range(rings - 1):
        a0, a1 = ring_idx[r], ring_idx[r + 1]
        for j in range(ncols):
            j1 = (j + 1) % M
            ua, ub = us[r][j], us[r][j + 1]
            ua2, ub2 = us[r + 1][j], us[r + 1][j + 1]
            uvq = [(ua, vlen[r] / total), (ub, vlen[r] / total), (ub2, vlen[r + 1] / total), (ua2, vlen[r + 1] / total)]
            uvp = [(ua * ref_perim, vlen[r]), (ub * ref_perim, vlen[r]), (ub2 * ref_perim, vlen[r + 1]), (ua2 * ref_perim, vlen[r + 1])]
            mb.add_face([a0 + j, a0 + j1, a1 + j1, a1 + j], uvp, uvq, island)
        for j in sharp_cols:
            mb.sharp.add((a0 + j, a1 + j))

    T = polyline_tangents(pts)

    def cap(ring_i, mode, flip):
        """mode: None, 'flat', or a float = length of a pointed tip along the tangent."""
        base = ring_idx[ring_i]
        if mode is None:
            return
        cap_island = mb.new_island()
        prof = profiles[ring_i]
        span = max(np.ptp(prof[:, 0]), np.ptp(prof[:, 1]), 1e-9)
        ctr = pts[ring_i] + prof.mean(axis=0)[0] * N[ring_i] + prof.mean(axis=0)[1] * B[ring_i]
        if mode != 'flat':
            ctr = ctr + T[ring_i] * float(mode) * (-1.0 if flip else 1.0)
        tip = mb.add_verts([ctr])
        for j in range(M):
            j1 = (j + 1) % M
            if not closed_profile and j1 == 0:
                continue
            q = lambda k: (0.5 + prof[k, 0] / span, 0.5 + prof[k, 1] / span)
            p = lambda k: (prof[k, 0] + span, prof[k, 1] + span)
            idx = [base + j, base + j1, tip]
            uvq = [q(j), q(j1), (0.5, 0.5)]
            uvp = [p(j), p(j1), (span, span)]
            if flip:
                idx.reverse()
                uvq.reverse()
                uvp.reverse()
            mb.add_face(idx, uvp, uvq, cap_island)

    cap(0, cap_start, flip=True)
    cap(rings - 1, cap_end, flip=False)
    return ring_idx


def extrude_shape(mb, outline, depth, bevel=0.35, bevel_rings=3, island=None):
    """Extrude a hand-drawn 2D outline (N,2, CCW) into a pillowy plaque.

    The rim is rounded with a quarter-ellipse bevel so the result reads like a cast
    or carved piece rather than a hard boolean slab.  Returns per-vertex positions in
    the shape's local (x, y, z) frame, z = thickness axis.
    """
    outline = np.asarray(outline, dtype=np.float64)
    area = np.sum(outline[:, 0] * np.roll(outline[:, 1], -1) - np.roll(outline[:, 0], -1) * outline[:, 1])
    if area < 0:
        outline = outline[::-1]
    n = len(outline)
    # outward normals of the outline
    tang = np.roll(outline, -1, axis=0) - np.roll(outline, 1, axis=0)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    inset = depth * bevel if bevel_rings > 0 else 0.0
    if inset > 0:
        # never inset more than ~40% of the local width, so thin tails and
        # cloud tips do not fold over themselves
        inset = np.minimum(inset, 0.4 * local_width(outline))
        for _ in range(4):
            inset = 0.5 * inset + 0.25 * (np.roll(inset, 1) + np.roll(inset, -1))
    # ring list from back face rim -> side -> front face rim
    prof = []
    if bevel_rings == 0:
        prof = [(0.0, -0.5 * depth), (0.0, 0.5 * depth)]
    for k in range(bevel_rings, -1, -1) if bevel_rings else ():
        a = (k / bevel_rings) * (math.pi / 2)
        prof.append((-inset * (1 - math.cos(a)), -(depth * 0.5 - inset) - inset * math.sin(a)))
    for k in range(0, bevel_rings + 1) if bevel_rings else ():
        a = (k / bevel_rings) * (math.pi / 2)
        prof.append((-inset * (1 - math.cos(a)), (depth * 0.5 - inset) + inset * math.sin(a)))
    prof = [(np.broadcast_to(off, (n,)), np.broadcast_to(z, (n,))) for off, z in prof]
    island_side = mb.new_island() if island is None else island
    perim = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.vstack([outline, outline[:1]]), axis=0), axis=1))])
    ring_base = []
    for (off, z) in prof:
        pts = np.column_stack([outline + nrm * off[:, None], z])
        ring_base.append(mb.add_verts(pts))
    zs = [float(np.mean(z)) for _, z in prof]
    # long rims are cut into ~10 cm UV strips so they pack into the atlas
    chunk = np.minimum((perim[:-1] / 0.1).astype(int), int(perim[-1] / 0.1))
    chunk_isl = {c: (island_side if c == 0 else mb.new_island()) for c in np.unique(chunk)}
    for r in range(len(prof) - 1):
        a0, a1 = ring_base[r], ring_base[r + 1]
        for j in range(n):
            j1 = (j + 1) % n
            c = int(chunk[j])
            u0, u1 = perim[j] - c * 0.1, perim[j + 1] - c * 0.1
            uvp = [(u0, zs[r] + depth), (u1, zs[r] + depth), (u1, zs[r + 1] + depth), (u0, zs[r + 1] + depth)]
            uvq = [(u0 / perim[-1], r / (len(prof) - 1)), (u1 / perim[-1], r / (len(prof) - 1)),
                   (u1 / perim[-1], (r + 1) / (len(prof) - 1)), (u0 / perim[-1], (r + 1) / (len(prof) - 1))]
            mb.add_face([a0 + j, a0 + j1, a1 + j1, a1 + j], uvp, uvq, chunk_isl[c])
    # caps: triangulate inset outline with ear clipping via bmesh
    inner = outline + nrm * (-np.broadcast_to(inset, (n,)))[:, None]
    tris = triangulate_polygon(inner)
    mn = inner.min(axis=0)
    ext = max(np.ptp(inner[:, 0]), np.ptp(inner[:, 1]))
    for face_ring, flip in ((ring_base[0], True), (ring_base[-1], False)):
        isl = mb.new_island()
        for t in tris:
            idx = [face_ring + i for i in t]
            uvp = [tuple(inner[i] - mn) for i in t]
            uvq = [tuple((inner[i] - mn) / ext) for i in t]
            if flip:
                idx.reverse()
                uvp.reverse()
                uvq.reverse()
            mb.add_face(idx, uvp, uvq, isl)


def local_width(outline):
    """Approximate width of a closed outline at each vertex (distance across the shape)."""
    P = np.asarray(outline)
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    L = seg.sum()
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    arc = np.abs(cum[:, None] - cum[None, :])
    arc = np.minimum(arc, L - arc)
    across = arc > 2.2 * D + 1e-9
    W = np.where(across, D, np.inf).min(axis=1)
    return np.where(np.isfinite(W), W, np.ptp(P, axis=0).max())


def triangulate_polygon(poly):
    """Triangulate a simple 2D polygon with Blender's tessellator, returned CCW (+Z normal)."""
    from mathutils.geometry import tessellate_polygon
    pts = [Vector((float(p[0]), float(p[1]), 0.0)) for p in poly]
    tris = tessellate_polygon([pts])
    out = []
    for a, b, c in tris:
        pa, pb, pc = poly[a], poly[b], poly[c]
        cr = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        out.append((a, b, c) if cr > 0 else (a, c, b))
    return out


# ---------------------------------------------------------------------------
# blade
# ---------------------------------------------------------------------------

class Blade:
    """Shinogi-zukuri katana blade described in curved (tau, f, x) coordinates.

    tau : 0 at the habaki -> 1 at the tip (along the back line)
    f   : 0 at the back ridge (mune) -> 1 at the cutting edge
    x   : thickness offset along +/-X
    """

    L = 0.745          # blade length
    SORI = 0.042       # curvature (sagitta at the tip)
    W0 = 0.0345        # width at the habaki
    WY = 0.0255        # width at the yokote
    TY = 0.915         # tau of the yokote (kissaki starts)
    K0 = 0.0047        # half thickness at the shinogi, base
    KY = 0.0033        # half thickness at the yokote
    TIP_F = 0.30       # where the tip sits between back and edge (fraction of WY)
    HI_END = 0.80      # end of the fuller
    F_SHINOGI = 0.30
    F_SJ = 0.06
    GROOVE_C = 0.16
    GROOVE_HW = 0.075
    JI_S = (0.12, 0.27, 0.42, 0.57, 0.71, 0.83, 0.92, 0.97)
    U0, U1 = 0.02, 0.48  # texture strip of the +X side (the -X side is mirrored)

    def __init__(self):
        tt = np.linspace(0.0, 1.0, 4001)
        p = self.C(tt)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        self._tt = tt
        self._arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self._arc[-1])

    # reference curve = the back line before the kissaki
    def C(self, tau):
        tau = np.asarray(tau, dtype=np.float64)
        y = 0.5 * self.W0 + self.SORI * (0.62 * tau ** 2 + 0.38 * tau ** 4)
        z = tau * self.L
        return np.stack([np.zeros_like(tau), y, z], axis=-1)

    def frame(self, tau):
        tau = np.asarray(tau, dtype=np.float64)
        dy = self.SORI * (1.24 * tau + 1.52 * tau ** 3)
        dz = np.full_like(tau, self.L)
        n = np.sqrt(dy * dy + dz * dz)
        T = np.stack([np.zeros_like(tau), dy / n, dz / n], axis=-1)
        Nrm = np.stack([np.zeros_like(tau), -T[..., 2], T[..., 1]], axis=-1)  # towards the edge
        return T, Nrm

    def bounds(self, tau):
        """Distance of back line and edge from the reference curve."""
        tau = np.asarray(tau, dtype=np.float64)
        w = self.W0 + (self.WY - self.W0) * np.minimum(tau, self.TY) / self.TY
        s = np.clip((tau - self.TY) / (1.0 - self.TY), 0.0, 1.0)
        d_tip = self.TIP_F * self.WY
        d_b = d_tip * s ** 1.8
        d_e = np.where(tau <= self.TY, w, d_tip + (self.WY - d_tip) * np.clip(1 - s ** 2.2, 0, 1) ** (1 / 1.7))
        return d_b, d_e

    def k(self, tau):
        tau = np.asarray(tau, dtype=np.float64)
        s = np.clip((tau - self.TY) / (1.0 - self.TY), 0.0, 1.0)
        base = self.K0 + (self.KY - self.K0) * np.minimum(tau, self.TY) / self.TY
        return np.where(tau <= self.TY, base, self.KY * np.clip(1 - s ** 1.6, 0, 1) ** 0.7)

    def groove(self, tau):
        zone = 0.04
        if tau < self.HI_END - zone:
            return 1.0
        if tau < self.HI_END:
            q = (tau - (self.HI_END - zone)) / zone
            return math.sqrt(max(0.0, 1 - q * q))
        return 0.0

    def arc(self, tau):
        return np.interp(tau, self._tt, self._arc)

    def tau_of_arc(self, a):
        return np.interp(a, self._arc, self._tt)

    def pos(self, tau, f, x):
        tau = np.asarray(tau, dtype=np.float64)
        T, Nrm = self.frame(tau)
        d_b, d_e = self.bounds(tau)
        d = d_b + np.asarray(f) * (d_e - d_b)
        P = self.C(tau) + d[..., None] * Nrm
        P = P + np.stack([np.asarray(x, dtype=np.float64) * np.ones_like(d), np.zeros_like(d), np.zeros_like(d)], axis=-1)
        return P

    def profile(self, tau):
        """(f, x) points of the +X half, back ridge -> edge."""
        k = float(self.k(tau))
        g = self.groove(tau)
        sj, sh = self.F_SJ, self.F_SHINOGI

        def surf(f):
            return k * (0.80 + 0.20 * (f - sj) / (sh - sj))

        hw = self.GROOVE_HW * (0.35 + 0.65 * g)
        c = self.GROOVE_C
        depth = 0.36 * k * g
        pts = [(0.0, 0.0), (0.035, 0.58 * k), (sj, 0.80 * k)]
        pts += [(c - hw, surf(c - hw)),
                (c - 0.62 * hw, surf(c - 0.62 * hw) - 0.78 * depth),
                (c, surf(c) - depth),
                (c + 0.62 * hw, surf(c + 0.62 * hw) - 0.78 * depth),
                (c + hw, surf(c + hw))]
        pts.append((sh, k))
        expo = 0.82 if tau <= self.TY else 0.98
        for s in self.JI_S:
            pts.append((sh + (1 - sh) * s, k * (1 - s) ** expo))
        pts.append((1.0, 0.0))
        return pts

    def u_of(self, f, side):
        return self.U0 + (self.U1 - self.U0) * f if side > 0 else 1.0 - (self.U0 + (self.U1 - self.U0) * f)

    @staticmethod
    def v_of(tau):
        return 0.004 + 0.992 * tau

    def taus(self):
        out, t = [], 0.0
        while t < self.TY - 1e-6:
            out.append(t)
            near = abs(t - (self.HI_END - 0.02)) < 0.035
            t += 0.0035 if near else 0.0105
        out.append(self.TY)
        u = np.linspace(0.0, 1.0, 24)[1:-1]
        s = 1 - (1 - u) ** 1.35
        out += list(self.TY + (1 - self.TY) * s)
        return np.array(out)


BLADE = Blade()


def build_blade(col):
    bl = BLADE
    mb = MeshBuilder()
    taus = bl.taus()
    n = len(bl.profile(0.0))
    M = 2 * n - 2

    def ring_layout(prof):
        """Closed ring: +X side back->edge, then -X side edge->back."""
        out = [(f, x, +1) for f, x in prof]
        out += [(prof[i][0], -prof[i][1], -1) for i in range(n - 2, 0, -1)]
        return out

    ring_base = []
    ring_meta = []
    for tau in taus:
        lay = ring_layout(bl.profile(tau))
        P = [bl.pos(tau, f, x) for f, x, _ in lay]
        ring_base.append(mb.add_verts(P))
        ring_meta.append(lay)

    def uv_corner(r, j, side_hint):
        f, x, side = ring_meta[r][j]
        if j == 0 or j == n - 1:   # shared ridge / edge vertices take the face's side
            side = side_hint
        return (bl.u_of(f, side), bl.v_of(taus[r])), (f, taus[r])

    def mirror(i):
        return (2 * n - 2 - i) % M

    for r in range(len(taus) - 1):
        a0, a1 = ring_base[r], ring_base[r + 1]
        for j in range(M):
            j1 = (j + 1) % M
            side = +1 if j < n - 1 else -1
            corners = [(r, j1), (r, j), (r + 1, j), (r + 1, j1)]
            uv = [uv_corner(rr, jj, side) for rr, jj in corners]
            mb.add_face([a0 + j1, a0 + j, a1 + j, a1 + j1], [u for u, _ in uv], [q for _, q in uv], 0)
        g0, g1 = bl.groove(taus[r]), bl.groove(taus[r + 1])
        cols = [0, n - 1, 8, mirror(8)]
        if g0 > 0.05 and g1 > 0.05:
            cols += [3, 7, mirror(3), mirror(7)]
        for j in cols:
            mb.sharp.add((a0 + j, a1 + j))
    # crisp yokote across the ji
    ry = int(np.argmin(np.abs(taus - bl.TY)))
    for j in range(8, n - 1):
        mb.sharp.add((ring_base[ry] + j, ring_base[ry] + j + 1))
        mb.sharp.add((ring_base[ry] + mirror(j), ring_base[ry] + mirror(j + 1)))

    # pointed tip
    last = len(taus) - 1
    tip = mb.add_verts([bl.pos(1.0, 0.0, 0.0)])
    for j in range(M):
        j1 = (j + 1) % M
        side = +1 if j < n - 1 else -1
        uv = [uv_corner(last, j, side), uv_corner(last, j1, side)]
        tip_uv = ((bl.u_of(0.5, side), bl.v_of(1.0)), (0.5, 1.0))
        mb.add_face([ring_base[last] + j1, ring_base[last] + j, tip],
                    [uv[1][0], uv[0][0], tip_uv[0]], [uv[1][1], uv[0][1], tip_uv[1]], 0)
    # closed base inside the habaki
    base_c = mb.add_verts([bl.pos(0.0, 0.5, 0.0)])
    for j in range(M):
        j1 = (j + 1) % M
        side = +1 if j < n - 1 else -1
        uv = [uv_corner(0, j, side), uv_corner(0, j1, side)]
        mb.add_face([ring_base[0] + j, ring_base[0] + j1, base_c],
                    [uv[0][0], uv[1][0], (bl.u_of(0.5, side), 0.0)], [uv[0][1], uv[1][1], (0.5, 0.0)], 0)

    obj = mb.to_object("Katana_Blade", col)
    return obj


# ---------------------------------------------------------------------------
# preview rendering (Cycles -> linear EXR -> numpy bloom -> PNG)
# ---------------------------------------------------------------------------

def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def box_blur(img, rx, ry):
    """Separable box blur (axis 0 = rows) using cumulative sums, edge clamped."""
    out = img
    for axis, r in ((0, ry), (1, rx)):
        r = int(r)
        if r < 1:
            continue
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r + 1, r)
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float64)
        hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
        out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


def gauss_blur(img, rx, ry=None):
    ry = rx if ry is None else ry
    for _ in range(3):
        img = box_blur(img, max(1, rx / 1.73), max(1, ry / 1.73))
    return img


def setup_world_and_lights(col):
    scene = bpy.context.scene
    world = bpy.data.worlds.new("PreviewWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*srgb_to_linear(hexc("#26305e")), 1.0)
    bg.inputs["Strength"].default_value = 0.55

    def area(name, loc, target, energy, color, size):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy = energy
        ld.color = tuple(srgb_to_linear(hexc(color)))
        ld.size = size
        ob = bpy.data.objects.new(name, ld)
        ob.location = loc
        d = Vector(target) - Vector(loc)
        ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        col.objects.link(ob)
        return ob

    area("Key", (0.9, 0.5, 0.75), (0, 0, 0.15), 9, "#fff4ea", 0.9)
    area("Rim", (-0.8, -0.7, 0.9), (0, 0, 0.25), 12, "#ff7fb5", 0.6)
    area("Fill", (0.6, -0.9, -0.1), (0, 0, 0.1), 3, "#8fa6ff", 1.2)


def add_camera(col, name, loc, target, lens=85.0, ortho=None):
    cd = bpy.data.cameras.new(name)
    if ortho:
        cd.type = 'ORTHO'
        cd.ortho_scale = ortho
    else:
        cd.lens = lens
    cd.clip_start = 0.01
    cd.clip_end = 50
    ob = bpy.data.objects.new(name, cd)
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    col.objects.link(ob)
    return ob


def render_view(cam, path, res=(1200, 1600), samples=64, bloom=True, bg_grad=True):
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    exr = path[:-4] + "_tmp.exr"
    scene.render.filepath = exr
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(exr)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    bpy.data.images.remove(img)
    os.remove(exr)
    rgb, alpha = px[..., :3], px[..., 3:4]
    # backdrop: the game's navy night sky with a soft vignette
    yy, xx = np.mgrid[0:h, 0:w]
    top = srgb_to_linear(hexc("#2b376b"))
    bot = srgb_to_linear(hexc("#141a3a"))
    t = (yy / max(h - 1, 1))[..., None]
    back = bot * (1 - t) + top * t
    r = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)[..., None]
    back = back * (1.0 - 0.45 * np.clip(r * 1.4, 0, 1) ** 2)
    out = rgb + back * (1 - alpha)
    if bloom:
        bright = np.clip(rgb - 0.9, 0, None)
        glow = gauss_blur(bright, w * 0.004) * 0.6 + gauss_blur(bright, w * 0.015) * 0.5 + gauss_blur(bright, w * 0.045) * 0.35
        out = out + glow
    out = out / (1.0 + 0.12 * np.maximum(out - 1.0, 0))  # gentle shoulder
    out = linear_to_srgb(np.clip(out, 0, 1.0))
    save_rgba(path, np.concatenate([out, np.ones_like(alpha)], axis=-1), srgb=True)


def save_rgba(path, arr, srgb=True, name=None):
    """arr: (h, w, 3|4) float in display space (already sRGB-encoded if colour)."""
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape[:2]
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), np.float32)], axis=-1)
    img = bpy.data.images.new(name or os.path.basename(path), w, h, alpha=True)
    img.colorspace_settings.name = 'sRGB' if srgb else 'Non-Color'
    img.pixels.foreach_set(np.clip(arr, 0, 1).ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save()
    return img


# ---------------------------------------------------------------------------
# hand-drawn brush painter for the blade texture strip
# ---------------------------------------------------------------------------

def noise1d(n, seed, knots=12):
    rng = np.random.default_rng(seed)
    k = rng.random(knots + 3) * 2 - 1
    x = np.linspace(0, knots, n)
    xi = np.floor(x).astype(int)
    fx = x - xi
    fx = fx * fx * (3 - 2 * fx)
    return k[xi] * (1 - fx) + k[xi + 1] * fx


def brush_stroke(path, radius, seed, wobble=0.016, taper_in=0.14, taper_out=0.45, blob=0.22, spacing=0.02, smooth=True):
    """Turn a control polyline into a calligraphic stroke (points + radii).

    Pressure starts with a small ink blob, swells, then lifts off with a long taper;
    the spine wobbles slightly like a real hand.
    """
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) > 2 and smooth:
        pts = catmull_rom(pts, 10)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = max(seg.sum(), 1e-9)
    n = max(4, int(L / spacing) + 2)
    p = resample_polyline(pts, n)
    s = np.linspace(0, 1, n)
    tng = polyline_tangents(np.column_stack([p, np.zeros(n)]))[:, :2]
    nrm = np.stack([-tng[:, 1], tng[:, 0]], axis=1)
    p = p + nrm * (noise1d(n, seed, knots=max(2, int(L * 6))) * wobble)[:, None]
    press = (0.62 + 0.38 * smoothstep(0, taper_in, s)) * (1 - 0.82 * smoothstep(1 - taper_out, 1.0, s) ** 1.3)
    press = press + blob * np.exp(-((s - 0.035) / 0.045) ** 2)
    rng = np.random.default_rng(seed + 7)
    return p, radius * press * (0.9 + 0.2 * rng.random())


class StripCanvas:
    """Pixel grid of the blade strip with each pixel's physical (along, across) coordinate."""

    def __init__(self, W, H):
        bl = BLADE
        self.W, self.H = W, H
        u = (np.arange(W) + 0.5) / W
        v = (np.arange(H) + 0.5) / H
        U, V = np.meshgrid(u, v)
        tau = np.clip((V - 0.004) / 0.992, 0, 1)
        side = np.where(U < 0.5, 1, -1)
        f_raw = np.where(side > 0, (U - bl.U0) / (bl.U1 - bl.U0), (1 - U - bl.U0) / (bl.U1 - bl.U0))
        self.F = np.clip(f_raw, 0, 1)
        self.TAU = tau
        self.SIDE = side
        d_b, d_e = bl.bounds(tau)
        self.WID = np.maximum(d_e - d_b, 1e-5)
        self.D = d_b + self.F * self.WID
        self.A = bl.arc(tau)
        self.px_v = bl.length / 0.992 / H
        self.aa = 0.00018

    def rows_for(self, a_lo, a_hi):
        bl = BLADE
        t0, t1 = bl.tau_of_arc(a_lo), bl.tau_of_arc(a_hi)
        y0 = int(max(0, math.floor(bl.v_of(t0) * self.H) - 2))
        y1 = int(min(self.H, math.ceil(bl.v_of(t1) * self.H) + 2))
        return y0, y1

    def cols_for(self, f_lo, f_hi, side):
        bl = BLADE
        u_a, u_b = bl.u_of(max(f_lo, -0.01), side), bl.u_of(min(f_hi, 1.01), side)
        x0 = int(max(0, math.floor(min(u_a, u_b) * self.W) - 2))
        x1 = int(min(self.W, math.ceil(max(u_a, u_b) * self.W) + 2))
        return x0, x1

    def stamp(self, layer, pts_ad, radii, side, opacity=1.0):
        """Rasterise a variable-width polyline given in (along, across) metres."""
        pts_ad = np.asarray(pts_ad)
        for i in range(len(pts_ad) - 1):
            p0, p1 = pts_ad[i], pts_ad[i + 1]
            r0, r1 = radii[i], radii[i + 1]
            rm = max(r0, r1) + 2 * self.aa
            y0, y1 = self.rows_for(min(p0[0], p1[0]) - rm, max(p0[0], p1[0]) + rm)
            if y1 <= y0:
                continue
            Dlo, Dhi = min(p0[1], p1[1]) - rm, max(p0[1], p1[1]) + rm
            wid = self.WID[y0:y1, 0]
            dbase = (self.D[y0:y1, 0] - self.F[y0:y1, 0] * wid)
            f_lo = float(np.min((Dlo - dbase) / wid))
            f_hi = float(np.max((Dhi - dbase) / wid))
            x0, x1 = self.cols_for(f_lo, f_hi, side)
            if x1 <= x0:
                continue
            A = self.A[y0:y1, x0:x1]
            D = self.D[y0:y1, x0:x1]
            sm = (self.SIDE[y0:y1, x0:x1] == side)
            da, dd = p1[0] - p0[0], p1[1] - p0[1]
            l2 = da * da + dd * dd
            t = np.clip(((A - p0[0]) * da + (D - p0[1]) * dd) / l2, 0, 1) if l2 > 0 else np.zeros_like(A)
            dist = np.hypot(A - (p0[0] + t * da), D - (p0[1] + t * dd))
            r = r0 + (r1 - r0) * t
            cov = np.clip((r - dist) / self.aa + 0.5, 0, 1) * opacity * sm
            layer[y0:y1, x0:x1] = np.maximum(layer[y0:y1, x0:x1], cov)


# -- a small invented script -------------------------------------------------
# Each letter is built from a handful of brush strokes in a 1 x 1 cell
# (x across the blade, y along it).  A fixed alphabet means letters repeat
# like real writing instead of every mark being random noise.

def make_alphabet(seed=4242, count=20):
    """Angular carved letters: a stem plus branches, bars, bowls and hooks.

    Strokes are drawn top-to-bottom / stem-outward like a carver would, so the
    taper always lands at the free end of a stroke.
    """
    rng = np.random.default_rng(seed)
    letters, seen = [], set()
    while len(letters) < count:
        strokes, sig = [], []
        x0 = float(rng.choice([-0.08, 0.0, 0.07]))
        has_stem = rng.random() < 0.86
        if has_stem:
            short = rng.random() < 0.15
            bend = float(rng.uniform(-0.04, 0.04))
            strokes.append([(x0, 0.5), (x0 + bend, 0.0 if not short else 0.25), (x0, -0.5 if not short else 0.0)])
            sig.append(("stem", short))
        n_feat = int(rng.choice([1, 2, 2, 3]))
        used = set()
        for _ in range(n_feat):
            kind = str(rng.choice(["branch", "branch", "bar", "bowl", "hook", "chev", "dot"]))
            side = float(rng.choice([-1.0, 1.0]))
            y = float(rng.choice([0.42, 0.12, -0.18]))
            slot = (kind, side, y)
            if slot in used:
                continue
            used.add(slot)
            if kind == "branch":
                up = float(rng.choice([-1.0, 1.0]))
                strokes.append([(x0, y), (x0 + side * 0.2, y + up * 0.15), (x0 + side * 0.4, y + up * 0.31)])
            elif kind == "bar":
                yy = 0.22 if y > 0 else -0.08
                strokes.append([(-0.36, yy + 0.04), (0.0, yy), (0.36, yy - 0.05)])
            elif kind == "bowl":   # pointed, thorn-like bowl rather than a round Latin one
                strokes.append([(x0, 0.46), (x0 + side * 0.34, 0.27), (x0, 0.06)])
            elif kind == "hook":
                strokes.append([(x0, 0.5), (x0 + side * 0.17, 0.6), (x0 + side * 0.33, 0.48)])
            elif kind == "chev":
                strokes.append([(x0 - 0.34, y + 0.2), (x0, y - 0.1), (x0 + 0.34, y + 0.2)])
            else:
                strokes.append([(x0 + side * 0.3, y + 0.03), (x0 + side * 0.33, y - 0.05)])
            sig.append(slot)
        key = tuple(sorted(map(str, sig)))
        if key in seen or len(strokes) < 2:
            continue
        seen.add(key)
        letters.append(strokes)
    return letters


ALPHABET = make_alphabet()


def paint_blade(res=(512, 4096)):
    W, H = res
    bl = BLADE
    cv = StripCanvas(W, H)
    F, A, D, TAU, SIDE, WID = cv.F, cv.A, cv.D, cv.TAU, cv.SIDE, cv.WID
    nz = Noise2(11)
    nz2 = Noise2(12)
    zeros = lambda: np.zeros((H, W), np.float32)

    # ---------------- hamon line (temper line) ----------------
    hamon_f = np.zeros((H, W))
    for side in (1, -1):
        rng = np.random.default_rng(100 + side)
        a_samp = np.linspace(0, bl.length, 6000)
        fh = np.full_like(a_samp, 0.86)
        pos = 0.02
        while pos < bl.length:
            per = rng.uniform(0.010, 0.021)
            amp = rng.uniform(0.026, 0.062) * sstep(0.03, 0.085, pos)
            x = (a_samp - pos) / per
            inside = (x >= 0) & (x <= 1)
            bump = np.where(inside, np.sin(np.clip(x, 0, 1) * math.pi) ** 0.75, 0)
            if rng.random() < 0.3:   # choji: a second small crown on the bump
                bump = bump + np.where(inside, 0.45 * np.sin(np.clip(x, 0, 1) * 2 * math.pi) ** 2 * (x > 0.5), 0)
            fh = fh - amp * bump
            pos += per * rng.uniform(0.85, 1.05)
        fh += 0.008 * noise1d(len(a_samp), 300 + side, knots=180)
        # boshi: inside the kissaki the line follows the fukura and turns back
        tau_s = bl.tau_of_arc(a_samp)
        s = np.clip((tau_s - bl.TY) / (1 - bl.TY), 0, 1)
        fh = fh * (1 - smoothstep(0.0, 0.35, s)) + (0.80 - 0.50 * smoothstep(0.55, 1.0, s)) * smoothstep(0.0, 0.35, s)
        m = SIDE == side
        hamon_f[m] = np.interp(A[m], a_samp, fh)
    dist_h = (F - hamon_f) * WID          # metres, >0 on the edge side
    ha = smoothstep(-0.00025, 0.00025, dist_h)

    # ---------------- strokes: runes, crest, ashi, scratches ----------------
    rune = zeros()
    rune_dim = zeros()
    ashi = zeros()
    scratch = zeros()
    sparkle = zeros()
    band_f = 0.565

    def glyph_to_ad(strokes, a_c, scale_a, scale_d, rot, side, seed, fade=1.0):
        tau = float(bl.tau_of_arc(a_c))
        d_b, d_e = bl.bounds(tau)
        w = float(d_e - d_b)
        d_c = float(d_b) + band_f * w
        cr, sr = math.cos(rot), math.sin(rot)
        rng = np.random.default_rng(seed)
        for si, st in enumerate(strokes):
            if fade < 1.0 and rng.random() > fade + 0.25:
                continue
            p, r = brush_stroke(st, 0.058, seed * 31 + si, wobble=0.012, blob=0.14, smooth=False)
            if fade < 1.0:   # letters near the tip break apart like burnt paper
                cut = int(len(p) * rng.uniform(0.45 + 0.5 * fade, 1.0))
                p, r = p[:max(cut, 3)], r[:max(cut, 3)]
            x = p[:, 0] * cr - p[:, 1] * sr
            y = p[:, 0] * sr + p[:, 1] * cr
            ad = np.column_stack([a_c + y * scale_a, d_c + x * scale_d * side])
            target = rune if fade >= 0.999 else rune_dim
            cv.stamp(target, ad, r * scale_d * 0.92, side, opacity=1.0 if fade >= 0.999 else 0.35 + 0.65 * fade)

    for side in (1, -1):
        rng = np.random.default_rng(900 + side)
        # crest: a crescent moon with a four-point star, echoing the menuki
        a_c = 0.098
        tau = float(bl.tau_of_arc(a_c))
        d_b, d_e = bl.bounds(tau)
        w = float(d_e - d_b)
        d_c = float(d_b) + band_f * w
        R = 0.0052
        ang = np.linspace(math.radians(-35), math.radians(215), 40)
        moon = np.column_stack([a_c + R * np.sin(ang), d_c + R * np.cos(ang) * side])
        rr = 0.00085 * (np.sin(np.linspace(0, math.pi, 40)) ** 0.7 + 0.08)
        cv.stamp(rune, moon, rr, side)
        for k, ln in ((0, 0.0042), (1, 0.0026)):
            th = k * math.pi / 2 + 0.2
            pts = np.array([[a_c + 0.0006 - ln * math.cos(th), d_c - ln * math.sin(th) * side],
                            [a_c + 0.0006, d_c],
                            [a_c + 0.0006 + ln * math.cos(th), d_c + ln * math.sin(th) * side]])
            cv.stamp(rune, pts, np.array([0.00008, 0.0006, 0.00008]), side)

        # running text
        a = 0.121
        a_end = 0.585
        word_left = int(rng.integers(2, 5))
        while a < a_end:
            tau = float(bl.tau_of_arc(a))
            d_b, d_e = bl.bounds(tau)
            w = float(d_e - d_b)
            gh = 0.0172 * (w / bl.W0) ** 0.5
            gw = 0.46 * w * 0.84
            fade = 1.0 - sstep(0.47, a_end, a) * 0.8
            letter = ALPHABET[int(rng.integers(0, len(ALPHABET)))]
            glyph_to_ad(letter, a + gh * 0.5, gh, gw, float(rng.uniform(-0.07, 0.07)), side,
                        seed=int(rng.integers(0, 1 << 30)), fade=fade)
            a += gh * rng.uniform(1.28, 1.42)
            word_left -= 1
            if word_left == 0:
                # word separator: a small lozenge of two short dabs
                sep_a = a + gh * 0.12
                d_cs = float(d_b) + band_f * w
                for k in (-1, 1):
                    pts = np.array([[sep_a, d_cs + k * 0.0011 * side], [sep_a + 0.0012, d_cs + k * 0.00025 * side]])
                    tgt = rune if fade >= 0.999 else rune_dim
                    cv.stamp(tgt, pts, np.array([0.00055, 0.0002]), side, opacity=fade)
                a += gh * 0.62
                word_left = int(rng.integers(2, 6))
        # embers trailing off toward the tip
        for _ in range(9):
            ea = rng.uniform(0.58, 0.66)
            tau = float(bl.tau_of_arc(ea))
            d_b, d_e = bl.bounds(tau)
            ed = float(d_b) + rng.uniform(0.4, 0.75) * float(d_e - d_b)
            ln = rng.uniform(0.0006, 0.0018)
            th = rng.uniform(0.3, 1.2)
            pts = np.array([[ea, ed], [ea + ln * math.cos(th), ed + ln * math.sin(th) * side]])
            cv.stamp(rune_dim, pts, np.array([0.00035, 0.0001]), side, opacity=rng.uniform(0.35, 0.8))

        # ashi: soft legs running from the temper line toward the edge
        a = 0.05
        while a < bl.arc(bl.TY):
            tau = float(bl.tau_of_arc(a))
            d_b, d_e = bl.bounds(tau)
            w = float(d_e - d_b)
            m = (SIDE == side)
            row = int(bl.v_of(tau) * H)
            fh = float(hamon_f[min(row, H - 1), :][m[min(row, H - 1), :]].mean())
            d0 = float(d_b) + fh * w + 0.0002
            ln = rng.uniform(0.0012, 0.0028)
            lean = rng.uniform(-0.0006, 0.0012)
            pts = np.array([[a, d0], [a + lean * 0.5, d0 + ln * 0.5], [a + lean, d0 + ln]])
            cv.stamp(ashi, pts, np.array([0.00042, 0.0003, 0.00008]), side, opacity=rng.uniform(0.5, 1.0))
            a += rng.uniform(0.006, 0.016)

        # fine polishing scratches, mostly running with the polish direction
        for _ in range(45):
            sa = rng.uniform(0.04, bl.length * 0.97)
            tau = float(bl.tau_of_arc(sa))
            d_b, d_e = bl.bounds(tau)
            sd = float(d_b) + rng.uniform(0.05, 0.95) * float(d_e - d_b)
            ln = rng.uniform(0.003, 0.013)
            th = rng.uniform(-0.35, 0.35) + (rng.uniform(0.6, 1.1) if rng.random() < 0.3 else 0)
            mid = np.array([sa + 0.5 * ln * math.cos(th), sd + 0.5 * ln * math.sin(th) + rng.uniform(-1, 1) * 0.0002])
            pts = np.array([[sa, sd], mid, [sa + ln * math.cos(th), sd + ln * math.sin(th)]])
            cv.stamp(scratch, pts, np.array([0.00004, 0.000045, 0.00001]), side, opacity=rng.uniform(0.2, 0.6))

        # nie: frosty crystals scattered just inside the temper line
        for _ in range(220):
            na = rng.uniform(0.035, bl.arc(1.0) * 0.985)
            tau = float(bl.tau_of_arc(na))
            d_b, d_e = bl.bounds(tau)
            w = float(d_e - d_b)
            row = min(int(bl.v_of(tau) * H), H - 1)
            fh = float(hamon_f[row][SIDE[row] == side].mean())
            nd = float(d_b) + fh * w + rng.exponential(0.0007) * (1 if rng.random() < 0.85 else -0.4)
            r = rng.uniform(0.00005, 0.00013)
            pts = np.array([[na, nd], [na + rng.uniform(-1, 1) * 0.0002, nd + 0.0001]])
            cv.stamp(sparkle, pts, np.array([r, r * 0.6]), side, opacity=rng.uniform(0.4, 1.0))

    # ---------------- fields ----------------
    streak = nz.fbm(D / 0.0007, A / 0.045, octaves=4)             # long polish strokes
    grain = nz2.fbm(D / 0.0024 + 2.0 * nz(A / 0.05, D / 0.006), A / 0.035, octaves=3)
    grain_lines = (np.abs(np.sin(grain * 30.0)) ** 18).astype(np.float32)   # itame-like hada
    mottle = nz.fbm(A / 0.006, D / 0.0016, octaves=3)
    in_kissaki = TAU > bl.TY
    g = np.vectorize(bl.groove)(TAU[:, 0])[:, None] * np.ones((1, W))
    hw = bl.GROOVE_HW * (0.35 + 0.65 * g)
    groove_m = smoothstep(0.0, 0.25, g) * (1 - smoothstep(hw * 0.85, hw * 1.02, np.abs(F - bl.GROOVE_C)))
    groove_floor = groove_m * (1 - smoothstep(hw * 0.12, hw * 0.38, np.abs(F - bl.GROOVE_C)))
    shinogi_hl = np.exp(-((F - bl.F_SHINOGI) * WID / 0.00028) ** 2)
    mune_hl = np.exp(-(F * WID / 0.00035) ** 2)
    yokote = np.exp(-((TAU - bl.TY) * bl.length / 0.00035) ** 2) * (F > bl.F_SHINOGI - 0.02)

    def mixc(base, col, t):
        col = col if col.ndim == 3 else col[None, None, :]
        return base * (1 - t[..., None]) + col * t[..., None]

    # ---------------- albedo ----------------
    shade = smoothstep(0.0, 0.9, F)
    alb = mixc(np.broadcast_to(PAL["steel_deep"], (H, W, 3)).copy(), PAL["steel_mid"], shade * 0.9)
    alb = mixc(alb, PAL["steel_lift"], smoothstep(0.35, 0.86, F) * (1 - ha) * 0.75)
    sheen = 0.5 + 0.5 * np.sin((A - F * WID * 9.0) / 0.075 * 2 * math.pi + 1.3)
    alb = alb * (0.88 + 0.16 * sheen[..., None] * (F > 0.06)[..., None])
    alb = alb * (0.92 + 0.16 * streak[..., None])
    alb = alb + 0.035 * grain_lines[..., None] * (1 - ha)[..., None] * (F > bl.F_SHINOGI)[..., None]
    alb = mixc(alb, PAL["steel_deep"] * 0.7, groove_m * 0.85)
    alb = mixc(alb, PAL["ha_pale"] * (0.86 + 0.18 * mottle[..., None]), ha * 0.95)
    alb = mixc(alb, PAL["ha_warm"], ha * np.exp(-np.clip(dist_h, 0, None) / 0.0016) * 0.55)
    alb = mixc(alb, PAL["steel_edge"], np.clip(shinogi_hl * 0.85 + mune_hl * 0.5 + yokote * 0.8, 0, 1))
    alb = alb + 0.14 * scratch[..., None]
    alb = alb + 0.35 * sparkle[..., None]
    halo = gauss_blur(rune, 9, 7) + 0.6 * gauss_blur(rune_dim, 7, 6)
    carve = np.clip(gauss_blur(rune + rune_dim, 2.2, 2.0) * 2.2 - (rune + rune_dim) * 2.0, 0, 1)
    alb = mixc(alb, PAL["ink"], carve * 0.75)
    alb = mixc(alb, PAL["glow_hot"] * 0.55, np.clip(halo * 0.5, 0, 0.5) * (1 - carve))
    alb = mixc(alb, PAL["glow_core"], np.clip(rune + rune_dim * 0.8, 0, 1))
    alb = mixc(alb, PAL["glow_hot"], np.clip(ashi * 0.4, 0, 1))
    alb = np.clip(alb, 0, 1)

    # ---------------- emission ----------------
    emi = np.zeros((H, W, 3))

    def add(col, amt):
        nonlocal emi
        emi = emi + col[None, None, :] * np.clip(amt, 0, None)[..., None]

    flick = 0.7 + 0.3 * nz(A / 0.018, SIDE * 3.1)
    line_core = np.exp(-(dist_h / 0.00028) ** 2)
    nioi = np.exp(-np.clip(dist_h, 0, None) / 0.0013) * (dist_h > 0) + np.exp(-np.clip(-dist_h, 0, None) / 0.0003) * (dist_h <= 0)
    add(PAL["glow_hot"], (nioi * 0.55 + line_core * 0.5) * flick)
    add(PAL["glow_core"], line_core * 0.45 * flick)
    add(PAL["glow_hot"], ashi * 0.55)
    edge_glow = smoothstep(0.972, 1.0, F)
    add(PAL["glow_hot"], edge_glow * 0.9 * (0.75 + 0.25 * nz(A / 0.01, 7.0)))
    add(PAL["glow_core"], smoothstep(0.988, 1.0, F) * 0.6)
    dash = smoothstep(0.42, 0.5, nz2(A / 0.012, 4.0 + SIDE))
    add(PAL["glow_hot"], np.exp(-(F * WID / 0.00022) ** 2) * 0.32 * dash)
    vein = 0.45 + 0.55 * smoothstep(0.3, 0.8, nz(A / 0.025, 11.0 + SIDE))
    add(PAL["glow_deep"], groove_floor * 0.38 * vein)
    add(PAL["glow_hot"], np.clip(halo, 0, 1) * 0.55)
    add(PAL["glow_core"], np.clip(rune + 0.7 * rune_dim, 0, 1) * 0.95)
    add(PAL["glow_core"], sparkle * 0.45)
    emi = np.clip(emi, 0, 1)

    # ---------------- height -> tangent-space normal ----------------
    height = (-0.00028 * np.clip(rune + rune_dim, 0, 1) - 0.00003 * scratch
              - 0.00004 * ashi + 0.00002 * mottle * ha)
    height = gauss_blur(height.astype(np.float32), 1.2, 1.2)
    px_u = (bl.U1 - bl.U0) / W * WID
    gx = np.gradient(height, axis=1) / px_u
    gy = np.gradient(height, axis=0) / cv.px_v
    nrm = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = nrm * 0.5 + 0.5

    os.makedirs(TEX_DIR, exist_ok=True)
    out = {
        "albedo": os.path.join(TEX_DIR, "katana_blade_albedo.png"),
        "emission": os.path.join(TEX_DIR, "katana_blade_emission.png"),
        "normal": os.path.join(TEX_DIR, "katana_blade_normal.png"),
    }
    save_rgba(out["albedo"], alb, srgb=True, name="katana_blade_albedo")
    save_rgba(out["emission"], emi, srgb=True, name="katana_blade_emission")
    save_rgba(out["normal"], nrm, srgb=False, name="katana_blade_normal")
    return out


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def load_image(path, non_color=False):
    name = os.path.basename(path)
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.load(path)
    else:
        img.filepath = path
        img.reload()
    img.colorspace_settings.name = 'Non-Color' if non_color else 'sRGB'
    return img


def pbr_material(name, albedo, emission=None, normal=None, orm=None, metallic=0.0, roughness=0.5,
                 emission_strength=3.0, uv_name="UVMap"):
    """glTF-friendly Principled material fed by painted / baked textures."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = True       # closed meshes -> single-sided in glTF
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_name
    uv.location = (-900, 0)

    def tex(img, y, non_color=False):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = img
        n.location = (-550, y)
        n.interpolation = 'Linear'
        nt.links.new(uv.outputs["UV"], n.inputs["Vector"])
        return n

    a = tex(albedo, 300)
    nt.links.new(a.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None:
        e = tex(emission, 0)
        nt.links.new(e.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    if normal is not None:
        nimg = tex(normal, -300)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-150, -300)
        nm.uv_map = uv_name
        nt.links.new(nimg.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if orm is not None:
        o = tex(orm, -600)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-150, -600)
        nt.links.new(o.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    return mat


def flat_emissive_material(name, color_hex, strength=2.5, double_sided=True):
    """Unshaded-looking glow: black base, all colour in emission (exports cleanly to glTF)."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    bsdf.inputs["Emission Color"].default_value = (*srgb_to_linear(hexc(color_hex)), 1.0)
    bsdf.inputs["Emission Strength"].default_value = strength
    mat.use_backface_culling = not double_sided
    return mat


def blade_material(paths):
    return pbr_material("M_Katana_Blade", load_image(paths["albedo"]), load_image(paths["emission"]),
                        load_image(paths["normal"], non_color=True), metallic=0.2, roughness=0.38,
                        emission_strength=3.0)


# ---------------------------------------------------------------------------
# hilt: flame guard, seppa, handle, wrap, menuki, fuchi, kashira
# ---------------------------------------------------------------------------

Z_FUCHI_TOP = -0.0212
Z_WRAP_TOP = -0.0332
Z_WRAP_BOT = -0.2470
Z_KASHIRA_BOT = -0.2625
GRIP_CENTRE = 0.5 * (Z_WRAP_TOP + Z_WRAP_BOT)


def superellipse(n, hx, hy, e=2.3, phase=0.0):
    th = np.linspace(0, 2 * math.pi, n, endpoint=False) + phase
    c, s = np.cos(th), np.sin(th)
    x = hx * np.sign(c) * np.abs(c) ** (2 / e)
    y = hy * np.sign(s) * np.abs(s) ** (2 / e)
    return np.column_stack([x, y])


def loft_z(mb, rings, outline_fn, cap_top=None, cap_bottom=None, sharp_rings=(), island=None, split_every=None):
    """Stack 2D outlines (in XY) along Z.  rings = [(z, params...)] top -> bottom."""
    pts = np.array([[0.0, 0.0, r[0]] for r in rings])
    profiles = []
    for r in rings:
        o = outline_fn(*r[1:])
        # sweep frames: path runs -Z, so use N = +X, B = T x N
        profiles.append(np.column_stack([o[:, 0], -o[:, 1]])[::-1])   # CCW in (N, B)
    T = np.tile([0.0, 0.0, -1.0], (len(rings), 1))
    N = np.tile([1.0, 0.0, 0.0], (len(rings), 1))
    B = np.cross(T, N)
    ring_idx = sweep(mb, pts, (N, B), profiles, cap_start=cap_top, cap_end=cap_bottom, island=island)
    M = len(profiles[0])
    for ri in sharp_rings:
        for j in range(M):
            mb.sharp.add((ring_idx[ri] + j, ring_idx[ri] + (j + 1) % M))
    return ring_idx


def handle_axes(z):
    """Half sizes of the handle core (x = flat direction, y = edge/back direction)."""
    t = np.clip((Z_WRAP_TOP - z) / (Z_WRAP_TOP - Z_WRAP_BOT), 0, 1)
    waist = np.sin(np.pi * t) ** 1.2
    hx = 0.0122 - 0.0007 * waist - 0.0003 * t
    hy = 0.0160 - 0.0011 * waist - 0.0006 * t
    return hx, hy


HANDLE_E = 2.35


def handle_surface(theta, z, lift=0.0):
    """Point and normal on the superelliptic handle core at angle theta (0 = +X face)."""
    hx, hy = handle_axes(z)
    c, s = np.cos(theta), np.sin(theta)
    e = HANDLE_E
    x = hx * np.sign(c) * np.abs(c) ** (2 / e)
    y = hy * np.sign(s) * np.abs(s) ** (2 / e)
    nx = np.sign(x) * np.abs(x / hx) ** (e - 1) / hx
    ny = np.sign(y) * np.abs(y / hy) ** (e - 1) / hy
    nl = np.sqrt(nx * nx + ny * ny) + 1e-12
    nx, ny = nx / nl, ny / nl
    z = np.broadcast_to(z, np.shape(x))
    P = np.stack([x + nx * lift, y + ny * lift, z], axis=-1)
    Nn = np.stack([nx, ny, np.zeros_like(nx)], axis=-1)
    return P, Nn


# ---- flame guard -----------------------------------------------------------------
# Traced from the reference drawing (pixel coords, y down).  Each layer becomes a
# rounded plaque, stacked thicker toward the centre, and every layer sits on a
# slightly larger dark plaque so the drawing's heavy ink lines exist as geometry.

GUARD_OUTER = [(708, 212), (680, 262), (674, 330), (702, 390), (762, 432), (830, 470), (848, 505), (822, 560),
               (802, 610), (812, 632), (842, 628), (872, 590), (888, 532), (916, 560), (936, 612), (940, 662),
               (928, 722), (898, 770), (858, 802), (820, 830), (814, 850), (842, 853), (884, 820), (881, 882),
               (860, 940), (822, 974), (790, 982), (758, 992), (718, 1022), (678, 1050), (620, 1068), (560, 1062),
               (510, 1040), (480, 1010), (450, 960), (402, 895), (370, 858), (346, 802), (340, 766), (370, 795),
               (405, 808), (425, 795), (420, 770), (390, 740), (356, 700), (338, 640), (346, 580), (372, 518),
               (390, 452), (382, 410), (368, 380), (410, 384), (450, 402), (456, 432), (460, 470), (480, 481),
               (503, 465), (507, 422), (495, 390), (490, 345), (505, 300), (540, 266), (600, 236), (660, 220)]
GUARD_MID = [(620, 395), (652, 432), (692, 476), (745, 505), (768, 542), (760, 600), (756, 642), (776, 682),
             (810, 684), (840, 670), (853, 692), (845, 732), (800, 782), (762, 826), (745, 890), (720, 960),
             (680, 1000), (620, 1012), (560, 1005), (515, 970), (495, 900), (490, 810), (480, 730), (445, 662),
             (430, 600), (436, 556), (460, 536), (495, 546), (535, 540), (556, 508), (572, 450)]
GUARD_INNER = [(628, 527), (666, 546), (690, 600), (705, 680), (718, 740), (745, 746), (760, 740), (752, 772),
               (720, 802), (700, 842), (690, 890), (670, 922), (625, 935), (580, 920), (550, 880), (540, 820),
               (535, 760), (520, 720), (496, 680), (486, 630), (494, 614), (510, 640), (528, 650), (546, 630),
               (556, 595), (576, 560), (600, 536)]
GUARD_SLIT = [(625, 606), (637, 650), (645, 720), (648, 790), (640, 820), (625, 836), (610, 820), (600, 790),
              (603, 720), (612, 650)]
GUARD_H = 0.112                # overall height of the guard
GUARD_Z0 = -0.0195             # bottom of the flame rests on the lower seppa
GUARD_PX = GUARD_H / (1068 - 212)


def guard_yz(pts_px):
    P = np.asarray(pts_px, dtype=np.float64)
    return np.column_stack([(P[:, 0] - 625) * GUARD_PX, (1068 - P[:, 1]) * GUARD_PX + GUARD_Z0])


def guard_outline(pts_px, spacing=0.0011, grow=0.0):
    o = catmull_rom(guard_yz(pts_px), 8, closed=True)
    n = max(24, int(np.linalg.norm(np.diff(np.vstack([o, o[:1]]), axis=0), axis=1).sum() / spacing))
    o = resample_polyline(np.vstack([o, o[:1]]), n + 1)[:-1]
    if grow:
        area = np.sum(o[:, 0] * np.roll(o[:, 1], -1) - np.roll(o[:, 0], -1) * o[:, 1])
        t = np.roll(o, -1, axis=0) - np.roll(o, 1, axis=0)
        nrm = np.column_stack([t[:, 1], -t[:, 0]]) * (1 if area > 0 else -1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        o = o + nrm * grow
        for _ in range(2):
            o = 0.5 * o + 0.25 * (np.roll(o, 1, axis=0) + np.roll(o, -1, axis=0))
    return o


# (outline, half thickness, material slot); ink plaques sit just under each colour layer
GUARD_LAYERS = (
    (GUARD_OUTER, 0.0047, 0), (GUARD_MID, 0.0062, 1), (GUARD_INNER, 0.0077, 2), (GUARD_SLIT, 0.0088, 3),
)
GUARD_INK_SLOT = 4


def build_flame_guard(col):
    mb = MeshBuilder()
    X = np.array([1.0, 0.0, 0.0])
    Y = np.array([0.0, 1.0, 0.0])
    Z = np.array([0.0, 0.0, 1.0])
    prev_t = 0.0
    for pts, half, slot in GUARD_LAYERS:
        ink_half = prev_t + 0.6 * (half - prev_t) if prev_t else half - 0.0004
        for outline, h, mat, bevel, rings in ((guard_outline(pts, grow=0.0009), ink_half, GUARD_INK_SLOT, 0.0, 0),
                                              (guard_outline(pts), half, slot, 0.3, 2)):
            sub = MeshBuilder()
            sub.cur_mat = mat
            extrude_shape(sub, outline, 2 * h, bevel=bevel, bevel_rings=rings)
            sub.face_mat = [mat] * len(sub.faces)
            if mat == GUARD_INK_SLOT:   # flat ink: no need to spend atlas space on it
                sub.uv_phys = [[(u * 0.06, v * 0.06) for u, v in f] for f in sub.uv_phys]
            # local (x, y, thickness) -> world (Y, Z, X); Y x Z = X keeps it right-handed
            place_local(sub, np.zeros(3), Y, Z, X)
            mb.merge(sub)
        prev_t = half
    return mb.to_object("Hilt_FlameGuard", col)


def guard_tie_frame():
    """Frame round the lower-right flame tongue where the cord is knotted."""
    C = guard_yz([(858, 838)])[0]
    C = np.array([0.0, C[0], C[1]])
    T = np.array([0.0, 0.92, 0.39])
    T /= np.linalg.norm(T)
    N = np.array([0.0, 0.39, -0.92])
    return C, T, N, np.array([1.0, 0.0, 0.0]), 0.0034 * GUARD_H / 0.09, GUARD_LAYERS[0][1]


def plate(col, name, z_top, z_bot, hx, hy, e=2.6, wobble=0.0):
    mb = MeshBuilder()
    th = 0.25 * (z_top - z_bot)

    def outline(g):
        o = superellipse(48, hx + g, hy + g, e)
        if wobble:
            o = o * (1 + wobble * noise1d(48, 5, 6))[:, None]
        return o

    rings = [(z_top, -th * 0.8), (z_top - th * 0.3, -th * 0.1), (z_top - th, 0.0),
             (z_bot + th, 0.0), (z_bot + th * 0.3, -th * 0.1), (z_bot, -th * 0.8)]
    loft_z(mb, rings, outline, cap_top='flat', cap_bottom='flat')
    return mb.to_object(name, col)


# ---- handle core, ito wrap, menuki -------------------------------------------

def build_tsuka_core(col):
    mb = MeshBuilder()
    zs = np.linspace(Z_FUCHI_TOP - 0.004, Z_WRAP_BOT - 0.004, 34)

    def outline(z):
        hx, hy = handle_axes(z)
        return superellipse(32, hx, hy, HANDLE_E)

    loft_z(mb, [(z, z) for z in zs], outline, cap_top='flat', cap_bottom='flat')
    return mb.to_object("Hilt_Tsuka", col)


ITO_PITCH = 0.0425
ITO_W = 0.0086
ITO_T = 0.00105
ITO_Z0 = Z_WRAP_TOP - 0.0045    # first front crossing


def ito_ribbon(mb, direction, z_start, z_end, step=0.003):
    """One silk ribbon spiralling round the handle; direction = +1 / -1.

    The ribbons ride high on one face and low on the other, so they weave
    over / under each other at every crossing.  Near a crossing the ribbon
    bunches up into the raised twist of a real hineri-maki wrap.
    """
    # parametrise by z, densify by arc length later
    zz = np.linspace(z_start, z_end, 1600)
    th = direction * 2 * math.pi * (ITO_Z0 - zz) / ITO_PITCH
    base, h = 0.00025, 0.00125
    lift = base + h * (0.5 + 0.5 * direction * np.cos(th))
    P, Nn = handle_surface(th, zz, lift)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    m = int(cum[-1] / step) + 2
    target = np.linspace(0, cum[-1], m)
    zz = np.interp(target, cum, zz)
    th = direction * 2 * math.pi * (ITO_Z0 - zz) / ITO_PITCH
    lift = base + h * (0.5 + 0.5 * direction * np.cos(th))
    P, Nn = handle_surface(th, zz, lift)
    T = polyline_tangents(P)
    Nn = Nn - T * np.einsum("ij,ij->i", Nn, T)[:, None]
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    B = np.cross(T, Nn)
    near = np.exp(-(np.angle(np.exp(1j * th * 2)) / 0.5) ** 2)   # 1 at crossings (theta = 0, pi)
    profiles = []
    for k in range(m):
        w = ITO_W * (1 - 0.22 * near[k])
        t = ITO_T * (1 + 0.75 * near[k])
        sink = -lift[k] + 0.00012
        profiles.append(np.array([(sink, -0.5 * w), (0.55 * t, -0.46 * w), (0.9 * t, -0.26 * w), (t, 0.0),
                                  (0.9 * t, 0.26 * w), (0.55 * t, 0.46 * w), (sink, 0.5 * w), (sink - 0.0002, 0.0)]))
    isl = None
    chunk = int(0.11 / step)
    for c0 in range(0, m - 1, chunk):
        c1 = min(m, c0 + chunk + 1)
        sweep(mb, P[c0:c1], (Nn[c0:c1], B[c0:c1]), profiles[c0:c1],
              cap_start='flat' if c0 == 0 else None, cap_end='flat' if c1 == m else None)
    return P, th


def stitch_chunks(obj):
    """Weld the duplicated rings where a ribbon was split into UV islands."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bm.to_mesh(obj.data)
    bm.free()


def build_ito(col):
    mb = MeshBuilder()
    for d in (1, -1):
        ito_ribbon(mb, d, Z_WRAP_TOP + 0.003, Z_WRAP_BOT - 0.003)
    obj = mb.to_object("Hilt_Ito", col)
    stitch_chunks(obj)
    return obj


def menuki_outline():
    """Tsuki-ni-kumo: a crescent moon whose outer rim breaks into cloud scallops."""
    R, r, off = 0.5, 0.40, np.array([0.19, 0.12])
    angs = np.linspace(0, 2 * math.pi, 7200, endpoint=False)
    unit = np.column_stack([np.cos(angs), np.sin(angs)])
    inside = np.linalg.norm(unit * R - off, axis=1) < r
    k_in = np.where(~inside[:-1] & inside[1:])[0][0] + 1
    k_out = np.where(inside[:-1] & ~inside[1:])[0][0]
    # outer rim: from the horn where we leave the inner disc, the long way round
    a0, a1 = angs[k_out], angs[k_in]
    if a1 < a0:
        a1 += 2 * math.pi
    oa = np.linspace(a0, a1, 90)
    t = (oa - a0) / (a1 - a0)
    cloud = smoothstep(0.38, 0.45, t) * (1 - smoothstep(0.88, 0.95, t))
    bump = 0.075 * np.abs(np.sin(4.5 * math.pi * (t - 0.38) / 0.57)) ** 0.55 * cloud
    rim = np.column_stack([np.cos(oa), np.sin(oa)]) * (R + bump)[:, None]
    ia0 = math.atan2(rim[-1, 1] - off[1], rim[-1, 0] - off[0])
    ia1 = math.atan2(rim[0, 1] - off[1], rim[0, 0] - off[0])
    if ia1 > ia0:
        ia1 -= 2 * math.pi
    ia = np.linspace(ia0, ia1, 50)
    inner = off + np.column_stack([np.cos(ia), np.sin(ia)]) * r
    return np.vstack([rim, inner[1:-1]])


def build_menuki(col):
    """Two cloud-and-flame ornaments laid onto the samegawa under the wrap."""
    outline = menuki_outline()
    outline = resample_polyline(np.vstack([outline, outline[:1]]), 73)[:-1]
    outline = outline - outline.mean(axis=0)
    L, Wd = 0.026, 0.026
    loc = np.column_stack([outline[:, 0] * L, outline[:, 1] * Wd])
    mb_all = MeshBuilder()
    for face, zc, flipv in ((0.0, ITO_Z0 - 2.5 * ITO_PITCH, -1.0), (math.pi, ITO_Z0 - 3.0 * ITO_PITCH, -1.0)):
        mb = MeshBuilder()
        extrude_shape(mb, loc, 0.0026, bevel=0.3, bevel_rings=2)
        hx, hy = handle_axes(zc)
        r_eff = 0.5 * (hx + hy)
        out = []
        for (u, v, w) in mb.verts:
            th = face + flipv * v / (hy * 1.05)
            P, Nn = handle_surface(np.array(th), np.array(zc + u), lift=w + 0.0011)
            out.append(tuple(P))
        mb.verts = out
        mb_all.merge(mb)
    return mb_all.to_object("Hilt_Menuki", col)


def build_fuchi(col):
    mb = MeshBuilder()

    def outline(g, e):
        hx, hy = handle_axes(Z_WRAP_TOP)
        return superellipse(48, hx + g, hy + g, e)

    rings = [(Z_FUCHI_TOP, 0.0010, 2.5), (Z_FUCHI_TOP - 0.0006, 0.0018, 2.5), (Z_FUCHI_TOP - 0.0022, 0.0020, 2.45),
             (Z_FUCHI_TOP - 0.0030, 0.0016, 2.4), (Z_FUCHI_TOP - 0.0042, 0.0016, 2.4), (Z_FUCHI_TOP - 0.0050, 0.0021, 2.4),
             (Z_WRAP_TOP + 0.0012, 0.0020, 2.38), (Z_WRAP_TOP + 0.0002, 0.0014, 2.36), (Z_WRAP_TOP - 0.0006, 0.0004, 2.35)]
    loft_z(mb, rings, outline, cap_top='flat', cap_bottom=None, sharp_rings=(3, 4))
    return mb.to_object("Hilt_Fuchi", col)


def build_kashira(col):
    mb = MeshBuilder()

    def outline(g, sc, e):
        hx, hy = handle_axes(Z_WRAP_BOT)
        return superellipse(48, (hx + g) * sc, (hy + g) * sc, e)

    zb = Z_WRAP_BOT
    rings = [(zb + 0.0008, 0.0004, 1.0, 2.35), (zb, 0.0015, 1.0, 2.35), (zb - 0.0012, 0.0021, 1.0, 2.4),
             (zb - 0.0045, 0.0021, 1.0, 2.45), (zb - 0.0062, 0.0017, 1.0, 2.45), (zb - 0.0072, 0.0022, 1.0, 2.5),
             (zb - 0.0098, 0.0019, 0.97, 2.5), (zb - 0.0122, 0.0010, 0.86, 2.4), (zb - 0.0140, 0.0, 0.66, 2.3),
             (zb - 0.0151, 0.0, 0.40, 2.2), (Z_KASHIRA_BOT, 0.0, 0.12, 2.1)]
    loft_z(mb, rings, outline, cap_top=None, cap_bottom='flat', sharp_rings=(4, 5))
    return mb.to_object("Hilt_Kashira", col)


# ---- cord, tassel, magatama, small ornaments ---------------------------------

MAGATAMA_OUTLINE = [(0.02, 0.50), (0.20, 0.45), (0.29, 0.28), (0.27, 0.06), (0.17, -0.16), (0.02, -0.34),
                    (-0.14, -0.47), (-0.27, -0.52), (-0.20, -0.38), (-0.10, -0.21), (-0.07, -0.04),
                    (-0.15, 0.08), (-0.26, 0.22), (-0.22, 0.40), (-0.11, 0.48)]


def place_local(mb, origin, xaxis, yaxis, zaxis):
    """Map MeshBuilder verts from a local frame into world space (in place)."""
    o, X, Y, Z = (np.asarray(v, dtype=np.float64) for v in (origin, xaxis, yaxis, zaxis))
    mb.verts = [tuple(o + X * x + Y * y + Z * z) for (x, y, z) in mb.verts]


def magatama_mesh(height, depth, bevel=0.5, detail=4, rings=3):
    mb = MeshBuilder()
    outline = catmull_rom(MAGATAMA_OUTLINE, detail, closed=True) * height
    extrude_shape(mb, outline, depth, bevel=bevel, bevel_rings=rings)
    return mb


def rope_profiles(path, radius, twist_len=0.0052, n=9, taper=None):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    profs = []
    for k, L in enumerate(cum):
        ph = np.linspace(0, 2 * math.pi, n, endpoint=False)
        tw = 2 * math.pi * L / twist_len
        r = radius * (1 + 0.17 * np.cos(3 * (ph - tw)))
        if taper is not None:
            r = r * taper[k]
        profs.append(np.column_stack([r * np.cos(ph), r * np.sin(ph)]))
    return profs


def build_sageo(col):
    """Cord knotted round a guard tongue; one end carries a magatama, one a tassel."""
    C, T, N, X, a, b = guard_tie_frame()
    rc = 0.00105
    # wraps round the tongue, advancing along it
    ph = np.linspace(-0.6, 2 * math.pi * 2.35, 150)
    loops = (C[None, :] + T[None, :] * (0.0034 * ph / (2 * math.pi) - 0.004)[:, None]
             + N[None, :] * ((a + rc * 1.05) * np.cos(ph))[:, None]
             + X[None, :] * ((b + rc * 1.05) * np.sin(ph))[:, None])
    exit_p = loops[-1]
    down = np.array([0.0, 0.0, -1.0])
    # two hanging ends with a lazy S-curve, splayed apart a little
    end_a = catmull_rom([exit_p, exit_p + N * 0.004 + down * 0.003, exit_p + N * 0.006 + X * 0.002 + down * 0.016,
                         exit_p + N * 0.0045 + X * 0.003 + down * 0.033, exit_p + N * 0.0055 + X * 0.002 + down * 0.046], 14)
    end_b = catmull_rom([exit_p, exit_p + N * 0.003 - X * 0.0015 + down * 0.002, exit_p + N * 0.0015 - X * 0.0045 + down * 0.012,
                         exit_p + N * 0.0025 - X * 0.0055 + down * 0.021], 14)
    cord = np.vstack([loops, end_a[1:]])
    mb = MeshBuilder()
    cord = resample_polyline(cord, 140)
    T2, N2, B2 = rotation_minimizing_frames(cord, X)
    sweep(mb, cord, (N2, B2), rope_profiles(cord, rc), cap_start='flat', cap_end='flat')
    end_b = resample_polyline(end_b, 28)
    T3, N3, B3 = rotation_minimizing_frames(end_b, X)
    sweep(mb, end_b, (N3, B3), rope_profiles(end_b, rc * 0.92), cap_start='flat', cap_end='flat')
    cord_obj = mb.to_object("Hilt_Sageo", col)

    # tassel head: a bound knot that the strands burst out of
    top = end_a[-1] + down * 0.0006
    head_prof = [(0.0000, 0.0004), (0.0005, 0.0017), (0.0016, 0.0027), (0.0032, 0.0030), (0.0046, 0.0025),
                 (0.0051, 0.0032), (0.0068, 0.0032), (0.0073, 0.0026), (0.0090, 0.0028), (0.0106, 0.0031),
                 (0.0112, 0.0024)]
    mbh = MeshBuilder()
    hp = np.array([top + down * h for h, _ in head_prof])
    profs = []
    for k, (h, r) in enumerate(head_prof):
        ph = np.linspace(0, 2 * math.pi, 14, endpoint=False)
        rr = r * (1 + 0.05 * np.cos(ph * 2 + k))
        profs.append(np.column_stack([rr * np.cos(ph), rr * np.sin(ph)]))
    Nh = np.tile([1.0, 0, 0], (len(hp), 1))
    Bh = np.cross(np.tile(down, (len(hp), 1)), Nh)
    sweep(mbh, hp, (Nh, Bh), profs, cap_start='flat', cap_end='flat')

    # strands: each a slim tapering cord with its own sway
    rng = np.random.default_rng(77)
    base_c = top + down * 0.0100
    for k in range(24):
        ang = 2 * math.pi * k / 24 + rng.uniform(-0.08, 0.08)
        rad0 = 0.0024 * math.sqrt(rng.uniform(0.35, 1.0))
        L = rng.uniform(0.034, 0.046)
        spread = rng.uniform(0.0016, 0.0032)
        d0 = np.array([math.cos(ang), math.sin(ang), 0.0])
        sway = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0]) * 0.0015
        hs = np.linspace(0, 1, 10)
        path = np.array([base_c + d0 * (rad0 + spread * h ** 1.4) + down * (L * h) + sway * math.sin(math.pi * h * 1.3)
                         + np.array([0.0012, 0.0006, 0.0]) * h ** 2 for h in hs])
        Ts, Ns, Bs = rotation_minimizing_frames(path, d0)
        profs = []
        for i, h in enumerate(hs):
            r = 0.00042 * (1 - 0.65 * h ** 1.5)
            ph = np.linspace(0, 2 * math.pi, 4, endpoint=False) + 0.4 * k
            profs.append(np.column_stack([r * np.cos(ph), r * np.sin(ph)]))
        sweep(mbh, path, (Ns, Bs), profs, cap_start='flat', cap_end=0.0012)
    tassel_obj = mbh.to_object("Hilt_Tassel", col)

    # magatama on the short end, its curl facing the blade flats
    hang = end_b[-1]
    mm = magatama_mesh(0.0175, 0.0066)
    rot = math.radians(18)
    zax = np.array([math.cos(rot), math.sin(rot), 0.0])
    xax = np.cross(np.array([0.0, 0.0, 1.0]), zax)
    place_local(mm, hang + down * 0.0082, xax, np.array([0.0, 0.0, 1.0]), zax)
    mag_obj = mm.to_object("Hilt_Magatama", col)
    return [cord_obj, tassel_obj, mag_obj]


def build_mekugi(col):
    mb_all = MeshBuilder()
    zc = ITO_Z0 - 0.5 * ITO_PITCH
    ring = superellipse(22, 0.0021, 0.0022, 2.0) * (1 + 0.04 * noise1d(22, 9, 5))[:, None]
    for face in (0.0, math.pi):
        mb = MeshBuilder()
        extrude_shape(mb, ring, 0.0016, bevel=0.5, bevel_rings=2)
        out = []
        for (u, v, w) in mb.verts:
            P, Nn = handle_surface(np.array(face - v / 0.0155), np.array(zc + u), lift=w + 0.0003)
            out.append(tuple(P))
        mb.verts = out
        mb_all.merge(mb)
    return mb_all.to_object("Hilt_Mekugi", col)


# ---------------------------------------------------------------------------
# hand-drawn style FX
# ---------------------------------------------------------------------------
# Everything here is a flat, two-tone shape (glow body + pale core) drawn from a
# spline spine with a brush-like width profile -- the way 2D effects animators
# draw fire -- instead of glowing cylinders / spheres / cones.  The core floats a
# hair in front of the body on both sides so the two tones never z-fight.

CORE_OFFSET = 0.00028


def flat_shape(mb, outline, origin, X, Y, Z, z=0.0, mat=0, both_sides=False):
    """Add a triangulated flat 2D outline in the plane (X, Y) at offset z along Z."""
    outline = np.asarray(outline, dtype=np.float64)
    area = np.sum(outline[:, 0] * np.roll(outline[:, 1], -1) - np.roll(outline[:, 0], -1) * outline[:, 1])
    if area < 0:
        outline = outline[::-1]
    tris = triangulate_polygon(outline)
    mn = outline.min(axis=0)
    ext = max(float(np.ptp(outline[:, 0])), float(np.ptp(outline[:, 1])), 1e-9)
    mb.cur_mat = mat
    for zz, flip in ((z, False), (-z, True)) if both_sides else ((z, False),):
        base = mb.add_verts([origin + X * p[0] + Y * p[1] + Z * zz for p in outline])
        isl = mb.new_island()
        for t in tris:
            tt = t[::-1] if flip else t
            uvq = [tuple((outline[i] - mn) / ext) for i in tt]
            mb.add_face([base + i for i in tt], uvq, uvq, isl)
    mb.cur_mat = 0


def flame_spine(h, lean, curl, n=18):
    """Spine of one tongue: rises, sways in an S, and whips over at the tip."""
    s = np.linspace(0, 1, n)
    cx = (lean * h * s ** 1.35
          + curl * 0.22 * h * np.sin(2 * math.pi * s ** 1.15) * (1 - 0.4 * s)
          + curl * 0.30 * h * smoothstep(0.62, 1.0, s) ** 2)
    cy = h * (s - 0.16 * abs(curl) * smoothstep(0.8, 1.0, s) ** 2)
    c = np.column_stack([cx, cy])
    t = np.gradient(c, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.column_stack([-t[:, 1], t[:, 0]])
    return s, c, nrm


def flame_outlines(h, b, lean, curl, asym, seed):
    """Outer tongue + inner core outlines for one hand-drawn flame lick."""
    rng = np.random.default_rng(seed)
    s, c, nrm = flame_spine(h, lean, curl)
    swell = 1 + 0.45 * np.sin(math.pi * np.clip(s / 0.55, 0, 1))
    w = 0.5 * b * (0.3 + 0.7 * smoothstep(0.0, 0.18, s)) * (1 - s) ** 1.15 * swell
    w *= 1 + 0.08 * noise1d(len(s), seed, 4)
    left = c + nrm * (w * (1 + asym))[:, None]
    right = c - nrm * (w * (1 - asym))[:, None]
    outer = np.vstack([right, left[::-1][1:-1]])
    # optional side flick that splits off the tongue, drawn as part of the outline
    if rng.random() < 0.45:
        k = int(len(s) * rng.uniform(0.3, 0.5))
        side = 1 if rng.random() < 0.5 else -1
        root = c[k]
        tipv = nrm[k] * side * 0.38 * h + np.array([lean * 0.25 * h, 0.22 * h])
        flick = catmull_rom([root + nrm[k] * side * w[k] * 0.8, root + tipv * 0.6 + nrm[k] * side * w[k] * 0.3,
                             root + tipv, root + tipv * 0.55, root + nrm[k] * side * w[k] * 0.5 + np.array([0, 0.12 * h])], 4)
        if side > 0:
            idx = len(right) + (len(s) - 2 - k)
        else:
            idx = k
        outer = np.vstack([outer[:idx + 1], flick[1:-1], outer[idx + 1:]])
    m = s <= 0.6
    sc = s[m] / 0.6
    wc = w[m] * 0.42 * (1 - sc ** 2.5) + 0.00012
    cl = c[m] + nrm[m] * (wc * (1 + asym * 0.5))[:, None] + np.array([0, 0.03 * h])
    cr = c[m] - nrm[m] * (wc * (1 - asym * 0.5))[:, None] + np.array([0, 0.03 * h])
    core = np.vstack([cr, cl[::-1][1:-1]])
    return outer, core


def spark_outline(R, stretch=1.45, n_side=6):
    """Four-point 'kira' star with concave, hand-curved sides."""
    pts = []
    for k in range(4):
        a0 = k * math.pi / 2
        a1 = a0 + math.pi / 2
        r0 = R * (stretch if k % 2 == 0 else 1.0)
        r1 = R * (stretch if (k + 1) % 2 == 0 else 1.0)
        p0 = np.array([math.sin(a0), math.cos(a0)]) * r0
        p1 = np.array([math.sin(a1), math.cos(a1)]) * r1
        ctrl = np.array([math.sin(a0 + math.pi / 4), math.cos(a0 + math.pi / 4)]) * R * 0.16
        for j in range(n_side):
            t = j / n_side
            pts.append((1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t * t * p1)
    return np.array(pts)


def blade_edge_frames(taus, f=1.0):
    bl = BLADE
    E = bl.pos(taus, np.full_like(taus, f), np.zeros_like(taus))
    dE = np.gradient(E, axis=0)
    nl = np.linalg.norm(dE, axis=1)
    for i in range(1, len(nl)):
        if nl[i] < 1e-12:
            dE[i], nl[i] = dE[i - 1], nl[i - 1]
    dE /= nl[:, None]
    X = np.tile([1.0, 0.0, 0.0], (len(taus), 1))
    O = np.cross(dE, X)
    mid = bl.pos(taus, np.full_like(taus, 0.5), np.zeros_like(taus))
    sgn = np.sign(np.einsum("ij,ij->i", O, E - mid))
    O *= np.where(sgn == 0, 1.0, sgn)[:, None]
    return E, dE, O


def field_contours(V, x0, y0, cell, t, min_area=2e-6):
    """Marching squares: closed outer loops of {V > t} on a regular grid, in metres.

    Holes are dropped (filled); separate islands come back as separate loops,
    which is what gives the fire its detached, drifting pieces.
    """
    P = np.pad(V, 1, constant_values=-1e9) - t
    ny, nx = P.shape
    ins = P > 0
    pts = {}

    def edge_pt(key):
        if key not in pts:
            kind, i, j = key
            if kind == 'h':
                a, b = P[i, j], P[i, j + 1]
                f = a / (a - b)
                pts[key] = (j + f, i)
            else:
                a, b = P[i, j], P[i + 1, j]
                f = a / (a - b)
                pts[key] = (j, i + f)
        return key

    case = (ins[:-1, :-1].astype(int) | ins[:-1, 1:] << 1 | ins[1:, 1:] << 2 | ins[1:, :-1] << 3)
    nbr = {}
    for i, j in zip(*np.nonzero((case > 0) & (case < 15))):
        c = int(case[i, j])
        bot, rgt, top, lft = ('h', i, j), ('v', i, j + 1), ('h', i + 1, j), ('v', i, j)
        e = []
        crossed = [k for k, bit_a, bit_b in ((bot, 1, 2), (rgt, 2, 4), (top, 8, 4), (lft, 1, 8))
                   if bool(c & bit_a) != bool(c & bit_b)]
        if len(crossed) == 2:
            e.append((crossed[0], crossed[1]))
        else:                                   # saddle: split by the centre value
            centre = 0.25 * (P[i, j] + P[i, j + 1] + P[i + 1, j + 1] + P[i + 1, j])
            if (centre > 0) == bool(c & 1):
                e += [(bot, rgt), (top, lft)]
            else:
                e += [(bot, lft), (top, rgt)]
        for u, v in e:
            edge_pt(u), edge_pt(v)
            nbr.setdefault(u, []).append(v)
            nbr.setdefault(v, []).append(u)
    loops, seen = [], set()
    for start in nbr:
        if start in seen:
            continue
        loop, prev, cur = [], None, start
        while cur not in seen:
            seen.add(cur)
            loop.append(pts[cur])
            nxt = [n for n in nbr[cur] if n != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
        if len(loop) < 8:
            continue
        L = np.array(loop)
        L = np.column_stack([x0 + (L[:, 0] - 1) * cell, y0 + (L[:, 1] - 1) * cell])
        area = 0.5 * np.sum(L[:, 0] * np.roll(L[:, 1], -1) - np.roll(L[:, 0], -1) * L[:, 1])
        if abs(area) < min_area:
            continue
        loops.append((area, L))
    if not loops:
        return []
    # keep outer boundaries only: a hole has the opposite winding to the largest loop
    sign = np.sign(max(loops, key=lambda t: abs(t[0]))[0])
    out = []
    for area, L in loops:
        if np.sign(area) != sign:
            continue
        seg = np.linalg.norm(np.diff(np.vstack([L, L[:1]]), axis=0), axis=1).sum()
        n = max(10, int(seg / 0.0011))
        L = resample_polyline(np.vstack([L, L[:1]]), n + 1)[:-1]
        for _ in range(2):
            L = 0.5 * L + 0.25 * (np.roll(L, 1, axis=0) + np.roll(L, -1, axis=0))
        out.append(L)
    return out


def flame_clump(rng, span, seed, scale=1.0):
    """One big, lopsided mass of fire in (along, out) space.

    A dominant tongue plus a few unequal minor ones are summed as a soft distance
    field over a low wavy bed, the field is domain-warped so every edge wobbles
    like a drawn line, then sliced at three depths into body / mid / core shapes.
    """
    nz1, nz2 = Noise2(seed), Noise2(seed + 1)
    tongues = []
    h_dom = rng.uniform(0.032, 0.068) * scale
    count = int(rng.integers(2, 6))
    for k in range(count):
        dom = k == 0
        h = h_dom if dom else h_dom * rng.uniform(0.18, 0.62)
        x = rng.uniform(0.15, 0.6) * span if dom else rng.uniform(0.0, 0.95) * span
        w = (rng.uniform(0.010, 0.017) if dom else rng.uniform(0.006, 0.012)) * (h / 0.04) ** 0.35
        lean = rng.uniform(0.25, 1.15)
        curl = float(rng.choice([-1, 1]) * rng.uniform(0.35, 1.1))
        tongues.append((x, 0.0, h, w, lean, curl))
    for _ in range(int(rng.integers(0, 3))):     # pieces torn off the top
        h = h_dom * rng.uniform(0.15, 0.3)
        tongues.append((rng.uniform(0.3, 1.1) * span, h_dom * rng.uniform(0.8, 1.15), h,
                        h * rng.uniform(0.45, 0.7), rng.uniform(0.4, 1.0), float(rng.choice([-1, 1]))))
    cell = 0.0004
    x_lo, x_hi = -0.012, span + h_dom * 1.3 + 0.01
    y_lo, y_hi = -0.0016, h_dom * 1.6 + 0.01
    xs = np.arange(x_lo, x_hi, cell)
    ys = np.arange(y_lo, y_hi, cell)
    X, Y = np.meshgrid(xs, ys)
    amp = 0.0022 + 0.05 * np.clip(Y, 0, None)          # wobble grows toward the tips
    Xw = X + amp * (nz1.fbm(X / 0.014, Y / 0.014, 3) - 0.5) * 2
    Yw = Y + 0.6 * amp * (nz2.fbm(X / 0.014 + 9, Y / 0.014, 3) - 0.5) * 2
    F = np.full(X.shape, -1.0)
    for (x0, y0, h, w, lean, curl) in tongues:
        s, c, _ = flame_spine(h, lean, curl, n=26)
        r = 0.5 * w * (1 + 0.35 * np.sin(np.pi * np.clip(s / 0.45, 0, 1))) * (1 - s) ** 0.85 + 0.0002
        for (cx, cy), rr in zip(c, r):
            F = np.maximum(F, rr - np.hypot(Xw - (x0 + cx), Yw - (y0 + cy)))
    Ft = F.copy()
    # low bed of flame hugging the edge, uneven along the clump
    bed_h = rng.uniform(0.003, 0.006) * (0.6 + 0.8 * nz1(Xw / 0.02, 3.3))
    win = smoothstep(-0.004, 0.012, Xw) * (1 - smoothstep(span - 0.006, span + 0.01, Xw))
    F = np.maximum(F, (bed_h * win - Yw) * np.where(win > 0.05, 1.0, 0.0) - (1 - win) * 0.01)
    layers = []
    for t, ysq in ((0.0, 1.0), (0.0021, 1.0), (0.0042, 1.25)):
        G = F
        if ysq != 1.0:   # core: tongues only (no bed), sitting lower in the flame than the body
            G = np.array([np.interp(ys * ysq, ys, col_) for col_ in Ft.T]).T
        layers.append(field_contours(G, x_lo, y_lo, cell, t))
    return layers


def build_fx(col, mats):
    bl = BLADE
    mb = MeshBuilder()
    rng = np.random.default_rng(2024)
    dense = np.linspace(0.0, 1.0, 4001)

    def frames(f):
        E, T, O = blade_edge_frames(dense, f)
        seg = np.linalg.norm(np.diff(E, axis=0), axis=1)
        a = np.concatenate([[0.0], np.cumsum(seg)])
        return a, E, T, O

    edge = frames(1.0)
    back = frames(0.0)
    Xw = np.array([1.0, 0.0, 0.0])

    def to3d(fr, poly, z, roll):
        a_arr, E, T, O = fr
        out = []
        for a, y in poly:
            a_c = float(np.clip(a, a_arr[0], a_arr[-1]))
            Ei = np.array([np.interp(a_c, a_arr, E[:, k]) for k in range(3)])
            Ti = np.array([np.interp(a_c, a_arr, T[:, k]) for k in range(3)])
            Oi = np.array([np.interp(a_c, a_arr, O[:, k]) for k in range(3)])
            Ti /= np.linalg.norm(Ti)
            Oi /= np.linalg.norm(Oi)
            Y = Oi * math.cos(roll) + Xw * math.sin(roll)
            Z = np.cross(Ti, Y)
            over = max(0.0, a - a_arr[-1])          # past the point: keep streaming forward
            out.append(Ei + Ti * over + Y * y + Z * z)
        return out

    def emit(fr, poly, z, mat, roll, both):
        tris = triangulate_polygon(poly)
        for zz, flip in ((z, False), (-z, True)) if both else ((z, False),):
            mb.cur_mat = mat
            base = mb.add_verts(to3d(fr, poly, zz, roll))
            isl = mb.new_island()
            for t in tris:
                tt = t[::-1] if flip else t
                mb.add_face([base + i for i in tt], None, None, isl)
        mb.cur_mat = 0

    # cutting edge: a few big, lopsided masses of fire with ragged gaps between
    # them, each turned a little differently so the fire still reads in 3D
    total = edge[0][-1]
    crng = np.random.default_rng(4711)
    a = total * 0.105
    k = 0
    while a < total * 0.93:
        span = crng.uniform(0.035, 0.095)
        scale = 0.75 + 0.5 * sstep(0.1, 0.6, a / total)
        layers = flame_clump(crng, span, seed=800 + k, scale=scale)
        roll = crng.uniform(-0.42, 0.42)
        for (polys, off, mat) in zip(layers, (0.0, 0.00018, CORE_OFFSET), (0, 1, 2)):
            for poly in polys:
                emit(edge, poly + np.array([a, 0.0]), off, mat, roll, both=off > 0)
        k += 1
        a += span * crng.uniform(0.55, 1.05) + (crng.uniform(0.012, 0.05) if crng.random() < 0.7 else 0.0)
    # back of the blade: a handful of thin, separate licks toward the point
    for j, t in enumerate((0.63, 0.7, 0.755, 0.83, 0.88, 0.94)):
        i = int(t * 4000)
        E, T, O = back[1][i], back[2][i], back[3][i]
        roll = rng.uniform(-0.45, 0.45)
        Y = O * math.cos(roll) + Xw * math.sin(roll)
        Z = np.cross(T, Y)
        fh = rng.uniform(0.007, 0.014)
        o2, c2 = flame_outlines(fh, fh * 0.42, rng.uniform(0.6, 1.0), float(rng.choice([-1, 1])) * 0.8, 0.0, 600 + j)
        P = E - Y * 0.0008
        flat_shape(mb, o2, P, T, Y, Z, 0.0, mat=0)
        o3, _ = flame_outlines(fh * 0.66, fh * 0.42 * 0.7, 0.7, 0.5, 0.0, 600 + j)
        flat_shape(mb, o3, P, T, Y, Z, 0.00018, mat=1, both_sides=True)
        flat_shape(mb, c2, P, T, Y, Z, CORE_OFFSET, mat=2, both_sides=True)

    # detached flakes drifting above the fire
    for j in range(14):
        t = rng.uniform(0.25, 0.97)
        i = int(t * 4000)
        E, T, O = edge[1][i], edge[2][i], edge[3][i]
        roll = rng.uniform(-0.5, 0.5)
        Y = O * math.cos(roll) + Xw * math.sin(roll)
        Z = np.cross(T, Y)
        fh = rng.uniform(0.004, 0.0075)
        o2, c2 = flame_outlines(fh, fh * 0.5, rng.uniform(0.5, 0.9), float(rng.choice([-1, 1])), 0.0, 900 + j)
        P = E + Y * rng.uniform(0.026, 0.045) + T * rng.uniform(0.006, 0.02)
        flat_shape(mb, o2, P, T, Y, Z, 0.0, mat=1)
        flat_shape(mb, c2 * 0.8, P + Y * fh * 0.05, T, Y, Z, CORE_OFFSET, mat=2, both_sides=True)

    # sparks
    for j in range(18):
        t = rng.uniform(0.3, 1.0)
        i = int(t * 4000)
        fr = edge if rng.random() < 0.75 else back
        E, T, O = fr[1][i], fr[2][i], fr[3][i]
        P = E + O * rng.uniform(0.015, 0.05) + T * rng.uniform(-0.01, 0.02) + np.array([rng.uniform(-0.014, 0.014), 0, 0])
        ang = rng.uniform(-0.6, 0.6)
        Zs = np.array([math.cos(ang), math.sin(ang) * 0.4, rng.uniform(-0.2, 0.2)])
        Zs /= np.linalg.norm(Zs)
        Xs = np.cross(np.array([0.0, 0.0, 1.0]), Zs)
        Xs /= np.linalg.norm(Xs)
        Ys = np.cross(Zs, Xs)
        R = rng.uniform(0.0009, 0.0024) * (1.7 if rng.random() < 0.15 else 1.0)
        flat_shape(mb, spark_outline(R), P, Xs, Ys, Zs, 0.0, mat=1)
        flat_shape(mb, spark_outline(R * 0.42, 1.3, 4), P, Xs, Ys, Zs, CORE_OFFSET, mat=2, both_sides=True)

    obj = mb.to_object("FX_SpiritFlame", col, smooth=False)
    for m in mats:
        obj.data.materials.append(m)
    obj.visible_shadow = False
    return obj


# ---------------------------------------------------------------------------
# hand-painted bake materials for the hilt
# ---------------------------------------------------------------------------
# Each part type gets a procedural "painter" shader: half-lambert light from the
# top-front, ambient occlusion pooled in crevices, bevel-detected edges picked out
# in a highlight colour, and a little brush noise.  The result is baked into an
# albedo atlas, so the painted look survives in any engine.

LIGHT_DIR = Vector((-0.30, 0.35, 0.89)).normalized()


def lin(hex_or_rgb):
    c = hexc(hex_or_rgb) if isinstance(hex_or_rgb, str) else np.asarray(hex_or_rgb)
    return (*map(float, srgb_to_linear(c)), 1.0)


class NodeKit:
    """Tiny helper to wire shader graphs from Python without index soup."""

    def __init__(self, mat):
        mat.use_nodes = True
        self.mat = mat
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.geo = self.node("ShaderNodeNewGeometry")

    def node(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def feed(self, sock, val):
        if isinstance(val, bpy.types.NodeSocket):
            self.nt.links.new(val, sock)
        elif val is not None:
            sock.default_value = val

    def math(self, op, a, b=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.feed(n.inputs[0], a)
        if b is not None:
            self.feed(n.inputs[1], b)
        return n.outputs[0]

    def add(self, a, b):
        return self.math('ADD', a, b)

    def mul(self, a, b, clamp=False):
        return self.math('MULTIPLY', a, b, clamp)

    def sub(self, a, b):
        return self.math('SUBTRACT', a, b)

    def sstep(self, e0, e1, x):
        n = self.node("ShaderNodeMapRange", interpolation_type='SMOOTHSTEP')
        self.feed(n.inputs["Value"], x)
        n.inputs["From Min"].default_value = e0
        n.inputs["From Max"].default_value = e1
        return n.outputs["Result"]

    def dot(self, a, b):
        n = self.node("ShaderNodeVectorMath", operation='DOT_PRODUCT')
        self.feed(n.inputs[0], a)
        self.feed(n.inputs[1], b)
        return n.outputs["Value"]

    def mix(self, a, b, fac):
        n = self.node("ShaderNodeMix", data_type='RGBA', clamp_factor=True)
        self.feed(n.inputs[0], fac)
        self.feed(n.inputs[6], a)
        self.feed(n.inputs[7], b)
        return n.outputs[2]

    def mulc(self, col, fac):
        n = self.node("ShaderNodeMix", data_type='RGBA', blend_type='MULTIPLY', clamp_factor=True)
        n.inputs[0].default_value = 1.0
        self.feed(n.inputs[6], col)
        self.feed(n.inputs[7], fac)
        return n.outputs[2]

    def gray(self, v):
        n = self.node("ShaderNodeCombineColor")
        for i in range(3):
            self.feed(n.inputs[i], v)
        return n.outputs[0]

    def rgb(self, r, g, b):
        n = self.node("ShaderNodeCombineColor")
        for i, v in enumerate((r, g, b)):
            self.feed(n.inputs[i], v)
        return n.outputs[0]

    def ramp(self, fac, stops, interp='EASE'):
        n = self.node("ShaderNodeValToRGB")
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) > 1:
            cr.elements.remove(cr.elements[-1])
        cr.elements[0].position = stops[0][0]
        cr.elements[0].color = lin(stops[0][1])
        for pos, col in stops[1:]:
            e = cr.elements.new(pos)
            e.color = lin(col)
        self.feed(n.inputs[0], fac)
        return n.outputs[0]

    def uv(self, name):
        n = self.node("ShaderNodeUVMap", uv_map=name)
        sep = self.node("ShaderNodeSeparateXYZ")
        self.nt.links.new(n.outputs[0], sep.inputs[0])
        return sep.outputs[0], sep.outputs[1]

    def noise(self, scale, detail=3.0, vec=None):
        n = self.node("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        self.feed(n.inputs["Vector"], vec if vec is not None else self.geo.outputs["Position"])
        return n.outputs[0]

    def voronoi(self, scale, vec=None):
        n = self.node("ShaderNodeTexVoronoi")
        n.inputs["Scale"].default_value = scale
        self.feed(n.inputs["Vector"], vec if vec is not None else self.geo.outputs["Position"])
        return n.outputs["Distance"]

    def painter_basics(self, ao_dist=0.005, bevel_r=0.0007, edge_gain=16.0):
        ao = self.node("ShaderNodeAmbientOcclusion", samples=16, only_local=False)
        ao.inputs["Distance"].default_value = ao_dist
        bev = self.node("ShaderNodeBevel", samples=8)
        bev.inputs["Radius"].default_value = bevel_r
        N = self.geo.outputs["Normal"]
        edge = self.math('MULTIPLY', self.sub(1.0, self.dot(bev.outputs[0], N)), edge_gain, clamp=True)
        hl = self.add(0.5, self.mul(self.dot(N, tuple(LIGHT_DIR)), 0.5))
        return ao.outputs["AO"], edge, hl

    def finish(self, albedo, emission=None, orm=(1.0, 0.5, 0.0), height=None):
        """Outputs: an Emission shader we re-point per bake pass + a BSDF carrying the bump."""
        self.sockets = {"albedo": albedo,
                        "emission": emission if emission is not None else self.gray(0.0),
                        "orm": self.rgb(*orm) if isinstance(orm, tuple) else orm}
        bsdf = self.node("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
        if height is not None:
            bump = self.node("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 1.0
            bump.inputs["Distance"].default_value = 1.0
            self.feed(bump.inputs["Height"], height)
            self.nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
        self.emit = self.node("ShaderNodeEmission")
        addn = self.node("ShaderNodeAddShader")
        self.nt.links.new(bsdf.outputs[0], addn.inputs[0])
        self.nt.links.new(self.emit.outputs[0], addn.inputs[1])
        out = self.node("ShaderNodeOutputMaterial")
        self.nt.links.new(addn.outputs[0], out.inputs["Surface"])
        self.bake_img = self.node("ShaderNodeTexImage")
        self.mat["painter"] = True
        self.point("albedo")
        return self

    def point(self, which):
        self.nt.links.new(self.sockets[which], self.emit.inputs["Color"])


def silver_painter(name, file_marks=False, tone=1.0):
    k = NodeKit(bpy.data.materials.new(name))
    ao, edge, hl = k.painter_basics()
    nz = k.noise(160.0, 4.0)
    v = k.add(k.add(k.mul(hl, 0.55), k.mul(k.math('POWER', ao, 1.2), 0.45)), k.mul(k.sub(nz, 0.5), 0.14))
    v = k.mul(v, tone)
    col = k.ramp(v, [(0.0, PAL["silver_cav"]), (0.3, PAL["silver_lo"]), (0.62, PAL["silver_mid"]), (0.9, PAL["silver_hi"])])
    cav = k.mul(k.math('POWER', k.sub(1.0, ao), 2.0), 0.65)
    col = k.mix(col, lin(PAL["silver_cav"]), cav)
    col = k.mix(col, lin(PAL["silver_hi"]), k.mul(edge, 0.8))
    height = None
    emission = None
    if file_marks:
        u, v_ = k.uv("UVPhys")
        stripes = k.math('SINE', k.mul(k.add(k.mul(u, 0.75), v_), 2 * math.pi / 0.0009))
        col = k.mulc(col, k.gray(k.add(0.93, k.mul(stripes, 0.07))))
        height = k.mul(stripes, 0.000018)
    return k.finish(col, emission, (1.0, 0.34, 0.15), height)


def enamel_painter(name, lo, mid, hi, stripe, glow=None, glow_amt=0.0):
    """Glossy lacquer like the drawing: flat painted tone, AO in the steps, diagonal sheen bands."""
    k = NodeKit(bpy.data.materials.new(name))
    ao, edge, hl = k.painter_basics(ao_dist=0.004)
    nz = k.noise(90.0, 3.0)
    v = k.add(k.add(k.mul(hl, 0.45), k.mul(ao, 0.55)), k.mul(k.sub(nz, 0.5), 0.1))
    col = k.ramp(v, [(0.0, lo), (0.55, mid), (0.95, hi)])
    pos = k.node("ShaderNodeSeparateXYZ")
    k.nt.links.new(k.geo.outputs["Position"], pos.inputs[0])
    diag = k.add(pos.outputs[1], pos.outputs[2])
    bands = k.math('SINE', k.mul(diag, 2 * math.pi / 0.011))
    bands2 = k.math('SINE', k.mul(diag, 2 * math.pi / 0.0043))
    sheen = k.mul(k.sstep(0.55, 0.7, bands), k.sstep(-0.2, 0.3, bands2))
    facing = k.sstep(0.6, 0.85, k.math('ABSOLUTE', k.dot(k.geo.outputs["Normal"], (1.0, 0.0, 0.0))))
    col = k.mix(col, lin(stripe), k.mul(sheen, k.mul(facing, 0.55)))
    col = k.mix(col, lin(hi), k.mul(edge, 0.35))
    emi = None
    if glow is not None:
        emi = k.mix(k.gray(0.0), lin(glow), k.add(glow_amt, k.mul(nz, glow_amt * 0.5)))
    return k.finish(col, emi, (1.0, 0.3, 0.0), None)


def iron_painter(name, wave_v=None):
    k = NodeKit(bpy.data.materials.new(name))
    ao, edge, hl = k.painter_basics(ao_dist=0.004)
    hammer = k.voronoi(1 / 0.0018)
    v = k.add(k.add(k.mul(hl, 0.5), k.mul(ao, 0.5)), k.mul(k.sub(hammer, 0.4), 0.08))
    col = k.ramp(v, [(0.0, "#0d0b12"), (0.45, PAL["iron_dark"]), (0.8, PAL["iron_mid"]), (1.0, PAL["silver_lo"])])
    col = k.mix(col, lin(PAL["silver_mid"]), k.mul(edge, 0.7))
    height = k.mul(k.sstep(0.0, 0.5, hammer), 0.00003)
    if wave_v is not None:
        u, v_ = k.uv("UVPhys")
        centre = k.add(wave_v, k.mul(k.math('SINE', k.mul(u, 2 * math.pi / 0.0052)), 0.00045))
        d = k.math('ABSOLUTE', k.sub(v_, centre))
        line = k.sub(1.0, k.sstep(0.00014, 0.00028, d))
        col = k.mix(col, lin(PAL["silver_hi"]), line)
        height = k.add(height, k.mul(line, 0.00005))
    return k.finish(col, None, (1.0, 0.48, 0.1), height)


def same_painter(name):
    k = NodeKit(bpy.data.materials.new(name))
    ao, edge, hl = k.painter_basics(ao_dist=0.003)
    d = k.voronoi(1 / 0.00115)
    nod = k.sub(1.0, k.sstep(0.12, 0.48, d))
    v = k.add(k.mul(hl, 0.35), k.mul(ao, 0.65))
    col = k.ramp(v, [(0.0, "#07060a"), (0.55, PAL["same_dark"]), (1.0, "#2a2433")])
    col = k.mix(col, lin(PAL["same_nod"]), k.mul(k.mul(nod, hl), 0.75))
    return k.finish(col, None, (1.0, 0.62, 0.0), k.mul(nod, 0.00011))


def silk_painter(name, twill=0.0011):
    k = NodeKit(bpy.data.materials.new(name))
    ao, edge, hl = k.painter_basics(ao_dist=0.0035, bevel_r=0.0003, edge_gain=8.0)
    u, v_ = k.uv("UVPhys")
    tw = k.math('SINE', k.mul(k.add(u, v_), 2 * math.pi / twill))
    tw2 = k.math('SINE', k.mul(k.sub(u, v_), 2 * math.pi / (twill * 1.7)))
    v = k.add(k.add(k.mul(hl, 0.6), k.mul(k.math('POWER', ao, 1.5), 0.4)), k.mul(k.add(tw, k.mul(tw2, 0.4)), 0.05))
    col = k.ramp(v, [(0.0, "#3d0820"), (0.3, PAL["ito_lo"]), (0.6, PAL["ito_mid"]), (0.92, PAL["ito_hi"])])
    col = k.mix(col, lin("#ffc2d6"), k.mul(edge, 0.25))
    return k.finish(col, None, (1.0, 0.82, 0.0), k.mul(k.add(tw, k.mul(tw2, 0.4)), 0.00002))


def jade_painter(name):
    k = NodeKit(bpy.data.materials.new(name))
    ao, edge, hl = k.painter_basics(ao_dist=0.003)
    cloud = k.noise(260.0, 5.0)
    col = k.ramp(k.add(k.mul(hl, 0.8), k.mul(ao, 0.2)), [(0.0, PAL["jade_lo"]), (0.45, PAL["jade_mid"]), (0.9, PAL["jade_hi"])])
    col = k.mix(col, lin(PAL["jade_hi"]), k.mul(k.math('POWER', cloud, 3.0), 0.7))
    col = k.mix(col, lin("#fff0f6"), k.mul(edge, 0.6))
    emi = k.mix(k.gray(0.0), lin(PAL["jade_mid"]), k.add(0.3, k.mul(cloud, 0.55)))
    return k.finish(col, emi, (1.0, 0.22, 0.0), None)


# ---------------------------------------------------------------------------
# UV atlas packing (shelf packer over the physical-scale UV islands)
# ---------------------------------------------------------------------------

def pack_atlas(objs, res, margin_px=6):
    islands = []
    for obj in objs:
        me = obj.data
        n = len(me.loops)
        uv = np.empty(n * 2, np.float32)
        me.uv_layers["UVPhys"].uv.foreach_get("vector", uv)
        uv = uv.reshape(n, 2).astype(np.float64)
        isl = np.empty(len(me.polygons), np.int32)
        me.attributes["island"].data.foreach_get("value", isl)
        loop_isl = np.repeat(isl, [p.loop_total for p in me.polygons])
        for i in np.unique(isl):
            m = np.where(loop_isl == i)[0]
            lo = uv[m].min(axis=0)
            hi = uv[m].max(axis=0)
            islands.append(dict(obj=obj, loops=m, lo=lo, size=np.maximum(hi - lo, 1e-5)))
    total_area = sum(float(i["size"][0] * i["size"][1]) for i in islands)

    def try_pack(side):
        """Skyline bottom-left packing; each island may rotate 90 degrees."""
        m = margin_px / res * side
        items = sorted(islands, key=lambda it: -max(it["size"]) * min(it["size"]))
        sky = [[0.0, 0.0, side]]            # segments: x, y, width
        place = []
        for it in items:
            best = None
            for rot in (False, True):
                w, h = (it["size"][::-1] if rot else it["size"]) + 2 * m
                for i in range(len(sky)):
                    x = sky[i][0]
                    if x + w > side + 1e-12:
                        break
                    y, span, j = 0.0, 0.0, i
                    while span < w - 1e-12:
                        y = max(y, sky[j][1])
                        span += sky[j][2]
                        j += 1
                        if j >= len(sky) and span < w - 1e-12:
                            y = math.inf
                            break
                    if y + h > side + 1e-12:
                        continue
                    if best is None or (y + h, x) < (best[0] + best[2], best[1]):
                        best = (y, x, h, w, rot)
            if best is None:
                return None
            y, x, h, w, rot = best
            place.append((x + m, y + m, rot, it))
            new, x1 = [], x + w
            for sx, sy, sw in sky:
                if sx + sw <= x + 1e-12 or sx >= x1 - 1e-12:
                    new.append([sx, sy, sw])
                    continue
                if sx < x:
                    new.append([sx, sy, x - sx])
                if sx + sw > x1:
                    new.append([x1, sy, sx + sw - x1])
            new.append([x, y + h, w])
            new.sort(key=lambda g: g[0])
            merged = []
            for g in new:
                if merged and abs(merged[-1][1] - g[1]) < 1e-12:
                    merged[-1][2] += g[2]
                else:
                    merged.append(g)
            sky = merged
        return place

    lo, hi = math.sqrt(total_area), math.sqrt(total_area) * 3
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if try_pack(mid):
            hi = mid
        else:
            lo = mid
    side = hi
    place = try_pack(side)
    per_obj = {}
    for x, y, rot, it in place:
        obj = it["obj"]
        if obj.name not in per_obj:
            n = len(obj.data.loops)
            buf = np.empty(n * 2, np.float32)
            obj.data.uv_layers["UVPhys"].uv.foreach_get("vector", buf)
            per_obj[obj.name] = (buf.reshape(n, 2).astype(np.float64), np.zeros((n, 2)))
        src, dst = per_obj[obj.name]
        loc = src[it["loops"]] - it["lo"]
        if rot:
            loc = np.column_stack([it["size"][1] - loc[:, 1], loc[:, 0]])
        dst[it["loops"]] = (loc + np.array([x, y])) / side
    for obj in objs:
        _, dst = per_obj[obj.name]
        layer = obj.data.uv_layers.get("UVMap") or obj.data.uv_layers.new(name="UVMap")
        layer.uv.foreach_set("vector", dst.astype(np.float32).ravel())
        obj.data.uv_layers.active = layer
        layer.active_render = True
    used = sum(float(i["size"][0] * i["size"][1]) for i in islands) / side ** 2
    print(f"[atlas] {len(islands)} islands, {side * 1000:.1f} mm square, {used * 100:.0f}% filled, "
          f"{res / side / 1000:.1f} px/mm")
    return side


def bake_hilt(objs, res, samples):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.render.bake.margin = 8
    scene.render.bake.margin_type = 'EXTEND'
    imgs = {
        "albedo": bpy.data.images.new("katana_hilt_albedo", res, res, alpha=False),
        "emission": bpy.data.images.new("katana_hilt_emission", res, res, alpha=False),
        "orm": bpy.data.images.new("katana_hilt_orm", res, res, alpha=False),
        "normal": bpy.data.images.new("katana_hilt_normal", res, res, alpha=False),
    }
    imgs["orm"].colorspace_settings.name = 'Non-Color'
    imgs["normal"].colorspace_settings.name = 'Non-Color'
    mats = {m for o in objs for m in o.data.materials if m and m.get("painter")}
    kits = {m.name: m for m in mats}
    hidden = []
    for o in scene.objects:
        if o.type == 'MESH' and o not in objs and o.name.startswith("FX"):
            o.hide_render = True
            hidden.append(o)
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    passes = (("albedo", 'EMIT', samples), ("emission", 'EMIT', 1), ("orm", 'EMIT', 1), ("normal", 'NORMAL', 4))
    for key, btype, spp in passes:
        scene.cycles.samples = spp
        for m in mats:
            nt = m.node_tree
            img_node = next(n for n in nt.nodes if n.type == 'TEX_IMAGE' and n.image is None or
                            (n.type == 'TEX_IMAGE' and n.image and n.image.name.startswith("katana_hilt_")))
            img_node.image = imgs[key]
            nt.nodes.active = img_node
            if btype == 'EMIT':
                emit = next(n for n in nt.nodes if n.type == 'EMISSION')
                sock = PAINTER_SOCKETS[m.name][key]
                nt.links.new(sock, emit.inputs["Color"])
        print(f"[bake] {key} ...")
        bpy.ops.object.bake(type=btype, margin=8, use_clear=True, uv_layer="UVMap", normal_space='TANGENT')
    for o in hidden:
        o.hide_render = False
    os.makedirs(TEX_DIR, exist_ok=True)
    paths = {}
    for key, img in imgs.items():
        path = os.path.join(TEX_DIR, f"katana_hilt_{key}.png")
        img.filepath_raw = path
        img.file_format = 'PNG'
        img.save()
        paths[key] = path
    return paths


PAINTER_SOCKETS = {}


def painter(kind, name, *args, **kw):
    fn = {"silver": silver_painter, "iron": iron_painter, "same": same_painter, "silk": silk_painter,
          "jade": jade_painter, "enamel": enamel_painter}[kind]
    kit = fn(name, *args, **kw)
    PAINTER_SOCKETS[kit.mat.name] = kit.sockets
    return kit.mat


# ---------------------------------------------------------------------------
# ink outline (inverted hull with hand-varied line weight)
# ---------------------------------------------------------------------------

def ink_material():
    """Preview-safe ink: Cycles ignores backface culling, so the shader hides backfaces itself.
    swap_ink_for_export() replaces it with a plain culled material for glTF."""
    mat = bpy.data.materials.new("M_Katana_Ink")
    mat.use_backface_culling = True
    k = NodeKit(mat)
    ink = k.node("ShaderNodeBsdfPrincipled")
    ink.inputs["Base Color"].default_value = lin(PAL["ink"])
    ink.inputs["Roughness"].default_value = 0.9
    tr = k.node("ShaderNodeBsdfTransparent")
    mix = k.node("ShaderNodeMixShader")
    k.nt.links.new(k.geo.outputs["Backfacing"], mix.inputs[0])
    k.nt.links.new(ink.outputs[0], mix.inputs[1])
    k.nt.links.new(tr.outputs[0], mix.inputs[2])
    out = k.node("ShaderNodeOutputMaterial")
    k.nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def swap_ink_for_export(mat):
    k = NodeKit(mat)
    bsdf = k.node("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = lin(PAL["ink"])
    bsdf.inputs["Roughness"].default_value = 0.9
    out = k.node("ShaderNodeOutputMaterial")
    k.nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    mat.use_backface_culling = True


def build_outline(objs, col, mat, base=0.00045):
    nz = Noise2(99)
    bm_all = bmesh.new()
    for o in objs:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.transform(o.matrix_world)
        bm.normal_update()
        size = max(o.dimensions) if max(o.dimensions) > 0 else 0.01
        w = base * min(1.0, 0.45 + size / 0.06)
        for v in bm.verts:
            p = v.co
            var = 0.6 + 0.8 * float(nz.fbm(np.array(p.x * 90 + p.z * 40), np.array(p.y * 90 - p.z * 60), octaves=2))
            v.co = p + v.normal * w * var
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        tmp = bpy.data.meshes.new("tmp")
        bm.to_mesh(tmp)
        bm.free()
        bm_all.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    bmesh.ops.triangulate(bm_all, faces=bm_all.faces)
    me = bpy.data.meshes.new("Katana_InkOutline")
    bm_all.to_mesh(me)
    bm_all.free()
    tmp_obj = bpy.data.objects.new("tmp_outline", me)
    col.objects.link(tmp_obj)
    dec = tmp_obj.modifiers.new("dec", 'DECIMATE')
    dec.ratio = 0.5
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp_obj.evaluated_get(dg))
    me.name = "Katana_InkOutline"
    old_me = tmp_obj.data
    bpy.data.objects.remove(tmp_obj)
    bpy.data.meshes.remove(old_me)
    for layer in list(me.uv_layers):
        me.uv_layers.remove(layer)
    me.shade_smooth()
    obj = bpy.data.objects.new("Katana_InkOutline", me)
    col.objects.link(obj)
    me.materials.append(mat)
    obj.visible_shadow = False
    return obj


# ---------------------------------------------------------------------------
# assembly / export / previews
# ---------------------------------------------------------------------------

def copy_uv(obj, src, dst="UVMap"):
    me = obj.data
    buf = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers[src].uv.foreach_get("vector", buf)
    layer = me.uv_layers.get(dst) or me.uv_layers.new(name=dst)
    layer.uv.foreach_set("vector", buf)
    me.uv_layers.active = layer
    layer.active_render = True


def build_hilt(col):
    parts = {
        "guard": build_flame_guard(col),
        "seppa_l": plate(col, "Hilt_SeppaLower", -0.0190, Z_FUCHI_TOP, 0.0147, 0.0192),
        "tsuka": build_tsuka_core(col),
        "ito": build_ito(col),
        "menuki": build_menuki(col),
        "fuchi": build_fuchi(col),
        "kashira": build_kashira(col),
        "mekugi": build_mekugi(col),
    }
    cord, tassel, mag = build_sageo(col)
    parts.update(sageo=cord, tassel=tassel, magatama=mag)
    mats = {
        "guard": [painter("enamel", "P_GuardOuter", "#b0154f", "#ff4f9a", "#ff8cbc", "#ffb3d2"),
                  painter("enamel", "P_GuardMid", "#3a0822", "#7e1549", "#a72a63", "#c5498a"),
                  painter("enamel", "P_GuardInner", "#e07aa6", "#ffc2da", "#ffe8f1", "#ffffff", "#ff7fb2", 0.35),
                  painter("enamel", "P_GuardSlit", "#ffd6e8", "#fff1f7", "#ffffff", "#ffffff", "#ffe3ef", 0.9),
                  painter("enamel", "P_GuardInk", "#0b050d", PAL["ink"], "#2c1630", "#2c1630")],
        "seppa_l": painter("silver", "P_Seppa", tone=0.9),
        "menuki": painter("silver", "P_Menuki", tone=1.05),
        "tsuka": painter("same", "P_Samegawa"),
        "ito": painter("silk", "P_Ito"),
        "sageo": painter("silk", "P_Sageo", twill=0.0016),
        "fuchi": painter("iron", "P_Fuchi", wave_v=0.0036),
        "kashira": painter("iron", "P_Kashira", wave_v=0.0075),
        "mekugi": painter("iron", "P_Mekugi"),
        "magatama": painter("jade", "P_Jade"),
    }
    mats["tassel"] = mats["sageo"]
    for key, obj in parts.items():
        for m in (mats[key] if isinstance(mats[key], list) else [mats[key]]):
            obj.data.materials.append(m)
    return parts


def shift_to_grip(objs):
    M = Matrix.Translation((0.0, 0.0, -GRIP_CENTRE))
    for o in objs:
        o.data.transform(M)
        o.data.update()


def assign_final_hilt_material(parts, paths):
    mat = pbr_material("M_Katana_Hilt", load_image(paths["albedo"]), load_image(paths["emission"]),
                       load_image(paths["normal"], non_color=True), load_image(paths["orm"], non_color=True),
                       emission_strength=3.0)
    for o in parts.values():
        o.data.materials.clear()
        o.data.materials.append(mat)
    return mat


def strip_helper_data(objs):
    for o in objs:
        me = o.data
        for name in ("UVPhys", "UVParam"):
            layer = me.uv_layers.get(name)
            if layer:
                me.uv_layers.remove(layer)
        for name in ("island",):
            a = me.attributes.get(name)
            if a:
                me.attributes.remove(a)


def export_glb(objs, path):
    """Join the hilt into one mesh and export the game asset."""
    bpy.ops.object.select_all(action='DESELECT')
    dupes = []
    for o in objs:
        d = o.copy()
        d.data = o.data.copy()
        bpy.context.scene.collection.objects.link(d)
        d.parent = None
        d.matrix_world = o.matrix_world
        dupes.append(d)
    strip_helper_data(dupes)
    clean = {}
    for o, d in zip(objs, dupes):
        clean[d] = o.name
        o.name = o.name + "__src"
        o.data.name = o.data.name + "__src"
    hilt = [d for d in dupes if clean[d].startswith("Hilt_")]
    others = [d for d in dupes if not clean[d].startswith("Hilt_")]
    for d in others:
        d.name = clean[d]
        d.data.name = clean[d]
    bpy.ops.object.select_all(action='DESELECT')
    for d in hilt:
        d.select_set(True)
    bpy.context.view_layer.objects.active = hilt[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = "Katana_Hilt"
    joined.data.name = "Katana_Hilt"
    final = [joined] + others
    scene_root = bpy.data.objects.get("MysticKatana")
    if scene_root:
        scene_root.name = "MysticKatana__src"
    root = bpy.data.objects.new("MysticKatana", None)
    bpy.context.scene.collection.objects.link(root)
    for d in final:
        d.parent = root
    bpy.ops.object.select_all(action='DESELECT')
    root.select_set(True)
    for d in final:
        d.select_set(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_apply=True,
                              export_yup=True, export_tangents=True, export_image_format='AUTO')
    stats = {d.name: sum(len(p.vertices) - 2 for p in d.data.polygons) for d in final}
    for d in final + [root]:
        me = d.data
        bpy.data.objects.remove(d)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    for o in objs:
        o.name = o.name[:-5]
        o.data.name = o.data.name[:-5]
    if scene_root:
        scene_root.name = "MysticKatana"
    return stats


def render_previews(col):
    os.makedirs(RENDER_DIR, exist_ok=True)
    setup_world_and_lights(col)
    o = -GRIP_CENTRE
    spp = 24 if FAST else 96
    shots = [
        ("hero", (1.62, 0.36, 0.30 + o), (0, 0.012, 0.25 + o), 50, None, (1100, 2000)),
        ("front", (1.6, 0.0, 0.25 + o), (0, 0.012, 0.25 + o), 50, 1.12, (900, 2000)),
        ("hilt", (0.36, 0.25, 0.0 + o), (0, 0.02, -0.06 + o), 50, None, (1400, 1400)),
        ("tip", (0.22, -0.16, 0.66 + o), (0, 0.0, 0.6 + o), 50, None, (1400, 1400)),
        ("back", (-1.5, -0.6, 0.35 + o), (0, 0.0, 0.25 + o), 50, None, (1100, 2000)),
    ]
    for name, loc, tgt, lens, ortho, res in shots:
        cam = add_camera(col, "Cam_" + name, loc, tgt, lens=lens, ortho=ortho)
        render_view(cam, os.path.join(RENDER_DIR, f"katana_{name}.png"), res=res if not FAST else (res[0] // 2, res[1] // 2),
                    samples=spp)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    col = get_collection("MysticKatana")

    blade = build_blade(col)
    paths = paint_blade((256, 2048) if FAST else (512, 4096))
    copy_uv(blade, "UVPhys")
    blade.data.materials.append(blade_material(paths))

    parts = build_hilt(col)
    fx = build_fx(col, [flat_emissive_material("M_FX_Outer", "#d81b6a", 2.0),
                        flat_emissive_material("M_FX_Mid", "#ff4f9a", 2.4),
                        flat_emissive_material("M_FX_Core", "#ffe3ef", 2.6)])
    shift_to_grip([blade, fx] + list(parts.values()))

    hilt_objs = list(parts.values())
    res = 1024 if FAST else 2048
    pack_atlas(hilt_objs, res)
    if not NO_BAKE:
        hpaths = bake_hilt(hilt_objs, res, 8 if FAST else 48)
    else:
        hpaths = {k: os.path.join(TEX_DIR, f"katana_hilt_{k}.png") for k in ("albedo", "emission", "orm", "normal")}
    hilt_mat = assign_final_hilt_material(parts, hpaths)

    ink = ink_material()
    outline_src = [parts[k] for k in ("guard", "seppa_l", "fuchi", "kashira", "ito", "sageo", "magatama")]
    outline = build_outline(outline_src, col, ink)

    root = bpy.data.objects.new("MysticKatana", None)
    col.objects.link(root)
    for o in [blade, fx, outline] + hilt_objs:
        o.parent = root

    if not NO_RENDER:
        render_previews(col)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    blend_path = os.path.join(HERE, "mystic_katana.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True, relative_remap=True)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_mainfile(compress=True)

    swap_ink_for_export(ink)
    stats = export_glb([blade, fx, outline] + hilt_objs, os.path.join(EXPORT_DIR, "mystic_katana.glb"))
    print("[export] triangles:", stats, "total:", sum(stats.values()))


if __name__ == "__main__":
    main()
