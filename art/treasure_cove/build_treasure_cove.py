"""
Lost Pirate Cove -- SeekThenHide
================================

A small treasure-hunt map: a tropical island about 55 m across.  A ruined temple
crowns the central hill, a pirate ship lies wrecked in the southern cove, a wooden
dock reaches out east where the players land, a rock arch hides a little cave on
the north-west hill, a pirate camp sits in the west palm grove and a tiny palm
islet off the north-east coast can be reached by wading over a sandbar.

Eight treasure spawns (``TreasureSpawn_##`` empties, each holding a chest) and six
player spawns (``PlayerSpawn_##``) are exported as glTF nodes with ``kind`` extras.
Everything is low-poly and flat shaded: the terrain is coloured per face through a
colour attribute, the props use flat-colour materials.

Reuses a few helpers from ../mystic_katana/build_mystic_katana.py.

    blender --background --python build_treasure_cove.py
    blender --background --python build_treasure_cove.py -- --fast
    blender --background --python build_treasure_cove.py -- --no-render
"""

import math
import os
import sys

import bpy  # noqa: E402  (bpy must load before bmesh when run as a module)
import bmesh
import numpy as np
from mathutils import Euler, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mystic_katana"))
import build_mystic_katana as K  # noqa: E402

EXPORT_DIR = os.path.join(HERE, "export")
RENDER_DIR = os.path.join(HERE, "renders")

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
FAST = "--fast" in ARGS
NO_RENDER = "--no-render" in ARGS

hexc, srgb_to_linear = K.hexc, K.srgb_to_linear
RNG = np.random.default_rng(1721)


# ---------------------------------------------------------------------------
# small maths helpers
# ---------------------------------------------------------------------------

def ss(e0, e1, x):
    """smoothstep that also works with e0 > e1 (falling edge)."""
    t = np.clip((np.asarray(x, dtype=float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


_PERM = np.random.default_rng(77).permutation(256)
_PERM = np.concatenate([_PERM, _PERM])
_VALS = np.random.default_rng(78).random(256)


def vnoise(x, y):
    """2D value noise in [-1, 1]."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - xi, y - yi
    u, v = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)

    def h(ix, iy):
        return _VALS[_PERM[(_PERM[ix & 255] + iy) & 255]]

    a, b = h(xi, yi), h(xi + 1, yi)
    c, d = h(xi, yi + 1), h(xi + 1, yi + 1)
    return 2 * ((a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v) - 1


def fbm(x, y, octaves=4, seed=0):
    x, y = np.asarray(x, float) + 31.7 * seed, np.asarray(y, float) - 17.3 * seed
    out, amp, f, norm = 0.0, 1.0, 1.0, 0.0
    for _ in range(octaves):
        out = out + amp * vnoise(x * f, y * f)
        norm += amp
        amp *= 0.5
        f *= 2.03
    return out / norm


def gauss(x, y, c, sigma):
    return np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * sigma ** 2))


# ---------------------------------------------------------------------------
# island shape (metres, Blender Z up, +Y north)
# ---------------------------------------------------------------------------

COVE_TH = math.radians(-100)    # the shipwreck bay bites into the south coast
TEMPLE = np.array([0.0, 6.0])   # central hill
ARCH = np.array([-15.0, 13.0])  # north-west hill with the rock arch
CAMP = np.array([-17.0, -6.0])  # west palm grove
ISLET_TH = math.radians(42)


def island_radius(th):
    return (26 + 2.5 * np.sin(2 * th + 1.0) + 1.6 * np.sin(3 * th + 0.4) + 0.9 * np.sin(5 * th + 2.1)
            - 5.5 * np.exp(-(wrap(th - COVE_TH) / 0.32) ** 2))


def _dir(th):
    return np.array([math.cos(th), math.sin(th)])


ISLET = _dir(ISLET_TH) * (island_radius(ISLET_TH) + 7.5)
SANDBAR = (_dir(ISLET_TH) * island_radius(ISLET_TH) * 0.93, ISLET)


def seg_dist(x, y, a, b):
    ab = b - a
    t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / (ab @ ab), 0, 1)
    return np.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1]))


def _ground_raw(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, th = np.hypot(x, y), np.arctan2(y, x)
    R = island_radius(th) + 1.2 * fbm(x / 14, y / 14, 3, seed=1)
    shore = 1 - r / R
    land = -0.3 + 0.9 * ss(0, 0.09, shore) + 1.2 * ss(0.09, 0.45, shore)
    sea = -0.3 - 3.2 * ss(0, 0.28, -shore)
    h = np.where(shore >= 0, land, sea)
    inland = ss(0.06, 0.32, shore)
    h = h + inland * (5.6 * gauss(x, y, TEMPLE, 7.5) + 3.2 * gauss(x, y, ARCH, 4.5)
                      + 0.5 * fbm(x / 8, y / 8, 3, seed=2))
    h = h + (0.10 * fbm(x / 2.2, y / 2.2, 2, seed=3) + 0.05 * fbm(x / 0.9, y / 0.9, 2, seed=13)) * ss(-0.05, 0.05, shore)
    # the islet and the sandbar that leads to it (knee-deep at -0.45 m)
    e = np.exp(-((x - ISLET[0]) ** 2 + (y - ISLET[1]) ** 2) / (2 * 2.4 ** 2))
    h = np.maximum(h, -3.5 + 4.7 * e ** 0.35)  # falls away to the deep sea floor all round
    h = np.maximum(h, -0.45 - 0.08 * seg_dist(x, y, *SANDBAR) ** 2)
    return h


TEMPLE_Z = float(_ground_raw(*TEMPLE)) - 1.0  # cut the hilltop rather than raise a mesa
CAMP_Z = float(_ground_raw(*CAMP))


def ground(x, y):
    """Final terrain height; also used to set every prop on the ground."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = _ground_raw(x, y)
    w = ss(9.5, 5.0, np.maximum(abs(x - TEMPLE[0]), abs(y - TEMPLE[1])))
    h = h * (1 - w) + TEMPLE_Z * w
    w = ss(5.0, 3.0, np.hypot(x - CAMP[0], y - CAMP[1]))
    h = h * (1 - w) + CAMP_Z * w
    return np.maximum(h, -3.5)


def gz(p):
    return float(ground(p[0], p[1]))


def shore_point(th, inland=0.0):
    """First sea-level crossing walking out along angle th, moved `inland` metres back."""
    d = _dir(th)
    rs = np.linspace(5, 45, 801)
    hs = ground(d[0] * rs, d[1] * rs)
    i = int(np.argmax(hs < 0))
    return d * (rs[i] - inland)


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

DOCK_TH = math.radians(-8)
DOCK_DIR = _dir(DOCK_TH)
DOCK_BASE = shore_point(DOCK_TH, inland=2.5)
DOCK_LEN = 11.0
DECK_Z = 0.95

WRECK_DIR = _dir(COVE_TH)
WRECK_POS = shore_point(COVE_TH, inland=0.6)

# more structures (built in build_lighthouse() and friends)
LIGHTHOUSE = shore_point(math.radians(18), inland=3.8)
HUT = shore_point(math.radians(-28), inland=1.4)
RUIN = np.array([9.0, -8.2])
WELL = np.array([12.3, -4.1])
GRAVEYARD = np.array([14.4, 5.4])
TOWER = np.array([-11.5, -12.0])
SHED = np.array([-23.4, 3.6])
STRUCTURES = [(LIGHTHOUSE, 3.4), (HUT, 4.6), (RUIN, 4.3), (WELL, 1.6), (GRAVEYARD, 3.8), (TOWER, 2.4), (SHED, 2.6)]

PATHS = [
    [DOCK_BASE - DOCK_DIR * 0.5, DOCK_BASE - DOCK_DIR * 6 + [0, 1.5], np.array([8.0, -0.5]),
     np.array([3.0, -0.5]), TEMPLE + [0, -5.8]],
    [WRECK_POS - WRECK_DIR * 4.5, np.array([-3.5, -10.0]), np.array([-1.5, -2.5]), TEMPLE + [0, -5.8]],
    [CAMP + [3.0, 0.5], np.array([-10.0, -2.5]), np.array([-4.0, -1.6]), TEMPLE + [-0.8, -5.8]],
    [TEMPLE + [-5.2, 1.5], np.array([-10.0, 9.0]), ARCH + [1.2, -2.6]],
    [TEMPLE + [5.2, 2.0], np.array([11.0, 10.5]), shore_point(ISLET_TH, inland=1.5)],
    [DOCK_BASE - DOCK_DIR * 3.5 + [-0.6, 2.2], np.array([21.0, 3.0]), LIGHTHOUSE + [-1.8, -2.6]],
    [np.array([12.0, -1.0]), WELL + [-0.3, -1.4], RUIN + [1.4, 2.6]],
    [RUIN + [4.4, -0.2], np.array([14.2, -9.2]), HUT + [-2.6, 1.6]],
    [np.array([12.3, 9.6]), GRAVEYARD + [-0.9, 2.6]],
    [np.array([-3.4, -9.6]), np.array([-7.5, -11.2]), TOWER + [1.6, 1.2]],
]


def path_dist(x, y):
    d = np.full(np.shape(x), 1e9)
    for pl in PATHS:
        for a, b in zip(pl[:-1], pl[1:]):
            d = np.minimum(d, seg_dist(x, y, np.asarray(a, float), np.asarray(b, float)))
    return d


# (name, xy, description) -- every chest sits at its spawn
TREASURE_SPAWNS = [
    ("Temple altar", TEMPLE + [0, 0.7]),
    ("Shipwreck deck", None),            # filled in by build_shipwreck()
    ("Under the rock arch", ARCH.copy()),
    ("Pirate camp tent", None),          # filled in by build_camp()
    ("Palm islet", ISLET + [0.9, -0.4]),
    ("South-east beach, under the X", shore_point(math.radians(-48), inland=3.2)),
    ("Behind the temple, fallen pillars", TEMPLE + [3.6, 5.6]),
    ("North beach boulders", shore_point(math.radians(102), inland=3.4)),
    ("Lighthouse gallery", None),        # filled in by build_lighthouse() and the builders below
    ("Fisherman's hut", None),
    ("Ruined house", None),
    ("Graveyard, by the open grave", None),
    ("Watchtower platform", None),
    ("Smuggler's shed", None),
]
TREASURE_Z = {}  # index -> z when the chest stands on a prop instead of the ground


# ---------------------------------------------------------------------------
# geometry builder
# ---------------------------------------------------------------------------

class Geo:
    """Verts / faces / material names; emitted as one flat-shaded object."""

    def __init__(self):
        self.v, self.f, self.m = [], [], []

    def add(self, verts, faces, mat):
        base = len(self.v)
        self.v.extend(np.asarray(verts, float).tolist())
        self.f.extend([[i + base for i in f] for f in faces])
        self.m.extend([mat] * len(faces))
        return self

    def add_multi(self, verts, faces, mats):
        """Like add(), with one material per face."""
        base = len(self.v)
        self.v.extend(np.asarray(verts, float).tolist())
        self.f.extend([[i + base for i in f] for f in faces])
        self.m.extend(mats)
        return self

    def merge(self, other, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0):
        v = xf(other.v, loc, rot, scale)
        base = len(self.v)
        self.v.extend(v.tolist())
        self.f.extend([[i + base for i in f] for f in other.f])
        self.m.extend(other.m)
        return self


def xf(verts, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0):
    v = np.asarray(verts, float).reshape(-1, 3) * np.asarray(scale, float)
    M = np.array(Euler(rot, 'XYZ').to_matrix())
    return v @ M.T + np.asarray(loc, float)


_BEVEL_TOPO = []


def _bevel_corners(h, b):
    """24 verts: for each corner (x fastest), its point on the x, y and z face."""
    v = []
    for ci in range(8):
        sgn = np.array([(ci & 1) * 2 - 1, ((ci >> 1) & 1) * 2 - 1, ((ci >> 2) & 1) * 2 - 1], float)
        base = sgn * (h - b)
        for a in range(3):
            q = base.copy()
            q[a] = sgn[a] * h[a]
            v.append(q)
    return np.array(v)


def _bevel_topology():
    """Faces of a chamfered box: 6 faces, 12 edge strips, 8 corner triangles, wound outwards."""
    if _BEVEL_TOPO:
        return _BEVEL_TOPO

    def corner(sg):
        return (sg[0] > 0) + 2 * (sg[1] > 0) + 4 * (sg[2] > 0)

    faces = []
    ring = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for a in range(3):
        b_, c_ = [x for x in range(3) if x != a]
        for sa in (-1, 1):
            f = []
            for sb, sc in ring:
                sg = [0, 0, 0]
                sg[a], sg[b_], sg[c_] = sa, sb, sc
                f.append(3 * corner(sg) + a)
            faces.append(f)
        for sb in (-1, 1):
            for sc in (-1, 1):
                sg1, sg2 = [0, 0, 0], [0, 0, 0]
                sg1[a], sg1[b_], sg1[c_] = -1, sb, sc
                sg2[a], sg2[b_], sg2[c_] = 1, sb, sc
                c1, c2 = corner(sg1), corner(sg2)
                faces.append([3 * c1 + b_, 3 * c2 + b_, 3 * c2 + c_, 3 * c1 + c_])
    faces += [[3 * ci, 3 * ci + 1, 3 * ci + 2] for ci in range(8)]
    v = _bevel_corners(np.array([1.0, 1.3, 1.6]), 0.2)
    for f in faces:
        pts = v[f]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        _BEVEL_TOPO.append(f if n @ pts.mean(axis=0) > 0 else f[::-1])
    return _BEVEL_TOPO


def box(g, center, size, mat, rot=(0, 0, 0), bevel=True):
    """Box with lightly chamfered edges (plain when it is too thin to show a bevel)."""
    h = np.asarray(size, float) / 2
    b = min(0.03, 0.16 * float(h.min()) * 2)
    if not bevel or b < 0.006:
        sx, sy, sz = h
        v = np.array([[x, y, z] for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)])
        f = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
        return g.add(xf(v, center, rot), f, mat)
    return g.add(xf(_bevel_corners(h, b), center, rot), _bevel_topology(), mat)


def _frame(axis):
    a = axis / np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


def loft(g, rings, mat, cap_start=True, cap_end=True):
    """rings: list of (n, 3) arrays with the same n; quads between consecutive rings."""
    n = len(rings[0])
    v = np.concatenate(rings)
    f = []
    for i in range(len(rings) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            f.append([a, b, b + n, a + n])
    if cap_start:
        f.append(list(range(n))[::-1])
    if cap_end:
        f.append([(len(rings) - 1) * n + j for j in range(n)])
    return g.add(v, f, mat)


def tube(g, p0, p1, r0, r1=None, mat="M_Wood", n=8, phase=0.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    r1 = r0 if r1 is None else r1
    _, u, w = _frame(p1 - p0)
    ang = phase + np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w
    if r1 <= 1e-6:  # cone
        g.add(np.vstack([p0 + r0 * ring, p1]), [[j, (j + 1) % n, n] for j in range(n)] + [list(range(n))[::-1]], mat)
        return g
    return loft(g, [p0 + r0 * ring, p1 + r1 * ring], mat)


def beam(g, p0, p1, w, h, mat):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a = p1 - p0
    L = np.linalg.norm(a)
    a /= L
    side = np.cross([0, 0, 1.0], a)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0, 0])
    side /= np.linalg.norm(side)
    up = np.cross(a, side)
    # same corner order as box(): k (up) fastest, then s (side), then t (along)
    v = [p0 + a * t + side * s + up * k for t in (0, L) for s in (-w / 2, w / 2) for k in (-h / 2, h / 2)]
    f = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    return g.add(np.array(v), f, mat)


_ICO = {}


def ico(subdiv=1):
    if subdiv not in _ICO:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
        v = np.array([tuple(x.co) for x in bm.verts])
        f = [[x.index for x in face.verts] for face in bm.faces]
        bm.free()
        _ICO[subdiv] = (v, f)
    return _ICO[subdiv]


def rock(g, center, size, mat="M_Rock", seed=0, yaw=0.0, subdiv=1, rough=0.22, sink=0.25):
    v, f = ico(subdiv)
    rng = np.random.default_rng(seed)
    k = 1 + rough * (2 * rng.random(len(v)) - 1)
    k = k * (1 + 0.15 * fbm(v[:, 0] * 1.3 + seed, v[:, 1] * 1.3 + v[:, 2], 2))
    p = v * k[:, None] * np.asarray(size, float)
    p[:, 2] = np.maximum(p[:, 2], -size[2] * (1 - sink))
    return g.add(xf(p, center, (0, 0, yaw)), f, mat)


def noise3(p, seed=0):
    """Cheap 3D fbm from three 2D slices, in about [-1, 1]."""
    p = np.asarray(p, float)
    return (fbm(p[:, 0] + seed, p[:, 1], 3, seed=seed) + fbm(p[:, 1], p[:, 2] - seed, 3, seed=seed + 1) +
            fbm(p[:, 2], p[:, 0] + 2 * seed, 3, seed=seed + 2)) / 1.6


def boulder(g, center, size, seed=0, yaw=0.0, mats=("M_Rock", "M_RockDark", "M_RockLight"), moss=0.0,
            subdiv=2, strata=0.35, sink=0.3, tilt=0.15):
    """A natural rock: lumpy at two scales, faceted into strata, speckled face by face in a few
    tones, and (with `moss` > 0) green on the faces that look up."""
    v, f = ico(subdiv)
    rng = np.random.default_rng(seed)
    q = v * 1.1 + rng.uniform(-50, 50, 3)
    k = 1 + 0.28 * noise3(q, seed % 97) + 0.09 * noise3(q * 3.1, seed % 89 + 5)
    p = v * k[:, None]
    if strata:  # terrace the height into layers, like bedded stone
        p[:, 2] = p[:, 2] * (1 - strata) + strata * np.round(p[:, 2] * 3.5) / 3.5
    p = p * np.asarray(size, float)
    p[:, 2] = np.maximum(p[:, 2], -size[2] * (1 - sink))
    rot = (rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), yaw)
    p = xf(p, center, rot)
    F = np.array(f)
    P = p[F]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nz = nrm[:, 2] / (np.linalg.norm(nrm, axis=1) + 1e-9)
    tone = noise3(P.mean(axis=1) * 1.7, seed % 83) + rng.uniform(-0.25, 0.25, len(F))
    out = [mats[0] if t < 0.05 else (mats[1] if t < 0.45 else mats[min(2, len(mats) - 1)]) for t in tone]
    if moss > 0:
        mn = noise3(P.mean(axis=1) * 0.9, seed % 71 + 3)
        out = ["M_Moss" if (z > 0.55 and m_ > 0.35 - moss) else o for z, m_, o in zip(nz, mn, out)]
    return g.add_multi(p, f, out)


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

MATS = {}


def _principled(mat):
    if not mat.node_tree:
        mat.use_nodes = True
    return next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')


def flat_mat(name, color, rough=0.85, metal=0.0, emit=None, emit_strength=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    bsdf = _principled(mat)
    bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(hexc(color)), 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*srgb_to_linear(hexc(emit)), 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit_strength
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        for attr, val in (("surface_render_method", 'BLENDED'), ("blend_method", 'BLEND')):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, val)
                except (TypeError, ValueError):
                    pass
    mat.diffuse_color = (*srgb_to_linear(hexc(color)), alpha)
    MATS[name] = mat
    return mat


def terrain_mat():
    mat = bpy.data.materials.new("M_Terrain")
    bsdf = _principled(mat)
    vc = mat.node_tree.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Col"
    mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.92
    MATS["M_Terrain"] = mat
    return mat


def make_materials():
    terrain_mat()
    flat_mat("M_Water", "#2fb3c9", rough=0.06, alpha=0.62)
    flat_mat("M_Seabed", "#1f5d78", rough=1.0)
    flat_mat("M_Stone", "#c4b79c", rough=0.9)
    flat_mat("M_StoneDark", "#958a74", rough=0.9)
    flat_mat("M_Moss", "#6f9444", rough=0.95)
    flat_mat("M_Rock", "#8a8276", rough=0.95)
    flat_mat("M_RockDark", "#6b645b", rough=0.95)
    flat_mat("M_WoodDark", "#5b3d27", rough=0.8)
    flat_mat("M_Wood", "#8a5d38", rough=0.8)
    flat_mat("M_WoodLight", "#b88c5a", rough=0.8)
    flat_mat("M_Cloth", "#eadfc4", rough=0.95)
    flat_mat("M_ClothRed", "#b1382c", rough=0.95)
    flat_mat("M_Iron", "#3d3b42", rough=0.45, metal=0.7)
    flat_mat("M_Gold", "#ffc53d", rough=0.25, metal=1.0, emit="#ffb321", emit_strength=0.35)
    flat_mat("M_PalmTrunk", "#a07a52", rough=0.9)
    flat_mat("M_PalmLeaf", "#4f9d3b", rough=0.8)
    flat_mat("M_PalmLeafDark", "#3c8231", rough=0.8)
    flat_mat("M_Coconut", "#6b4a2b", rough=0.8)
    flat_mat("M_Bush", "#4b8a35", rough=0.9)
    flat_mat("M_BushLight", "#68a843", rough=0.9)
    flat_mat("M_PaintRed", "#c63a2a", rough=0.8)
    flat_mat("M_Terracotta", "#c0673e", rough=0.9)
    flat_mat("M_Bone", "#eee4c8", rough=0.8)
    flat_mat("M_Dark", "#1c1712", rough=1.0)
    flat_mat("M_Glass", "#5fb488", rough=0.08, alpha=0.75)
    flat_mat("M_GlassBrown", "#7a4a1e", rough=0.12)
    flat_mat("M_Paper", "#f1e5c3", rough=0.95)
    flat_mat("M_Leather", "#6e3f22", rough=0.8)
    flat_mat("M_Sack", "#c9a974", rough=1.0)
    flat_mat("M_Rope", "#c2a067", rough=1.0)
    flat_mat("M_Shell", "#f4d3c0", rough=0.6)
    flat_mat("M_Starfish", "#ec7a3a", rough=0.8)
    flat_mat("M_Crab", "#d9482f", rough=0.6)
    flat_mat("M_Steel", "#b9bcc4", rough=0.3, metal=0.9)
    flat_mat("M_Driftwood", "#c9b9a0", rough=0.95)
    flat_mat("M_Mushroom", "#d8483a", rough=0.7)
    flat_mat("M_MushroomStem", "#f2e8d2", rough=0.9)
    flat_mat("M_Wax", "#f4ecd6", rough=0.6)
    flat_mat("M_FlagBlack", "#1d1b1e", rough=0.95)
    flat_mat("M_GemBlue", "#3aa6ff", rough=0.1, emit="#3aa6ff", emit_strength=1.5)
    flat_mat("M_GemRed", "#e2304d", rough=0.1, emit="#e2304d", emit_strength=1.5)
    flat_mat("M_LanternGlow", "#ffd27a", emit="#ffb340", emit_strength=4.0)
    flat_mat("M_CatalogFloor", "#eadcbd", rough=1.0)
    flat_mat("M_Thatch", "#c9a24f", rough=1.0)
    flat_mat("M_LighthouseWhite", "#f1ede2", rough=0.8)
    flat_mat("M_LighthouseRed", "#c23b2e", rough=0.8)
    flat_mat("M_DeadWood", "#6d655c", rough=1.0)
    flat_mat("M_WellWater", "#1e3e4a", rough=0.05)
    flat_mat("M_Dirt", "#8a6a44", rough=1.0)
    flat_mat("M_PalmTrunkTop", "#6f7d3a", rough=0.9)
    flat_mat("M_PalmDead", "#a5844b", rough=0.9)
    flat_mat("M_BushDark", "#3b7330", rough=0.9)
    flat_mat("M_Grass", "#5f9e3e", rough=0.9)
    flat_mat("M_GrassLight", "#8cc55a", rough=0.9)
    flat_mat("M_DuneGrass", "#b4b569", rough=0.9)
    flat_mat("M_FlowerCentre", "#f5c531", rough=0.8)
    flat_mat("M_FlowerRed", "#e0353b", rough=0.8)
    flat_mat("M_FlowerPink", "#f08cb5", rough=0.8)
    flat_mat("M_FlowerWhite", "#f7f5ee", rough=0.8)
    flat_mat("M_FlowerYellow", "#f7d23e", rough=0.8)
    flat_mat("M_FlowerBlue", "#6f86e6", rough=0.8)
    flat_mat("M_ThatchDark", "#a9853c", rough=1.0)
    flat_mat("M_Mortar", "#a59a84", rough=1.0)
    flat_mat("M_StoneLight", "#d6ccb4", rough=0.9)
    flat_mat("M_Brass", "#c99a3a", rough=0.3, metal=0.9)
    flat_mat("M_RockLight", "#a39b8d", rough=0.95)
    flat_mat("M_Bark", "#4e3a30", rough=0.9)
    flat_mat("M_Fern", "#3f8a3a", rough=0.85)
    flat_mat("M_FernLight", "#69ad48", rough=0.85)
    flat_mat("M_Reed", "#6e8f3e", rough=0.9)
    flat_mat("M_ReedLight", "#a9b766", rough=0.9)
    flat_mat("M_Cattail", "#6b4426", rough=0.9)
    flat_mat("M_Seagrass", "#4f8a5a", rough=0.8)
    flat_mat("M_Flame", "#ff7a1f", emit="#ff7a1f", emit_strength=8.0)
    flat_mat("M_FlameCore", "#ffe08a", emit="#ffe08a", emit_strength=12.0)
    for m in MATS.values():
        m.use_backface_culling = False


FLAT_MATS = {"M_Rock", "M_RockDark", "M_RockLight", "M_Water", "M_Seabed", "M_Terrain"}
SMOOTH_ANGLE = math.radians(65)


def emit(name, g, col, loc=(0, 0, 0)):
    """One object from a Geo: curved parts shade smooth, edges sharper than SMOOTH_ANGLE stay crisp."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in g.v], [], g.f)
    used = list(dict.fromkeys(g.m))
    for m in used:
        mesh.materials.append(MATS[m])
    idx = {m: i for i, m in enumerate(used)}
    mesh.polygons.foreach_set("material_index", [idx[m] for m in g.m])
    mesh.validate()
    mesh.set_sharp_from_angle(angle=SMOOTH_ANGLE)
    flat = {i for m, i in idx.items() if m in FLAT_MATS}
    mesh.polygons.foreach_set("use_smooth", [p.material_index not in flat for p in mesh.polygons])
    mesh.update()
    ob = bpy.data.objects.new(name, mesh)
    ob.location = loc
    col.objects.link(ob)
    return ob


def marker(name, col, loc, kind, yaw=0.0, size=0.6, **props):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = 'SINGLE_ARROW' if kind == "player_spawn" else 'PLAIN_AXES'
    ob.empty_display_size = size
    ob.location = loc
    ob.rotation_euler = (0, 0, yaw)
    ob["kind"] = kind
    for k, v in props.items():
        ob[k] = v
    col.objects.link(ob)
    return ob


# ---------------------------------------------------------------------------
# terrain
# ---------------------------------------------------------------------------

TERRAIN_EXTENT = 46.0


def build_terrain(col, n):
    xs = np.linspace(-TERRAIN_EXTENT, TERRAIN_EXTENT, n + 1)
    X, Y = np.meshgrid(xs, xs)
    Z = ground(X, Y)
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    faces = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i
            faces += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    F = np.array(faces)
    P = verts[F]
    cen = P.mean(axis=1)
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nz = nrm[:, 2] / np.linalg.norm(nrm, axis=1)
    hc = cen[:, 2]
    cx, cy = cen[:, 0], cen[:, 1]
    var = fbm(cx / 3.0, cy / 3.0, 2, seed=5)
    jit = 1 + 0.045 * (2 * RNG.random(len(F)) - 1)

    def c(h):
        return hexc(h)[None, :]

    var2 = fbm(cx / 1.1, cy / 1.1, 2, seed=15)
    var3 = fbm(cx / 7.0, cy / 7.0, 2, seed=16)

    def mix(base, col_, w):
        w = np.clip(w, 0, 1)[:, None]
        return base * (1 - w) + col_ * w

    deep = ss(-0.2, -3.0, hc)[:, None]
    col_rgb = c("#dcc48c") * (1 - deep) + c("#2a6d86") * deep
    col_rgb = mix(col_rgb, c("#4f8a7a"), (var2 > 0.35) * ss(-0.4, -1.2, hc) * 0.6)  # weed on the sea floor
    sand_line = 0.95 + 0.25 * var
    r_ = np.hypot(cx, cy)
    ripple = 0.5 + 0.5 * np.sin(r_ * 5.0 + 3.0 * var2)
    sand = c("#eed79c") * (0.96 + 0.05 * ripple[:, None]) * (0.97 + 0.04 * var2[:, None])
    sand = mix(sand, c("#d4b67c"), ss(0.45, 0.08, hc))                   # wet sand by the water
    sand = mix(sand, c("#f6f0e2"), ss(0.13, 0.04, hc) * ss(-0.08, 0.0, hc))  # foam line
    sand = mix(sand, c("#cdb688"), (var2 < -0.45) * 0.6)                  # darker patches
    col_rgb = np.where((hc >= -0.15)[:, None], sand, col_rgb)
    grass = c("#7dba50") * (0.5 + 0.5 * var[:, None]) + c("#5b9c3d") * (0.5 - 0.5 * var[:, None])
    grass = grass * (0.92 + 0.1 * var2[:, None])
    grass = mix(grass, c("#a9b860"), ss(0.35, 0.65, var3) * 0.7)          # sun-dried patches
    grass = mix(grass, c("#4a8a36"), ss(-0.3, -0.6, var3) * 0.7)          # lush hollows
    grass = mix(grass, c("#c9c08a"), ss(sand_line + 0.25, sand_line, hc))  # sandy fringe
    on_grass = hc >= sand_line
    col_rgb = np.where(on_grass[:, None], grass, col_rgb)
    band = np.floor(hc * 2.6 + 0.5 * var2) % 3  # bedding in the exposed rock
    rock_c = np.where((band == 0)[:, None], c("#938a7c"), np.where((band == 1)[:, None], c("#857c70"), c("#a39a8b")))
    col_rgb = np.where((on_grass & (nz < 0.86))[:, None], rock_c, col_rgb)
    col_rgb = np.where((on_grass & (nz < 0.74))[:, None], rock_c * 0.8, col_rgb)
    col_rgb = mix(col_rgb, c("#6b8f4a"), (on_grass & (nz < 0.86) & (nz > 0.78)) * 0.5)  # grass on the ledges
    pd = path_dist(cx, cy)
    dirt = ss(1.25, 0.85, pd + 0.25 * var + 0.15 * var2) * (hc > 0.25)
    dirt_c = c("#ae8853") * (0.93 + 0.08 * var2[:, None])
    dirt_c = mix(dirt_c, c("#8c6a40"), ss(0.75, 0.3, pd) * 0.45)  # worn centre of the path
    col_rgb = col_rgb * (1 - dirt[:, None]) + dirt_c * dirt[:, None]
    # temple courtyard and camp: trodden ground
    yard = (np.maximum(abs(cx - TEMPLE[0]), abs(cy - TEMPLE[1])) < 5.4) | (np.hypot(cx - CAMP[0], cy - CAMP[1]) < 3.4)
    col_rgb = np.where(yard[:, None], c("#b89462") * (0.94 + 0.06 * var[:, None]), col_rgb)
    col_rgb = np.clip(col_rgb * jit[:, None], 0, 1)

    mesh = bpy.data.meshes.new("Terrain")
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.materials.append(MATS["M_Terrain"])
    attr = mesh.color_attributes.new("Col", 'BYTE_COLOR', 'CORNER')
    rgba = np.concatenate([np.repeat(col_rgb, 3, axis=0), np.ones((len(F) * 3, 1))], axis=1)
    attr.data.foreach_set("color_srgb", rgba.astype(np.float32).ravel())
    mesh.polygons.foreach_set("use_smooth", (nz >= 0.8).tolist())  # soft ground, faceted rock faces
    mesh.update()
    ob = bpy.data.objects.new("Terrain", mesh)
    col.objects.link(ob)

    g = Geo()
    s = 120.0
    g.add([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], [[0, 1, 2, 3]], "M_Water")
    emit("Water", g, col)
    g = Geo()
    g.add([[-s, -s, -3.55], [s, -s, -3.55], [s, s, -3.55], [-s, s, -3.55]], [[0, 1, 2, 3]], "M_Seabed")
    emit("Seabed", g, col)
    return ob


# ---------------------------------------------------------------------------
# props
# ---------------------------------------------------------------------------

def chest_geo(open_lid=False):
    g = Geo()
    L, W, H = 0.9, 0.56, 0.42
    box(g, (0, 0, H / 2), (L - 0.03, W - 0.03, H - 0.01), "M_WoodDark", bevel=False)
    for k in range(3):  # body planks with a hairline gap between them
        z = (k + 0.5) * H / 3
        box(g, (0, 0, z), (L, W, H / 3 - 0.012), "M_Wood")
    for x in (-L / 2 + 0.09, L / 2 - 0.09):
        box(g, (x, 0, H / 2), (0.06, W + 0.025, H + 0.01), "M_Iron")
        for z in np.linspace(0.06, H - 0.06, 4):
            for sy in (-1, 1):
                ball(g, (x, sy * (W / 2 + 0.015), z), 0.013, "M_Gold", subdiv=1)
    for sx in (-1, 1):  # corner brackets
        for sy in (-1, 1):
            box(g, (sx * (L / 2 - 0.005), sy * (W / 2 - 0.035), H / 2), (0.025, 0.08, H + 0.01), "M_Iron")
            box(g, (sx * (L / 2 - 0.035), sy * (W / 2 - 0.005), H / 2), (0.08, 0.025, H + 0.01), "M_Iron")
        handle = Geo()
        ring_path(handle, 0.05, 0.01, "M_Iron", n=10, m=4)
        g.merge(handle, loc=(sx * (L / 2 + 0.02), 0, H * 0.62), rot=(0, math.pi / 2, 0))
    # rounded lid: half cylinder along x, in planks
    n = 9
    ang = np.linspace(0, np.pi, n)
    prof = np.stack([np.zeros(n), (W / 2) * np.cos(ang), 0.22 * np.sin(ang)], axis=1)
    lid = Geo()
    for j in range(n - 1):
        seg = prof[j:j + 2]
        ring_a = np.vstack([seg + [-L / 2, 0, 0], (seg * [1, 0.96, 0.96]) [::-1] + [-L / 2, 0, 0]])
        loft(lid, [ring_a, ring_a + [L, 0, 0]], "M_WoodDark" if j % 2 else "M_Wood")
    for x in (-L / 2 + 0.09, L / 2 - 0.09):
        band = [prof * [1, 1.05, 1.07] + [xx, 0, 0] for xx in (x - 0.03, x + 0.03)]
        loft(lid, band, "M_Iron", cap_start=False, cap_end=False)
        for j in range(1, n - 1):
            ball(lid, prof[j] * [1, 1.08, 1.1] + [x, 0, 0], 0.013, "M_Gold", subdiv=1)
    for x in (-0.25, 0.25):  # hinges at the back
        box(lid, (x, W / 2 + 0.01, 0.0), (0.1, 0.03, 0.06), "M_Iron")
    if open_lid:
        g.merge(lid, loc=(0, 0, H), rot=(math.radians(-105), 0, 0), scale=1.0)
    else:
        g.merge(lid, loc=(0, 0, H))
    box(g, (0, -W / 2 - 0.02, H - 0.03), (0.13, 0.04, 0.17), "M_Gold")
    box(g, (0, -W / 2 - 0.042, H - 0.05), (0.025, 0.01, 0.05), "M_Dark", bevel=False)
    ball(g, (0, -W / 2 - 0.045, H - 0.02), 0.014, "M_Dark", subdiv=1)
    return g


def coin_pile(g, center, r, seed):
    rng = np.random.default_rng(seed)
    tube(g, center, np.asarray(center) + [0, 0, r * 0.55], r, r * 0.25, "M_Gold", n=9)
    for _ in range(9):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(r * 0.9, r * 1.8)
        p = np.asarray(center) + [d * math.cos(a), d * math.sin(a), 0.005]
        tube(g, p, p + [0, 0, 0.02], 0.045, mat="M_Gold", n=7)


def palm_geo(seed, height):
    rng = np.random.default_rng(seed)
    g = Geo()
    lean = rng.uniform(0, 2 * np.pi)
    ld = np.array([math.cos(lean), math.sin(lean), 0])
    bend = rng.uniform(0.6, 1.6)
    rings = []
    nr, ns = 22, 10
    a = np.linspace(0, 2 * np.pi, ns, endpoint=False)
    for i, t in enumerate(np.linspace(0, 1, nr)):
        c = ld * bend * t * t + [0, 0, height * t - 0.25]
        r = 0.2 + 0.06 * (1 - t) + 0.12 * max(0.0, 0.12 - t) / 0.12   # flared foot
        r *= 1.09 if i % 2 else 0.97                                  # leaf-scar rings
        rings.append(c + r * np.stack([np.cos(a + 0.21 * i), np.sin(a + 0.21 * i), np.zeros(ns)], axis=1))
    loft(g, rings, "M_PalmTrunk")
    top = ld * bend + [0, 0, height - 0.25]
    ball(g, top + [0, 0, 0.12], 0.2, "M_PalmTrunkTop", subdiv=2, squash=(1, 1, 1.3))
    nf = int(rng.integers(10, 13))
    for k in range(nf + 2):
        dead = k >= nf
        yaw = 2 * np.pi * k / nf + rng.uniform(-0.2, 0.2) if not dead else rng.uniform(0, 2 * np.pi)
        d = np.array([math.cos(yaw), math.sin(yaw), 0])
        side = np.array([-d[1], d[0], 0])
        L = rng.uniform(2.3, 3.1) * (0.8 if dead else 1.0)
        lift = rng.uniform(0.25, 0.65) if not dead else -0.6
        mat = "M_PalmDead" if dead else ("M_PalmLeaf" if k % 2 else "M_PalmLeafDark")
        st = np.linspace(0, 1, 13)
        spine = [top + [0, 0, 0.15] + d * L * s_ * (0.85 if dead else 1) +
                 [0, 0, L * (lift * s_ - (1.4 if dead else 0.85) * s_ * s_)] for s_ in st]
        for i in range(len(st) - 1):
            tube(g, spine[i], spine[i + 1], 0.035 * (1 - st[i]) + 0.01, 0.035 * (1 - st[i + 1]) + 0.01,
                 "M_PalmTrunk" if dead else mat, n=4)
        for i in range(1, len(st) - 1):
            p = spine[i]
            fwd = spine[i + 1] - spine[i - 1]
            fwd /= np.linalg.norm(fwd)
            ln = 0.15 + 0.7 * math.sin(math.pi * st[i]) ** 0.6
            for sd in (-1, 1):
                droop = 0.35 + 0.25 * st[i] + (0.5 if dead else 0.0)
                mid = p + side * sd * ln * 0.5 + fwd * ln * 0.22 + [0, 0, -ln * droop * 0.3]
                tip = p + side * sd * ln + fwd * ln * 0.38 + [0, 0, -ln * droop]
                w = 0.06
                v = [p - fwd * w, p + fwd * w, mid + fwd * w * 0.7, mid - fwd * w * 0.7, tip]
                g.add(v, [[0, 1, 2, 3], [3, 2, 4]], mat)
    for k in range(int(rng.integers(4, 7))):
        a_ = 2 * np.pi * k / 5 + rng.uniform(0, 0.6)
        ball(g, top + [0.2 * math.cos(a_), 0.2 * math.sin(a_), -0.12 - 0.08 * (k % 2)], 0.12, "M_Coconut", subdiv=2)
    return g


def bush_geo(seed, flowers=None):
    """A dark leafy core covered in individual leaves of three greens, a few twigs showing."""
    rng = np.random.default_rng(seed)
    g = Geo()
    lobes = []
    for k in range(int(rng.integers(5, 8))):
        a = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(0.05, 0.55)
        s = rng.uniform(0.3, 0.5)
        c = np.array([d * math.cos(a), d * math.sin(a), s * 0.55 + rng.uniform(0, 0.25)])
        rock(g, c, (s * 0.85, s * 0.85, s * 0.7), "M_BushDark", seed=seed * 7 + k, rough=0.12, sink=0.6, subdiv=1)
        lobes.append((c, s))
    for k in range(5):  # twigs at the foot
        c, s = lobes[k % len(lobes)]
        tube(g, (c[0] * 0.3, c[1] * 0.3, 0), c * [1.05, 1.05, 0.6], 0.025, 0.012, "M_Bark", n=4)
    leaf_mats = ("M_Bush", "M_BushLight", "M_BushDark", "M_Bush")
    for c, s in lobes:
        for j in range(26):
            a = rng.uniform(0, 2 * np.pi)
            e = rng.uniform(-0.2, 1.3)
            n_ = np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
            base = c + n_ * s * rng.uniform(0.75, 0.95)
            t1 = np.cross(n_, [0, 0, 1.0])
            t1 = t1 / (np.linalg.norm(t1) + 1e-6)
            t2 = np.cross(n_, t1)
            ang = rng.uniform(0, 2 * np.pi)
            ax = t1 * math.cos(ang) + t2 * math.sin(ang)
            sd = np.cross(n_, ax)
            L, W = rng.uniform(0.13, 0.2), rng.uniform(0.05, 0.075)
            tip = base + ax * L + n_ * 0.05
            g.add([base, base + ax * L * 0.45 + sd * W + n_ * 0.03, tip, base + ax * L * 0.45 - sd * W + n_ * 0.03],
                  [[0, 1, 2, 3]], leaf_mats[int(rng.integers(4))])
    if flowers:
        for k in range(18):
            c, s = lobes[k % len(lobes)]
            a = rng.uniform(0, 2 * np.pi)
            e = rng.uniform(0.1, 1.2)
            n_ = np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
            p = c + s * 1.0 * n_
            for m in range(5):  # five petals round a yellow eye
                b_ = 2 * np.pi * m / 5
                t1 = np.cross(n_, [0, 0, 1.0])
                t1 = t1 / (np.linalg.norm(t1) + 1e-6)
                t2 = np.cross(n_, t1)
                dv = t1 * math.cos(b_) + t2 * math.sin(b_)
                sv = np.cross(n_, dv) * 0.025
                g.add([p + sv, p + dv * 0.08 + n_ * 0.02, p - sv], [[0, 1, 2]], flowers)
            ball(g, p + n_ * 0.01, 0.018, "M_FlowerCentre", subdiv=0)
    return g


# ---------------------------------------------------------------------------
# ground cover: grass tufts, flowers, pebbles (shared with the other maps)
# ---------------------------------------------------------------------------

def grass_tuft_geo(rng, mats=("M_Grass", "M_GrassLight"), h=0.35, blades=None):
    """Curved two-segment blades: dark at the root, light at the tip."""
    g = Geo()
    for k in range(blades or int(rng.integers(10, 16))):
        a = rng.uniform(0, 2 * np.pi)
        d = np.array([math.cos(a), math.sin(a), 0])
        base = d * rng.uniform(0, 0.09)
        hh = h * rng.uniform(0.55, 1.25)
        lean = rng.uniform(0.15, 0.7)
        mid = base + d * hh * lean * 0.3 + [0, 0, hh * 0.6]
        tip = base + d * hh * lean + [0, 0, hh * (1 - 0.25 * lean)]
        side = np.array([-d[1], d[0], 0]) * rng.uniform(0.016, 0.026)
        g.add([base - side, base + side, mid + side * 0.7, mid - side * 0.7], [[0, 1, 2, 3]], mats[0])
        g.add([mid - side * 0.7, mid + side * 0.7, tip], [[0, 1, 2]], mats[-1])
    return g


def flower_geo(rng, petal_mat, centre_mat="M_FlowerCentre", stem_mat="M_Grass"):
    g = Geo()
    h = rng.uniform(0.18, 0.32)
    tube(g, (0, 0, 0), (0.02, 0, h), 0.008, mat=stem_mat, n=3)
    for k in range(5):
        a = 2 * np.pi * k / 5
        d = np.array([math.cos(a), math.sin(a), 0])
        sd = np.array([-d[1], d[0], 0])
        c = np.array([0.02, 0, h])
        g.add([c + sd * 0.018, c + d * 0.06 + [0, 0, 0.012], c - sd * 0.018], [[0, 1, 2]], petal_mat)
    ball(g, (0.02, 0, h + 0.008), 0.014, centre_mat, subdiv=0)
    for s_ in (-1, 1):
        g.add([(0, 0, 0.04), (s_ * 0.07, 0.01, 0.09), (0.0, 0.02, 0.07)], [[0, 1, 2]], stem_mat)
    return g


def pebbles_geo(rng, mats=("M_Rock", "M_RockDark", "M_RockLight")):
    g = Geo()
    for k in range(int(rng.integers(3, 7))):
        s = rng.uniform(0.04, 0.13)
        boulder(g, (rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), 0), (s * 1.3, s, s * 0.55),
                seed=int(rng.integers(1 << 30)), mats=mats, subdiv=1, strata=0, sink=0.5)
    return g


def fern_geo(rng, mats=("M_Fern", "M_FernLight"), size=1.0):
    """Arching fronds with paired leaflets, from a tight crown."""
    g = Geo()
    nf = int(rng.integers(6, 9))
    for k in range(nf):
        a = 2 * np.pi * k / nf + rng.uniform(-0.25, 0.25)
        d = np.array([math.cos(a), math.sin(a), 0])
        side = np.array([-d[1], d[0], 0])
        L = rng.uniform(0.55, 0.85) * size
        pts = [d * L * t + [0, 0, L * (1.2 * t - 1.25 * t * t) + 0.02] for t in np.linspace(0, 1, 9)]
        mat = mats[k % len(mats)]
        for i in range(8):
            tube(g, pts[i], pts[i + 1], 0.008, mat=mat, n=3)
        for i in range(1, 8):
            ln = 0.16 * size * math.sin(math.pi * i / 8) + 0.03
            fwd = (pts[i + 1] - pts[i - 1]) / np.linalg.norm(pts[i + 1] - pts[i - 1])
            for sd in (-1, 1):
                tip = pts[i] + side * sd * ln + fwd * ln * 0.4 - [0, 0, ln * 0.25]
                g.add([pts[i] - fwd * 0.02, pts[i] + fwd * 0.025, tip], [[0, 1, 2]], mat)
    return g


def reeds_geo(rng, size=1.0):
    """A clump of reeds and a couple of cattail heads."""
    g = Geo()
    for k in range(int(rng.integers(9, 15))):
        a = rng.uniform(0, 2 * np.pi)
        base = np.array([math.cos(a), math.sin(a), 0]) * rng.uniform(0, 0.18)
        h = rng.uniform(0.7, 1.4) * size
        lean = np.array([math.cos(a), math.sin(a), 0]) * rng.uniform(0.05, 0.25) * h
        mid = base + lean * 0.4 + [0, 0, h * 0.55]
        tip = base + lean + [0, 0, h]
        side = np.array([-math.sin(a), math.cos(a), 0]) * 0.02
        g.add([base - side, base + side, mid + side * 0.7, mid - side * 0.7], [[0, 1, 2, 3]], "M_Reed")
        g.add([mid - side * 0.7, mid + side * 0.7, tip], [[0, 1, 2]], "M_ReedLight")
    for k in range(int(rng.integers(1, 4))):
        a = rng.uniform(0, 2 * np.pi)
        b = np.array([math.cos(a), math.sin(a), 0]) * 0.08
        h = rng.uniform(0.9, 1.3) * size
        tube(g, b, b + [0, 0, h], 0.01, mat="M_Reed", n=3)
        tube(g, b + [0, 0, h - 0.22], b + [0, 0, h - 0.02], 0.028, mat="M_Cattail", n=6)
    return g


def seagrass_geo(rng):
    g = Geo()
    for k in range(int(rng.integers(8, 14))):
        a = rng.uniform(0, 2 * np.pi)
        base = np.array([math.cos(a), math.sin(a), 0]) * rng.uniform(0, 0.3)
        h = rng.uniform(0.3, 0.7)
        sway = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 0])
        side = np.array([-math.sin(a), math.cos(a), 0]) * 0.025
        mid = base + sway * 0.5 + [0, 0, h * 0.5]
        tip = base + sway + [0, 0, h]
        g.add([base - side, base + side, mid + side, mid - side], [[0, 1, 2, 3]], "M_Seagrass")
        g.add([mid - side, mid + side, tip], [[0, 1, 2]], "M_Seagrass")
    return g


def scatter_cover(col, name, count, sample, ok, make, rng, tries_per=6):
    """Joins `count` small pieces into one object; `ok(p)` returns the ground z or None."""
    g = Geo()
    placed = 0
    for _ in range(count * tries_per):
        if placed >= count:
            break
        p = sample()
        z = ok(p)
        if z is None:
            continue
        g.merge(make(rng), loc=(p[0], p[1], z - 0.02), rot=(0, 0, rng.uniform(0, 2 * np.pi)),
                scale=rng.uniform(0.8, 1.25))
        placed += 1
    print(f"[cover] {name} {placed}")
    return emit(name, g, col) if placed else None


def barrel(g, center, seed=0, lying=False):
    b = Geo()
    prof = [(0.27 + 0.05 * math.sin(math.pi * t), 0.85 * t) for t in np.linspace(0, 1, 9)]
    lathe(b, [(0.0, 0.015)] + prof + [(0.255, 0.85), (0.255, 0.81), (0.0, 0.81)], "M_Wood", n=24, groove=0.018,
          cap_top=False, cap_bottom=False)
    for z in (0.07, 0.2, 0.65, 0.78):
        r = 0.27 + 0.05 * math.sin(math.pi * z / 0.85) + 0.006
        lathe(b, [(r - 0.004, z - 0.025), (r + 0.004, z - 0.02), (r + 0.004, z + 0.02), (r - 0.004, z + 0.025)],
              "M_Iron", n=24, cap_top=False, cap_bottom=False)
    box(b, (0, 0, 0.815), (0.05, 0.36, 0.012), "M_WoodDark", bevel=False)  # lid batten
    ball(b, (0.12, 0.05, 0.82), 0.025, "M_WoodDark", subdiv=1)              # bung
    rot = (math.radians(90), 0, seed) if lying else (0, 0, seed)
    loc = np.asarray(center) + ([0, 0, 0.29] if lying else [0, 0, 0])
    g.merge(b, loc=loc, rot=rot)


def crate(g, center, size=0.7, yaw=0.0):
    c = Geo()
    s = size
    box(c, (0, 0, s / 2), (s - 0.03, s - 0.03, s - 0.03), "M_WoodDark", bevel=False)
    n = 4
    w = s / n
    for k in range(n):  # planks on the four sides and the lid
        t = -s / 2 + (k + 0.5) * w
        for sx in (-1, 1):
            box(c, (sx * (s / 2 - 0.012), t, s / 2), (0.025, w - 0.012, s - 0.02), "M_WoodLight", bevel=False)
            box(c, (t, sx * (s / 2 - 0.012), s / 2), (w - 0.012, 0.025, s - 0.02), "M_WoodLight", bevel=False)
        box(c, (t, 0, s - 0.012), (w - 0.012, s - 0.02, 0.025), "M_WoodLight", bevel=False)
    e = 0.065
    for sx in (-1, 1):  # frame battens on every edge
        for sy in (-1, 1):
            box(c, (sx * (s / 2 - e / 2 + 0.01), sy * (s / 2 - e / 2 + 0.01), s / 2), (e, e, s + 0.01), "M_Wood")
        for z in (e / 2, s - e / 2 + 0.01):
            box(c, (sx * (s / 2 - e / 2 + 0.01), 0, z), (e, s, e), "M_Wood")
            box(c, (0, sx * (s / 2 - e / 2 + 0.01), z), (s, e, e), "M_Wood")
    for sy in (-1, 1):  # diagonal brace on two faces
        beam(c, (-s / 2 + e, sy * (s / 2 + 0.012), e), (s / 2 - e, sy * (s / 2 + 0.012), s - e), 0.022, 0.06, "M_Wood")
    g.merge(c, loc=center, rot=(0, 0, yaw))


def torch(g, base, lights):
    base = np.asarray(base, float)
    tube(g, base - [0, 0, 0.3], base + [0, 0, 1.6], 0.06, 0.05, "M_WoodDark", n=8)
    for z in (0.5, 1.1):
        tube(g, base + [0, 0, z], base + [0, 0, z + 0.06], 0.068, mat="M_Iron", n=8)
    lathe(g, [(0.05, 1.5), (0.09, 1.55), (0.17, 1.76), (0.15, 1.78)], "M_Iron", n=12, groove=0.06,
          cap_top=False)
    lathe(g, [(0.08, 1.62), (0.12, 1.66), (0.12, 1.74), (0.09, 1.78)], "M_Rope", n=10)
    tube(g, base + [0, 0, 1.7], base + [0, 0, 2.15], 0.12, 0.0, "M_Flame", n=6)
    tube(g, base + [0, 0, 1.72], base + [0, 0, 2.0], 0.065, 0.0, "M_FlameCore", n=6)
    lights.append(base + [0, 0, 2.0])


# ---------------------------------------------------------------------------
# landmarks
# ---------------------------------------------------------------------------

STONES = ("M_Stone", "M_StoneDark", "M_StoneLight")


def block_course(g, rng, half, z, h, depth=0.5):
    """One course of dressed blocks round a square of half size `half`."""
    for side in range(4):
        t = -half + rng.uniform(0, 0.4)
        while t < half - 0.05:
            L = min(rng.uniform(0.55, 1.05), half - t)
            m = t + L / 2
            out = half - depth / 2 + rng.uniform(-0.012, 0.012)
            q = [(m, -out), (out, m), (-m, out), (-out, -m)][side]
            size = (L - 0.025, depth, h - 0.025) if side % 2 == 0 else (depth, L - 0.025, h - 0.025)
            box(g, (q[0], q[1], z + h / 2), size, STONES[int(rng.integers(3))])
            t += L


def paving(g, rng, half, inner, z, tile=0.9):
    """Square paving slabs on the ring between `inner` and `half`."""
    n = int(math.ceil(2 * half / tile))
    step = 2 * half / n
    for i in range(n):
        for j in range(n):
            x, y = -half + (i + 0.5) * step, -half + (j + 0.5) * step
            if max(abs(x), abs(y)) < inner - step / 2:
                continue
            box(g, (x, y, z + 0.03 + rng.uniform(-0.01, 0.01)), (step - 0.03, step - 0.03, 0.06),
                STONES[int(rng.integers(3))])


def pillar_geo(h, r, rng, broken=False):
    """Fluted column with entasis, a moulded base and (if whole) an echinus and abacus."""
    g = Geo()
    lathe(g, [(r * 1.42, 0.0), (r * 1.42, 0.08), (r * 1.25, 0.12), (r * 1.18, 0.2), (r * 1.06, 0.24)], "M_Stone",
          n=24, cap_top=False)
    n = 24
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    flute = 1 - 0.07 * (np.arange(n) % 2)
    rings = []
    for t in np.linspace(0, 1, 7):
        rr = r * (1 + 0.04 * math.sin(math.pi * t * 0.8)) * (1 - 0.1 * t) * flute
        z = 0.24 + h * t
        zz = np.full(n, z)
        if broken and t == 1.0:
            zz = z + rng.uniform(-0.18, 0.12, n)
        rings.append(np.stack([rr * np.cos(a), rr * np.sin(a), zz], axis=1))
    loft(g, rings, "M_Stone", cap_start=False)
    if not broken:
        z = 0.24 + h
        lathe(g, [(r * 0.9, z), (r * 1.2, z + 0.1), (r * 1.3, z + 0.16)], "M_StoneLight", n=24, cap_bottom=False)
        box(g, (0, 0, z + 0.25), (r * 2.8, r * 2.8, 0.18), "M_StoneDark")
    else:
        for k in range(3):  # loose chips at the foot
            q = rng.uniform(-r * 2.5, r * 2.5, 2)
            box(g, (q[0], q[1], 0.06), (0.16, 0.12, 0.1), "M_Stone", rot=(0.2, 0.3, rng.uniform(0, 3)))
    return g


def build_temple(col, lights):
    g = Geo()
    z0 = -0.3  # object origin sits at TEMPLE_Z
    tiers = [(9.0, 0.75), (7.2, 0.6), (5.6, 0.55)]
    z = z0
    brng = np.random.default_rng(55)
    for k, (s, h) in enumerate(tiers):
        box(g, (0, 0, z + h / 2 - 0.03), (s - 0.1, s - 0.1, h - 0.06), "M_Mortar", bevel=False)
        for c in range(2):
            block_course(g, brng, s / 2, z + c * h / 2, h / 2)
        inner = tiers[k + 1][0] / 2 if k + 1 < len(tiers) else 0.0
        paving(g, brng, s / 2, inner, z + h - 0.06)
        z += h
    top = z
    for k in range(14):  # moss creeping over the stones
        a = brng.uniform(0, 2 * np.pi)
        tier = int(brng.integers(3))
        hs = tiers[tier][0] / 2
        zt = z0 + sum(t[1] for t in tiers[:tier + 1])
        q = np.clip(np.array([math.cos(a), math.sin(a)]) * hs * 1.5, -hs + 0.25, hs - 0.25)
        rock(g, (q[0], q[1], zt), (brng.uniform(0.3, 0.6), brng.uniform(0.25, 0.45), 0.06), "M_Moss",
             seed=4000 + k, sink=0.5, rough=0.2)
    # stair run down the south face
    steps = 7
    for k in range(steps):
        zt = top - (k + 0.5) * top / steps
        box(g, (0, -2.8 - k * 0.42, (zt - 1.5) / 2), (2.6, 0.42, zt + 1.5), "M_StoneDark" if k % 2 else "M_Stone")
        box(g, (0, -2.8 - k * 0.42 - 0.02, zt - 0.015), (2.62, 0.44, 0.06), "M_StoneLight")  # worn tread
    for sx in (-1, 1):
        box(g, (sx * 1.5, -2.8 - 3 * 0.42, (top - 1.5) / 2 - 0.2), (0.4, 7 * 0.42, top + 1.1), "M_StoneDark")
    # pillars; the back row still carries its lintel
    rng = np.random.default_rng(5)
    for px, py in [(-2.2, 2.2), (0, 2.2), (2.2, 2.2), (-2.2, 0), (2.2, 0), (-2.2, -2.2), (2.2, -2.2)]:
        full = py > 2
        h = 3.0 if full else rng.uniform(0.6, 1.9)
        box(g, (px, py, top + 0.06), (0.98, 0.98, 0.12), "M_StoneDark")  # plinth
        g.merge(pillar_geo(h - 0.12, 0.33, rng, broken=not full), loc=(px, py, top + 0.12))
        if not full:
            box(g, (px + 0.6, py - 0.3, top + 0.12), (0.5, 0.42, 0.22), "M_Stone", rot=(0.3, 0.2, rng.uniform(0, 3)))
    for k, x in enumerate(np.arange(-2.7, 2.71, 0.9)):  # lintel in blocks, with a cornice and moss
        box(g, (x, 2.2, top + 3.64), (0.88, 0.8, 0.45), STONES[k % 3])
    box(g, (0, 2.2, top + 3.92), (5.6, 0.95, 0.12), "M_StoneLight")
    for k, x in enumerate(np.arange(-2.6, 2.61, 0.26)):  # dentils under the cornice
        box(g, (x, 1.83, top + 3.38), (0.12, 0.1, 0.1), "M_StoneDark", bevel=False)
    rock(g, (-0.6, 2.2, top + 3.98), (1.6, 0.45, 0.08), "M_Moss", seed=4100, sink=0.4, rough=0.25)
    rock(g, (1.7, 2.3, top + 3.98), (0.8, 0.4, 0.07), "M_Moss", seed=4101, sink=0.4, rough=0.25)
    vine_rng = np.random.default_rng(77)
    for k in range(6):  # vines hanging from the lintel
        x = vine_rng.uniform(-2.6, 2.6)
        pts = [np.array([x, 1.78, top + 3.4 - 0.25 * i]) + [vine_rng.uniform(-0.05, 0.05), 0, 0]
               for i in range(int(vine_rng.integers(4, 9)))]
        for a_, b_ in zip(pts[:-1], pts[1:]):
            tube(g, a_, b_, 0.015, mat="M_Moss", n=4)
            g.add([b_, b_ + [0.09, -0.03, 0.03], b_ + [0.02, -0.03, 0.1]], [[0, 1, 2]], "M_BushLight")
    # altar
    box(g, (0, 0.7, top + 0.06), (1.85, 1.25, 0.12), "M_Stone")
    box(g, (0, 0.7, top + 0.45), (1.5, 0.95, 0.7), "M_StoneDark")
    for x in (-0.45, 0.0, 0.45):  # carved panels on the front
        box(g, (x, 0.7 - 0.49, top + 0.45), (0.32, 0.04, 0.46), "M_StoneLight")
        box(g, (x, 0.7 - 0.515, top + 0.45), (0.12, 0.02, 0.2), "M_Gold", bevel=False)
    box(g, (0, 0.7, top + 0.85), (1.75, 1.15, 0.1), "M_Stone")
    for x in (-0.72, 0.72):
        bowl = Geo()
        lathe(bowl, [(0.05, 0.0), (0.12, 0.05), (0.14, 0.12)], "M_Gold", n=12, cap_top=False)
        g.merge(bowl, loc=(x, 0.7, top + 0.9))
    TREASURE_Z[0] = TEMPLE_Z + top + 0.9
    # fallen pillars and loose blocks behind the temple
    for p0, p1 in [((1.8, 5.4), (4.6, 6.6)), ((5.0, 4.8), (6.4, 2.6)), ((-4.8, 5.2), (-3.0, 6.6))]:
        a = TEMPLE + np.array(p0)
        b = TEMPLE + np.array(p1)
        za, zb = gz(a), gz(b)
        pa = np.array([a[0] - TEMPLE[0], a[1] - TEMPLE[1], za + 0.3 - TEMPLE_Z])
        pb = np.array([b[0] - TEMPLE[0], b[1] - TEMPLE[1], zb + 0.3 - TEMPLE_Z])
        dv = pb - pa
        L = float(np.linalg.norm(dv))
        seg = pillar_geo(L - 0.24, 0.3, rng, broken=True)
        pitch = math.atan2(math.hypot(dv[0], dv[1]), dv[2])
        g.merge(seg, loc=pa - [0, 0, 0.0], rot=(0, pitch, math.atan2(dv[1], dv[0])))
    for k in range(9):
        a = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(5.2, 6.6)
        p = TEMPLE + [d * math.cos(a), d * math.sin(a)]
        if abs(p[0]) < 1.8 and p[1] < TEMPLE[1]:
            continue
        s = rng.uniform(0.35, 0.6)
        box(g, (p[0] - TEMPLE[0], p[1] - TEMPLE[1], gz(p) - TEMPLE_Z + s * 0.3), (s, s * 0.8, s * 0.6),
            "M_StoneDark", rot=(rng.uniform(-.3, .3), rng.uniform(-.3, .3), rng.uniform(0, 3)))
    for sx in (-1, 1):
        p = TEMPLE + [sx * 1.9, -6.1]
        torch(g, (p[0] - TEMPLE[0], p[1] - TEMPLE[1], gz(p) - TEMPLE_Z), lights)
    lights[:] = [l + [TEMPLE[0], TEMPLE[1], TEMPLE_Z] for l in lights]
    SOLIDS.extend((TEMPLE + [sx * 1.9, -6.1], 0.3) for sx in (-1, 1))
    for p0, p1 in [((1.8, 5.4), (4.6, 6.6)), ((5.0, 4.8), (6.4, 2.6)), ((-4.8, 5.2), (-3.0, 6.6))]:
        for t in (0.15, 0.5, 0.85):
            SOLIDS.append((TEMPLE + np.array(p0) * (1 - t) + np.array(p1) * t, 0.6))
    return emit("Temple_Ruins", g, col, loc=(TEMPLE[0], TEMPLE[1], TEMPLE_Z))


def hull_geo(L, W, D, n_len, n_prof, hole=None, mat="M_Wood"):
    """Open-topped hull along +x; `hole(i, j)` -> True drops that plank face."""
    g = Geo()
    xs = np.linspace(-L / 2, L / 2, n_len)
    ang = np.linspace(0, np.pi, n_prof)
    rings = []
    for x in xs:
        u = abs(2 * x / L)
        w = W / 2 * max(1 - u ** 2.4, 0) ** 0.55 + 0.04
        d = D * (0.55 + 0.45 * (1 - u ** 2))
        top = D * 0.35 * u ** 2
        rings.append(np.stack([np.full(n_prof, x), w * np.cos(ang), top - d * np.sin(ang) ** 0.8], axis=1))
    v = np.concatenate(rings)
    f, mats = [], []
    for i in range(n_len - 1):
        for j in range(n_prof - 1):
            if hole and hole(i, j):
                continue
            a = i * n_prof + j
            f.append([a, a + n_prof, a + n_prof + 1, a + 1])
    g.add(v, f, mat)
    for j in range(1, n_prof - 1):  # plank strakes along the hull
        for i in range(n_len - 1):
            if hole and (hole(i, j) or hole(i, j - 1)):
                continue
            a_, b_ = rings[i][j], rings[i + 1][j]
            na = np.array([0, a_[1], a_[2] - (D * 0.35 * abs(2 * a_[0] / L) ** 2 - D * 0.3)])
            off = na / (np.linalg.norm(na) + 1e-6) * 0.02
            beam(g, a_ + off, b_ + off, 0.035, 0.035, "M_WoodDark")
    for j in (0, n_prof - 1):  # gunwales
        for i in range(n_len - 1):
            beam(g, rings[i][j] + [0, 0, 0.05], rings[i + 1][j] + [0, 0, 0.05], 0.12, 0.12, "M_WoodDark")
    return g, rings


def build_shipwreck(col):
    L, W, D = 10.0, 3.6, 2.0
    nl, npf = 15, 9

    def hole(i, j):
        return (4 <= i <= 7 and 1 <= j <= 3) or (i == 8 and j == 2) or (10 <= i <= 11 and j in (6, 7))

    g, rings = hull_geo(L, W, D, nl, npf, hole)
    # ribs show through the breach
    for i in (4, 5, 6, 7, 8):
        r = rings[i]
        for j in range(0, 4):
            beam(g, r[j] * [1, 0.97, 1] + [0, 0, 0], r[j + 1] * [1, 0.97, 1], 0.12, 0.1, "M_WoodDark")
    # broken deck planks
    rng = np.random.default_rng(11)
    for x in np.arange(-3.4, 3.6, 0.38):
        if rng.random() < 0.35 and not -2.8 < x < -1.8:  # keep the planks under the chest
            continue
        u = abs(2 * x / L)
        w = W / 2 * max(1 - u ** 2.4, 0) ** 0.55 * 0.92
        box(g, (x, 0, -0.35 + rng.uniform(-0.04, 0.04)), (0.34, 2 * w * rng.uniform(0.55, 1.0), 0.06),
            "M_WoodLight", rot=(0, 0, rng.uniform(-0.05, 0.05)))
    # snapped mast, yard and a torn sail
    tube(g, (0.4, 0, -1.6), (0.9, 0.2, 3.6), 0.17, 0.13, "M_WoodDark", n=8)
    tube(g, (0.9, 0.2, 3.6), (1.05, 0.25, 4.1), 0.13, 0.0, "M_WoodDark", n=8)
    tube(g, (0.75, -1.7, 3.1), (0.85, 1.9, 3.3), 0.07, mat="M_WoodDark", n=6)
    sail = Geo()
    nu, nv = 6, 5
    sv, sf = [], []
    for jv in range(nv):
        for iu in range(nu):
            s, t = iu / (nu - 1), jv / (nv - 1)
            y = -1.6 + 3.4 * s
            z = 3.15 + 0.1 * s - 2.2 * t
            x = 0.9 + 0.35 * math.sin(math.pi * s) * (0.4 + t) + 0.08 * math.sin(7 * s + 3 * t)
            if jv == nv - 1:
                z += [0.3, -0.25, 0.5, -0.1, 0.6, 0.2][iu]
            sv.append([x, y, z])
    for jv in range(nv - 1):
        for iu in range(nu - 1):
            if jv >= 2 and iu in (3,):
                continue
            a = jv * nu + iu
            sf.append([a, a + 1, a + nu + 1, a + nu])
    sail.add(sv, sf, "M_Cloth")
    g.merge(sail)
    mast_top = np.array([0.9, 0.2, 3.55])
    for i in (3, 5, 9, 11):  # shrouds down to the gunwales, some snapped
        for j in (0, npf - 1):
            end = rings[i][j] + [0, 0, 0.1]
            if (i + j) % 3 == 0:
                end = mast_top + (end - mast_top) * 0.55 + [0, 0.15, -0.6]
            tube(g, mast_top, end, 0.012, mat="M_Rope", n=4)
    tube(g, (L / 2 - 0.4, 0, 0.55), (L / 2 + 1.7, 0.05, 1.5), 0.09, 0.05, "M_WoodDark", n=8)  # bowsprit
    tube(g, (L / 2 + 1.6, 0.05, 1.45), mast_top, 0.012, mat="M_Rope", n=4)
    # stern cabin with windows
    box(g, (-L / 2 + 1.5, 0, 0.15), (1.7, 2.2, 1.0), "M_Wood")
    box(g, (-L / 2 + 1.5, 0, 0.7), (1.9, 2.4, 0.1), "M_WoodDark")
    for y in (-0.6, 0.0, 0.6):
        box(g, (-L / 2 + 0.64, y, 0.25), (0.04, 0.3, 0.32), "M_Dark", bevel=False)
        box(g, (-L / 2 + 0.63, y, 0.25), (0.03, 0.36, 0.04), "M_WoodLight", bevel=False)
    tube(g, (-L / 2 + 1.0, 0, 0.75), (-L / 2 + 0.6, 0, 2.1), 0.05, mat="M_WoodDark", n=6)  # stern lantern pole
    lathe(g, [(0.06, 2.05), (0.13, 2.12), (0.13, 2.38), (0.05, 2.45)], "M_LanternGlow", n=8)
    for k in range(10):  # barnacles and weed low on the hull
        i, j = 2 + (k * 7) % (nl - 4), npf // 2 + (k % 3) - 1
        ball(g, rings[i][j] + [0, 0, 0.02], 0.06 + 0.02 * (k % 2), "M_Moss" if k % 2 else "M_StoneLight", subdiv=1)
    barrel(g, (-2.2, 0.5, -1.45), seed=0.4)
    crate(g, (2.6, -0.3, -1.3), 0.6, 0.3)

    yaw = math.atan2(WRECK_DIR[1], WRECK_DIR[0]) + math.pi / 2 + math.radians(22)
    roll = math.radians(17)
    ob_loc = (WRECK_POS[0], WRECK_POS[1], gz(WRECK_POS) + 0.85)
    ob = emit("Shipwreck", g, col)
    ob.location = ob_loc
    ob.rotation_euler = (roll, 0, yaw)
    # treasure: on the deck, next to the breach
    local = Vector((-2.3, 0.0, -0.3))  # on the broken deck
    world = Euler((roll, 0, yaw), 'XYZ').to_matrix() @ local + Vector(ob_loc)
    TREASURE_SPAWNS[1] = ("Shipwreck deck", np.array([world.x, world.y]))
    TREASURE_Z[1] = world.z
    # scattered cargo on the beach
    cargo = Geo()
    for k, (dx, dy) in enumerate([(-3.5, 2.0), (3.0, 3.2), (-4.6, -0.6)]):
        p = WRECK_POS - WRECK_DIR * 3.2 + np.array([dx, dy]) * [1, 0.6]
        if k == 1:
            crate(cargo, (p[0], p[1], gz(p) - 0.1), 0.6, 0.7)
        else:
            barrel(cargo, (p[0], p[1], gz(p) - 0.05), seed=k, lying=k == 0)
        SOLIDS.append((p, 0.7))
    emit("Shipwreck_Cargo", cargo, col)
    return ob


def build_dock(col):
    g = Geo()
    yaw = math.atan2(DOCK_DIR[1], DOCK_DIR[0])
    W = 2.2
    # local frame: x along the dock (out to sea), y across
    for x in np.arange(0, DOCK_LEN + 0.01, 1.6):
        for y in (-W / 2 + 0.1, W / 2 - 0.1):
            tube(g, (x, y, -3.4), (x, y, DECK_Z + (0.55 if x > DOCK_LEN - 0.1 else 0.1)), 0.12, mat="M_WoodDark", n=7)
    for y in (-W / 2 + 0.1, W / 2 - 0.1):
        beam(g, (0, y, DECK_Z - 0.12), (DOCK_LEN, y, DECK_Z - 0.12), 0.14, 0.16, "M_WoodDark")
    rng = np.random.default_rng(3)
    for k, x in enumerate(np.arange(0.15, DOCK_LEN, 0.3)):
        if k in (19, 26):
            continue  # two missing planks
        box(g, (x, rng.uniform(-0.04, 0.04), DECK_Z), (0.26, W + rng.uniform(-0.1, 0.12), 0.06),
            "M_Wood" if k % 3 else "M_WoodLight", rot=(0, 0, rng.uniform(-0.03, 0.03)))
    barrel(g, (DOCK_LEN - 0.8, 0.6, DECK_Z + 0.03), seed=1.1)
    crate(g, (DOCK_LEN - 1.5, -0.5, DECK_Z + 0.03), 0.55, 0.2)
    ob = emit("Dock", g, col)
    ob.location = (DOCK_BASE[0], DOCK_BASE[1], 0)
    ob.rotation_euler = (0, 0, yaw)

    boat, _ = hull_geo(3.4, 1.3, 0.55, 9, 7, mat="M_WoodLight")
    for x in (-0.6, 0.5):
        box(boat, (x, 0, -0.12), (0.22, 1.1, 0.05), "M_WoodDark")
    tube(boat, (0.0, 0.5, 0.0), (1.4, 1.5, 0.3), 0.035, mat="M_WoodDark", n=5)
    end = DOCK_BASE + DOCK_DIR * (DOCK_LEN - 2.4) + np.array([-DOCK_DIR[1], DOCK_DIR[0]]) * -2.0
    bob = emit("Rowboat", boat, col)
    bob.location = (end[0], end[1], 0.32)
    bob.rotation_euler = (math.radians(4), 0, yaw + 0.12)
    return ob


def build_arch(col):
    g = Geo()
    axis = np.array([math.cos(0.5), math.sin(0.5)])
    for k, s in enumerate((-1, 1)):
        p = ARCH + axis * 2.3 * s
        SOLIDS.append((p, 1.9))
        boulder(g, (p[0], p[1], gz(p) + 1.2), (1.5, 1.7, 2.6), seed=40 + k, yaw=0.5, sink=0.1, moss=0.3, subdiv=3)
    lz = gz(ARCH) + 3.6
    boulder(g, (ARCH[0], ARCH[1], lz), (3.9, 1.6, 1.15), seed=44, yaw=0.5, subdiv=3, moss=0.55, tilt=0.03,
            mats=("M_RockDark", "M_Rock", "M_RockLight"))
    boulder(g, (ARCH[0] + 0.4, ARCH[1] + 0.2, lz + 0.75), (1.8, 1.0, 0.55), seed=45, yaw=0.6, moss=0.9, tilt=0.05)
    for k in range(5):  # ferns growing on top of the arch
        q = np.array([ARCH[0] - 1.6 + 0.8 * k, ARCH[1] + 0.3 * math.sin(k), lz + 1.05])
        g.merge(fern_geo(np.random.default_rng(470 + k), size=0.8), loc=q, rot=(0, 0, k))
    rng = np.random.default_rng(46)
    for k in range(6):
        a = rng.uniform(0, 2 * np.pi)
        p = ARCH + rng.uniform(3.8, 5.5) * np.array([math.cos(a), math.sin(a)])
        s = rng.uniform(0.5, 1.1)
        boulder(g, (p[0], p[1], gz(p) + s * 0.3), (s * 1.2, s, s * 0.8), seed=50 + k, yaw=a, moss=0.25)
        SOLIDS.append((p, s * 1.3))
    return emit("Rock_Arch", g, col)


def build_camp(col, lights):
    g = Geo()
    c = np.array([CAMP[0], CAMP[1], CAMP_Z])
    rng = np.random.default_rng(9)
    for k in range(9):
        a = 2 * np.pi * k / 9
        boulder(g, c + [0.75 * math.cos(a), 0.75 * math.sin(a), 0.05], (0.2, 0.17, 0.14), seed=60 + k, yaw=a,
                subdiv=1, mats=("M_RockDark", "M_Dark", "M_RockDark"))
    for k in range(3):
        a = 2 * np.pi * k / 3 + 0.4
        d = np.array([math.cos(a), math.sin(a), 0])
        tube(g, c + d * 0.5 + [0, 0, 0.08], c - d * 0.25 + [0, 0, 0.3], 0.07, mat="M_WoodDark", n=6)
    tube(g, c + [0, 0, 0.1], c + [0, 0, 1.05], 0.32, 0.0, "M_Flame", n=7)
    tube(g, c + [0, 0, 0.1], c + [0, 0, 0.7], 0.18, 0.0, "M_FlameCore", n=7)
    lights.append(c + [0, 0, 0.7])
    # tent: an A-frame of red sailcloth, opening towards the fire
    t = c + [-1.8, 2.0, -0.05]
    tl, td, th = 2.6, 1.1, 1.5  # length (along y), half width, height
    v = np.array([[-td, -tl / 2, 0], [0, -tl / 2, th], [td, -tl / 2, 0],
                  [-td, tl / 2, 0], [0, tl / 2, th], [td, tl / 2, 0]]) @ np.array(
        Euler((0, 0, -0.6)).to_matrix()).T + t
    g.add(v, [[0, 3, 4, 1], [1, 4, 5, 2], [3, 5, 4]], "M_ClothRed")
    for p in (v[1], v[4]):
        tube(g, p - [0, 0, th], p + [0, 0, 0.15], 0.04, mat="M_WoodDark", n=5)
    tent_floor = t + Euler((0, 0, -0.6)).to_matrix() @ Vector((0, 0.15, 0))
    TREASURE_SPAWNS[3] = ("Pirate camp tent", np.array([tent_floor[0], tent_floor[1]]))
    # log benches, barrels, crates
    for a in (2.3, 4.1):
        d = np.array([math.cos(a), math.sin(a), 0])
        s = np.array([-d[1], d[0], 0])
        p = c + d * 1.9 + [0, 0, 0.2]
        tube(g, p - s * 0.8, p + s * 0.8, 0.2, mat="M_Wood", n=7)
    barrel(g, c + [1.8, 1.6, 0], seed=0.2)
    barrel(g, c + [2.3, 1.1, 0], seed=1.4)
    crate(g, c + [1.9, -1.6, 0], 0.65, 0.5)
    crate(g, c + [2.2, -1.6, 0.65], 0.5, 0.9)
    # a wanted-style signpost pointing to the temple
    p = c + [2.8, 0.8, 0]
    tube(g, p - [0, 0, 0.2], p + [0, 0, 1.7], 0.06, mat="M_WoodDark", n=6)
    d = (np.array([*TEMPLE, 0]) - p) * [1, 1, 0]
    d /= np.linalg.norm(d)
    beam(g, p + [0, 0, 1.45] - d * 0.1, p + [0, 0, 1.45] + d * 0.8, 0.22, 0.05, "M_WoodLight")
    cxy = CAMP
    SOLIDS.extend([(cxy, 1.3), (TREASURE_SPAWNS[3][1], 1.9), (cxy + [1.8, 1.6], 0.45), (cxy + [2.3, 1.1], 0.45),
                   (cxy + [1.9, -1.6], 0.55), (cxy + [2.8, 0.8], 0.3)])
    for a in (2.3, 4.1):
        SOLIDS.append((cxy + 1.9 * np.array([math.cos(a), math.sin(a)]), 0.9))
    return emit("Pirate_Camp", g, col)


def build_x_mark(col, xy):
    g = Geo()
    z = gz(xy) + 0.03
    for a in (math.radians(45), math.radians(-45)):
        d = np.array([math.cos(a), math.sin(a), 0]) * 0.75
        beam(g, (xy[0], xy[1], z) - d, (xy[0], xy[1], z) + d, 0.22, 0.04, "M_PaintRed")
    tube(g, (xy[0] + 1.1, xy[1] + 0.4, gz(xy) - 0.3), (xy[0] + 1.2, xy[1] + 0.5, gz(xy) + 1.0), 0.05,
         mat="M_WoodDark", n=6)  # a shovel handle left behind
    box(g, (xy[0] + 1.22, xy[1] + 0.52, gz(xy) + 1.0), (0.12, 0.3, 0.05), "M_Iron", rot=(0.2, 0, 0.3))
    SOLIDS.append((np.asarray(xy, float) + [0.6, 0.2], 1.4))
    return emit("X_Marks_The_Spot", g, col)


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------

def landmark_clear(p, extra=0.0):
    dt = max(abs(p[0] - TEMPLE[0]), abs(p[1] - TEMPLE[1]))
    if dt < 7.2 + extra:
        return False
    for c, r in ((ARCH, 4.5), (CAMP, 3.6), (WRECK_POS, 6.5), (DOCK_BASE, 3.5), *STRUCTURES):
        if np.hypot(*(p - c)) < r + extra:
            return False
    for _, s in TREASURE_SPAWNS:
        if s is not None and np.hypot(*(p - s)) < 1.6 + extra:
            return False
    return float(path_dist(np.array([p[0]]), np.array([p[1]]))[0]) > 1.5 + extra


def scatter(count, h_range, min_gap, existing, tries=4000, near=None):
    pts = []
    for _ in range(tries):
        if len(pts) >= count:
            break
        if near is not None:
            c, r = near
            a, d = RNG.uniform(0, 2 * np.pi), r * math.sqrt(RNG.random())
            p = c + d * np.array([math.cos(a), math.sin(a)])
        else:
            p = RNG.uniform(-32, 32, 2)
        h = gz(p)
        if not (h_range[0] <= h <= h_range[1]) or not landmark_clear(p):
            continue
        if any(np.hypot(*(p - q)) < min_gap for q in pts + existing):
            continue
        pts.append(p)
    return pts


def build_vegetation(col):
    taken = []
    palms = scatter(18, (0.25, 2.6), 3.2, taken)
    palms += scatter(4, (0.2, 3.5), 2.4, taken + palms, near=(CAMP, 6.5))
    palms += scatter(2, (0.1, 2.0), 1.8, taken + palms, near=(ISLET, 2.3))
    g = Geo()
    for k, p in enumerate(palms):
        g.merge(palm_geo(100 + k, RNG.uniform(4.2, 6.2)), loc=(p[0], p[1], gz(p)))
    emit("Palms", g, col)
    PALMS[:] = palms
    taken += palms
    bushes = scatter(26, (1.2, 9.0), 2.0, taken)
    g = Geo()
    for k, p in enumerate(bushes):
        g.merge(bush_geo(200 + k, "M_FlowerRed" if k % 3 == 0 else None), loc=(p[0], p[1], gz(p)), rot=(0, 0, RNG.uniform(0, 6.3)),
                scale=RNG.uniform(0.9, 1.4))
    emit("Bushes", g, col)
    taken += bushes
    rocks = scatter(22, (-0.8, 9.0), 2.2, taken)
    g = Geo()
    for k, p in enumerate(rocks):
        s = RNG.uniform(0.35, 1.0)
        boulder(g, (p[0], p[1], gz(p)), (s * RNG.uniform(1.0, 1.6), s, s * 0.7), moss=0.2 if gz(p) > 1 else 0.0,
             seed=300 + k, yaw=RNG.uniform(0, 6.3))
    # boulders around the north beach spawn, big enough to hide behind
    nb = TREASURE_SPAWNS[7][1]
    for k, (dx, dy, s) in enumerate([(-1.4, 0.9, 1.4), (1.3, 1.2, 1.1), (0.2, 2.0, 0.9)]):
        p = nb + [dx, dy]
        boulder(g, (p[0], p[1], gz(p) + 0.2), (s * 1.3, s, s), seed=400 + k, yaw=k, subdiv=3)
        SOLIDS.append((p, s * 1.4))
    emit("Rocks", g, col)
    SOLIDS.extend([(p, 0.45) for p in palms] + [(p, 0.8) for p in bushes] + [(p, 1.25) for p in rocks])
    return palms, bushes, rocks


# ---------------------------------------------------------------------------
# gameplay markers and chests
# ---------------------------------------------------------------------------

PALMS = []


def build_ground_cover(col):
    """Grass, flowers, dune grass, shells and pebbles, kept off paths, buildings and the wreck."""
    rng = np.random.default_rng(909)
    sol = np.array([c for c, _ in SOLIDS])
    rad = np.array([r for _, r in SOLIDS])

    def clear(p, z_lo, z_hi, path_gap=0.9):
        z = gz(p)
        if not (z_lo <= z <= z_hi):
            return None
        if max(abs(p[0] - TEMPLE[0]), abs(p[1] - TEMPLE[1])) < 5.8 or np.hypot(*(p - CAMP)) < 3.4:
            return None
        if np.hypot(*(p - WRECK_POS)) < 6.0:
            return None
        d = p - DOCK_BASE
        if -1.0 < d @ DOCK_DIR < DOCK_LEN + 1 and abs(d @ np.array([-DOCK_DIR[1], DOCK_DIR[0]])) < 1.8:
            return None
        if np.any(np.hypot(sol[:, 0] - p[0], sol[:, 1] - p[1]) < rad * 0.75):
            return None
        if float(path_dist(np.array([p[0]]), np.array([p[1]]))[0]) < path_gap:
            return None
        e = 0.5
        if math.hypot(gz(p + [e, 0]) - gz(p - [e, 0]), gz(p + [0, e]) - gz(p - [0, e])) / (2 * e) > 0.5:
            return None
        return z

    def sample():
        return rng.uniform(-31, 31, 2)

    n = 1 if FAST else 3
    scatter_cover(col, "Grass", 900 * n, sample, lambda p: clear(p, 1.05, 30), lambda r: grass_tuft_geo(r), rng)
    scatter_cover(col, "Flowers", 120 * n, sample, lambda p: clear(p, 1.1, 30),
                  lambda r: flower_geo(r, ("M_FlowerRed", "M_FlowerYellow", "M_FlowerWhite")[int(r.integers(3))]),
                  rng)
    scatter_cover(col, "Dune_Grass", 120 * n, sample, lambda p: clear(p, 0.3, 1.0, 0.6),
                  lambda r: grass_tuft_geo(r, ("M_DuneGrass", "M_GrassLight"), h=0.45, blades=9), rng)
    shell_g = item_geo("shell")
    star_g = item_geo("starfish")

    def beach_bits(r):
        k = int(r.integers(4))
        return pebbles_geo(r) if k < 2 else Geo().merge(shell_g if k == 2 else star_g, scale=0.7)

    scatter_cover(col, "Beach_Bits", 110 * n, sample, lambda p: clear(p, 0.05, 0.9, 0.5), beach_bits, rng)
    scatter_cover(col, "Pebbles", 60 * n, sample, lambda p: clear(p, 1.0, 30, 0.0), lambda r: pebbles_geo(r), rng)
    pal = np.array(PALMS) if PALMS else np.zeros((1, 2))

    def under_palm():
        return pal[int(rng.integers(len(pal)))] + rng.uniform(-2.2, 2.2, 2)

    scatter_cover(col, "Ferns", 45 * n, under_palm, lambda p: clear(p, 0.9, 30), lambda r: fern_geo(r), rng)
    scatter_cover(col, "Tall_Grass", 70 * n, sample, lambda p: clear(p, 1.3, 30, 1.2),
                  lambda r: grass_tuft_geo(r, ("M_Grass", "M_GrassLight"), h=0.75, blades=22), rng)

    def shallows(p):
        z = gz(p)
        return z if -1.6 < z < -0.35 and np.hypot(*(p - WRECK_POS)) > 5 else None

    scatter_cover(col, "Seagrass", 45 * n, lambda: shore_point(rng.uniform(-np.pi, np.pi), inland=-rng.uniform(0.5, 3.5)),
                  shallows, lambda r: seagrass_geo(r), rng)

    def shore(p):  # rocks breaking the waterline
        z = gz(p)
        if not (-0.6 < z < 0.25) or np.hypot(*(p - WRECK_POS)) < 6.5:
            return None
        d = p - DOCK_BASE
        if -1.0 < d @ DOCK_DIR < DOCK_LEN + 1 and abs(d @ np.array([-DOCK_DIR[1], DOCK_DIR[0]])) < 2.5:
            return None
        return z

    def shore_rock(r):
        g = Geo()
        for k in range(int(r.integers(1, 4))):
            sz = r.uniform(0.25, 0.65)
            boulder(g, (r.uniform(-0.5, 0.5), r.uniform(-0.5, 0.5), sz * 0.2), (sz * 1.4, sz, sz * 0.7),
                    seed=int(r.integers(1 << 30)), yaw=r.uniform(0, 6.3), subdiv=2)
        return g

    scatter_cover(col, "Shore_Rocks", 16 * n, lambda: rng.uniform(-36, 36, 2), shore, shore_rock, rng)

    def path_edge():
        pl = PATHS[int(rng.integers(len(PATHS)))]
        i = int(rng.integers(len(pl) - 1))
        a_, b_ = np.asarray(pl[i], float), np.asarray(pl[i + 1], float)
        t = rng.random()
        q = a_ + (b_ - a_) * t
        nrm_ = np.array([-(b_ - a_)[1], (b_ - a_)[0]]) / (np.linalg.norm(b_ - a_) + 1e-9)
        return q + nrm_ * rng.choice([-1, 1]) * rng.uniform(1.0, 1.4)

    scatter_cover(col, "Path_Stones", 70 * n, path_edge, lambda p: clear(p, 0.3, 30, 0.85), lambda r: pebbles_geo(r),
                  rng)


def build_spawns(col):
    chest_closed, chest_open = chest_geo(False), chest_geo(True)
    spawns = []
    for i, (label, xy) in enumerate(TREASURE_SPAWNS):
        z = TREASURE_Z.get(i, gz(xy) - (0.12 if i == 5 else 0.0))
        yaw = float(RNG.uniform(0, 2 * np.pi)) if i not in (0, 3) else (0.0 if i == 0 else -0.6)
        if i == 1:
            yaw = math.atan2(WRECK_DIR[1], WRECK_DIR[0]) + math.pi / 2
        em = marker(f"TreasureSpawn_{i + 1:02d}", col, (xy[0], xy[1], z), "treasure_spawn", yaw, index=i + 1,
                    label=label)
        g = Geo().merge(chest_open if i == 0 else chest_closed)
        if i == 0:
            coin_pile(g, (0.0, -0.05, 0.3), 0.18, seed=1)
        if i in (2, 6):
            coin_pile(g, (0.75, -0.4, 0.0), 0.12, seed=i)
        ch = emit(f"TreasureChest_{i + 1:02d}", g, col)
        ch.parent = em
        ch["kind"] = "treasure_chest"
        spawns.append((label, (xy[0], xy[1], z)))
    side = np.array([-DOCK_DIR[1], DOCK_DIR[0]])
    players = []
    for k in range(6):
        p = DOCK_BASE - DOCK_DIR * (3.0 + 1.6 * (k // 3)) + side * (1.6 * (k % 3 - 1))
        yaw = math.atan2(-DOCK_DIR[1], -DOCK_DIR[0])
        marker(f"PlayerSpawn_{k + 1:02d}", col, (p[0], p[1], gz(p)), "player_spawn", yaw, size=0.9, index=k + 1)
        players.append(p)
    SOLIDS.extend([(np.array(p[:2]), 1.0) for _, p in spawns] + [(p, 0.8) for p in players])
    return spawns, players


# ---------------------------------------------------------------------------
# treasure props: 200 everyday items, any of which can hide the treasure
# ---------------------------------------------------------------------------
# Every item type is built once; the placed props are separate objects sharing that
# mesh, so the glb stores each shape only once.  Items are placed by zone (beach,
# palm shade, temple, camp, wreck, arch, dock, islet, inland) so each spot gets
# things that belong there: shells on the beach, urns in the temple, cannonballs
# around the wreck.

def lathe(g, prof, mat, n=10, cap_top=True, cap_bottom=True, groove=0.0):
    """prof: list of (radius, z); a surface of revolution around +z.  `groove` insets every
    other vertex by that fraction, so an even `n` reads as n/2 staves or ribs."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = 1 - groove * (np.arange(n) % 2)
    rings = [np.stack([max(r, 1e-3) * k * np.cos(a), max(r, 1e-3) * k * np.sin(a), np.full(n, z)], axis=1)
             for r, z in prof]
    return loft(g, rings, mat, cap_start=cap_bottom, cap_end=cap_top)


def ring_path(g, R, r, mat, n=16, m=6, z=0.0, turns=1.0, rise=0.0):
    """Tube bent around a circle of radius R (a torus, or a coil when rise > 0)."""
    rings = []
    for i in range(int(n * turns) + 1):
        t = 2 * np.pi * i / n
        c = np.array([R * math.cos(t), R * math.sin(t), z + rise * i / n])
        out = np.array([math.cos(t), math.sin(t), 0])
        b = np.linspace(0, 2 * np.pi, m, endpoint=False)
        rings.append(c + r * (np.cos(b)[:, None] * out + np.sin(b)[:, None] * [0, 0, 1]))
    return loft(g, rings, mat)


def ball(g, c, r, mat, subdiv=1, squash=(1, 1, 1)):
    v, f = ico(subdiv)
    return g.add(v * r * np.asarray(squash, float) + c, f, mat)


def item_geo(kind):
    g = Geo()
    if kind == "barrel":
        barrel(g, (0, 0, 0))
    elif kind == "crate":
        crate(g, (0, 0, 0), 0.6)
    elif kind == "small_chest":
        g.merge(chest_geo(False), scale=0.6)
    elif kind == "coconut":
        ball(g, (0, 0, 0.11), 0.13, "M_Coconut", squash=(1, 0.9, 0.85))
    elif kind == "coconuts":
        for p in ((0, 0, 0.1), (0.2, 0.06, 0.1), (0.09, 0.18, 0.1), (0.1, 0.08, 0.27)):
            ball(g, p, 0.12, "M_Coconut", squash=(1, 0.9, 0.85))
    elif kind == "sack":
        rock(g, (0, 0, 0.26), (0.3, 0.25, 0.3), "M_Sack", seed=7, rough=0.08, sink=0.15)
        tube(g, (0, 0, 0.5), (0, 0, 0.66), 0.08, 0.13, "M_Sack", n=7)
        tube(g, (0, 0, 0.53), (0, 0, 0.57), 0.09, mat="M_Rope", n=7)
    elif kind == "coin_pouch":
        rock(g, (0, 0, 0.12), (0.14, 0.12, 0.14), "M_ClothRed", seed=8, rough=0.08, sink=0.15)
        tube(g, (0, 0, 0.23), (0, 0, 0.31), 0.04, 0.07, "M_ClothRed", n=7)
        coin_pile(g, (0.26, 0.05, 0), 0.09, seed=4)
    elif kind == "amphora":
        lathe(g, [(0.07, 0), (0.15, 0.1), (0.2, 0.28), (0.17, 0.46), (0.07, 0.6), (0.06, 0.72), (0.09, 0.75)],
              "M_Terracotta", n=10)
        for s in (-1, 1):
            tube(g, (s * 0.08, 0, 0.7), (s * 0.18, 0, 0.62), 0.02, mat="M_Terracotta", n=5)
            tube(g, (s * 0.18, 0, 0.62), (s * 0.17, 0, 0.47), 0.02, mat="M_Terracotta", n=5)
    elif kind == "urn":
        lathe(g, [(0.12, 0), (0.24, 0.12), (0.27, 0.3), (0.2, 0.46), (0.21, 0.5)], "M_StoneDark", n=10)
        lathe(g, [(0.22, 0.5), (0.18, 0.56), (0.05, 0.6), (0.06, 0.66)], "M_Stone", n=10)
        tube(g, (-0.27, 0, 0.32), (0.27, 0, 0.32), 0.02, mat="M_Gold", n=5)
    elif kind == "clay_pot":
        lathe(g, [(0.12, 0), (0.2, 0.08), (0.22, 0.18), (0.15, 0.3), (0.17, 0.33)], "M_Terracotta", n=10,
              cap_top=False)
    elif kind == "cooking_pot":
        lathe(g, [(0.15, 0.1), (0.24, 0.16), (0.25, 0.34), (0.27, 0.37)], "M_Iron", n=12, cap_top=False)
        for k in range(3):
            a = 2 * np.pi * k / 3
            tube(g, (0.16 * math.cos(a), 0.16 * math.sin(a), 0.12), (0.2 * math.cos(a), 0.2 * math.sin(a), 0),
                 0.025, mat="M_Iron", n=5)
        tube(g, (-0.27, 0, 0.36), (0, 0, 0.6), 0.015, mat="M_Iron", n=5)
        tube(g, (0, 0, 0.6), (0.27, 0, 0.36), 0.015, mat="M_Iron", n=5)
    elif kind == "bucket":
        lathe(g, [(0.15, 0), (0.19, 0.32)], "M_Wood", n=10, cap_top=False)
        for z in (0.05, 0.26):
            lathe(g, [(0.155 + z * 0.13 + 0.008, z), (0.16 + z * 0.13 + 0.008, z + 0.04)], "M_Iron", n=10)
        tube(g, (-0.19, 0, 0.3), (0, 0, 0.48), 0.012, mat="M_Iron", n=4)
        tube(g, (0, 0, 0.48), (0.19, 0, 0.3), 0.012, mat="M_Iron", n=4)
    elif kind == "bottle":  # message in a bottle, lying in the sand
        b = Geo()
        lathe(b, [(0.06, 0), (0.07, 0.03), (0.07, 0.2), (0.03, 0.27), (0.025, 0.32)], "M_Glass", n=8)
        tube(b, (0, 0, 0.31), (0, 0, 0.35), 0.022, mat="M_Wood", n=6)
        tube(b, (0, 0, 0.05), (0, 0, 0.2), 0.035, mat="M_Paper", n=6)
        g.merge(b, loc=(0, 0.17, 0.065), rot=(math.radians(90), 0, 0))
    elif kind == "rum_bottle":
        lathe(g, [(0.06, 0), (0.065, 0.18), (0.03, 0.25), (0.022, 0.32)], "M_GlassBrown", n=8)
        tube(g, (0, 0, 0.07), (0, 0, 0.14), 0.067, mat="M_Paper", n=8)
        tube(g, (0, 0, 0.31), (0, 0, 0.35), 0.02, mat="M_Wood", n=6)
    elif kind == "skull":
        ball(g, (0, 0, 0.13), 0.13, "M_Bone", subdiv=2, squash=(0.85, 1.0, 0.85))
        box(g, (0, -0.07, 0.04), (0.14, 0.1, 0.06), "M_Bone")
        for s in (-1, 1):
            ball(g, (s * 0.045, -0.105, 0.14), 0.035, "M_Dark")
            tube(g, (s * 0.25, -0.15, 0.02), (-s * 0.25, 0.15, 0.02), 0.022, mat="M_Bone", n=5)
        ball(g, (0, -0.125, 0.09), 0.018, "M_Dark")
    elif kind == "shell":
        n = 9
        ang = np.linspace(math.radians(25), math.radians(155), n)
        v = [[0, -0.07, 0.01], [0, -0.03, 0.0]]
        for i, a in enumerate(ang):
            r = 0.13
            v.append([r * math.cos(a), -0.07 + r * math.sin(a), 0.035 + (0.012 if i % 2 else 0)])
        f = [[0, 2 + i, 3 + i] for i in range(n - 1)] + [[1, 3 + i, 2 + i] for i in range(n - 1)]
        g.add(v, f, "M_Shell")
    elif kind == "starfish":
        n = 5
        v = [[0, 0, 0.045], [0, 0, 0.0]]
        for i in range(2 * n):
            a = math.pi / 2 + math.pi * i / n
            r = 0.17 if i % 2 == 0 else 0.065
            v.append([r * math.cos(a), r * math.sin(a), 0.012])
        m = 2 * n
        f = [[0, 2 + i, 2 + (i + 1) % m] for i in range(m)] + [[1, 2 + (i + 1) % m, 2 + i] for i in range(m)]
        g.add(v, f, "M_Starfish")
    elif kind == "lantern":
        box(g, (0, 0, 0.03), (0.2, 0.2, 0.06), "M_Iron")
        box(g, (0, 0, 0.17), (0.15, 0.15, 0.22), "M_LanternGlow")
        for sx in (-1, 1):
            for sy in (-1, 1):
                box(g, (sx * 0.085, sy * 0.085, 0.17), (0.025, 0.025, 0.24), "M_Iron")
        tube(g, (0, 0, 0.28), (0, 0, 0.38), 0.13, 0.0, "M_Iron", n=4, phase=math.pi / 4)
        tube(g, (0, 0, 0.38), (0, 0, 0.44), 0.012, mat="M_Iron", n=4)
    elif kind == "cannonballs":
        r = 0.1
        for layer, nn in enumerate((3, 2, 1)):
            off = (3 - nn) * r
            for i in range(nn):
                for j in range(nn):
                    ball(g, (-2 * r + off + 2 * r * i, -2 * r + off + 2 * r * j, r + layer * r * 1.42), r, "M_Iron")
    elif kind == "anchor":
        a = Geo()
        tube(a, (0, 0, 0.0), (0, 0, 1.1), 0.045, mat="M_Iron", n=6)
        tube(a, (-0.3, 0, 0.9), (0.3, 0, 0.9), 0.035, mat="M_Iron", n=6)
        ring_path(a, 0.08, 0.02, "M_Iron", n=10, m=5, z=1.17)
        for s in (-1, 1):
            tube(a, (0, 0, 0.02), (s * 0.3, 0, 0.16), 0.045, mat="M_Iron", n=6)
            tube(a, (s * 0.3, 0, 0.16), (s * 0.4, 0, 0.36), 0.04, mat="M_Iron", n=6)
            tube(a, (s * 0.4, 0, 0.36), (s * 0.46, 0, 0.42), 0.07, 0.0, "M_Iron", n=4)
        g.merge(a, loc=(0, -0.45, 0.07), rot=(math.radians(-78), 0, 0))
    elif kind == "ship_wheel":
        w = Geo()
        ring_path(w, 0.42, 0.035, "M_WoodDark", n=16, m=5)
        tube(w, (0, 0, -0.06), (0, 0, 0.06), 0.09, mat="M_WoodLight", n=8)
        for k in range(8):
            a = 2 * np.pi * k / 8
            d = np.array([math.cos(a), math.sin(a), 0])
            tube(w, d * 0.08, d * 0.58, 0.025, mat="M_WoodLight", n=5)
            tube(w, d * 0.5, d * 0.62, 0.035, 0.025, "M_WoodDark", n=5)
        g.merge(w, loc=(0, 0, 0.62), rot=(math.radians(78), 0, 0))
    elif kind == "cannon":
        box(g, (0, 0, 0.2), (0.45, 0.85, 0.2), "M_WoodDark")
        for sx in (-1, 1):
            for y in (-0.3, 0.3):
                tube(g, (sx * 0.22, y, 0.14), (sx * 0.3, y, 0.14), 0.14, mat="M_Wood", n=8)
        tube(g, (0, -0.45, 0.42), (0, 0.65, 0.48), 0.15, 0.11, "M_Iron", n=10)
        ball(g, (0, -0.48, 0.42), 0.12, "M_Iron")
        tube(g, (0, 0.6, 0.475), (0, 0.7, 0.48), 0.13, mat="M_Iron", n=10)
    elif kind == "rope_coil":
        ring_path(g, 0.2, 0.045, "M_Rope", n=14, m=6, z=0.045, turns=2.6, rise=0.075)
    elif kind == "idol":
        box(g, (0, 0, 0.08), (0.36, 0.36, 0.16), "M_StoneDark")
        box(g, (0, 0, 0.36), (0.24, 0.2, 0.4), "M_Stone")
        box(g, (0, 0, 0.67), (0.3, 0.26, 0.24), "M_Gold")
        for s in (-1, 1):
            box(g, (s * 0.07, -0.135, 0.7), (0.06, 0.02, 0.04), "M_GemRed")
            box(g, (s * 0.16, -0.02, 0.38), (0.06, 0.12, 0.2), "M_Stone")
        box(g, (0, 0, 0.83), (0.36, 0.08, 0.08), "M_Gold")
    elif kind == "gems":
        rock(g, (0, 0, 0.05), (0.25, 0.2, 0.12), "M_RockDark", seed=12, sink=0.4)
        rng = np.random.default_rng(13)
        for k in range(5):
            a = 2 * np.pi * k / 5 + rng.uniform(-.3, .3)
            p = np.array([0.1 * math.cos(a), 0.08 * math.sin(a), 0.08])
            tip = p + [0.12 * math.cos(a), 0.12 * math.sin(a), rng.uniform(0.18, 0.32)]
            tube(g, p, tip, 0.05, 0.0, "M_GemBlue" if k % 2 else "M_GemRed", n=5)
    elif kind == "stump":
        tube(g, (0, 0, -0.1), (0, 0, 0.38), 0.3, 0.27, "M_Wood", n=9)
        tube(g, (0, 0, 0.38), (0, 0, 0.4), 0.25, mat="M_WoodLight", n=9)
        for k in range(3):
            a = 2 * np.pi * k / 3 + 0.4
            d = np.array([math.cos(a), math.sin(a), 0])
            tube(g, d * 0.22 + [0, 0, 0.08], d * 0.5 + [0, 0, -0.02], 0.08, 0.04, "M_Wood", n=6)
    elif kind == "driftwood":
        tube(g, (-0.7, 0, 0.1), (0.7, 0.1, 0.09), 0.1, 0.07, "M_Driftwood", n=7)
        tube(g, (0.1, 0.05, 0.1), (0.45, 0.4, 0.14), 0.05, 0.02, "M_Driftwood", n=5)
    elif kind == "shovel":
        tube(g, (0, 0, -0.1), (0.12, 0.05, 1.0), 0.025, mat="M_Wood", n=6)
        box(g, (0.0, 0.0, 0.0), (0.2, 0.04, 0.26), "M_Iron", rot=(0, 0.12, 0))
        box(g, (0.13, 0.055, 1.02), (0.16, 0.035, 0.035), "M_WoodDark")
    elif kind == "sword":
        s = Geo()
        beam(s, (0, 0, -0.25), (0, 0, 0.55), 0.06, 0.012, "M_Steel")
        box(s, (0, 0, 0.57), (0.26, 0.05, 0.04), "M_Gold")
        tube(s, (0, 0, 0.59), (0, 0, 0.76), 0.022, mat="M_ClothRed", n=6)
        ball(s, (0, 0, 0.79), 0.035, "M_Gold")
        g.merge(s, rot=(0.15, -0.1, 0))
    elif kind == "helmet":
        lathe(g, [(0.2, 0.0), (0.19, 0.08), (0.15, 0.16), (0.08, 0.21), (0.01, 0.22)], "M_Steel", n=12)
        lathe(g, [(0.2, 0.0), (0.3, 0.02), (0.31, 0.04), (0.2, 0.03)], "M_Steel", n=12)
        beam(g, (0, -0.15, 0.17), (0, 0.15, 0.17), 0.02, 0.14, "M_Steel")
    elif kind == "book":
        box(g, (0, 0, 0.04), (0.3, 0.22, 0.08), "M_ClothRed", rot=(0, 0, 0.2))
        box(g, (0.01, 0, 0.04), (0.29, 0.2, 0.066), "M_Paper", rot=(0, 0, 0.2))
        box(g, (0.03, 0.01, 0.1), (0.26, 0.19, 0.05), "M_Leather", rot=(0, 0, -0.35))
        box(g, (0.04, 0.01, 0.1), (0.25, 0.17, 0.038), "M_Paper", rot=(0, 0, -0.35))
    elif kind == "scroll":
        tube(g, (-0.2, 0, 0.045), (0.2, 0, 0.045), 0.045, mat="M_Paper", n=8)
        tube(g, (-0.03, 0, 0.045), (0.03, 0, 0.045), 0.05, mat="M_ClothRed", n=8)
        tube(g, (-0.24, 0, 0.045), (-0.2, 0, 0.045), 0.02, mat="M_WoodDark", n=6)
        tube(g, (0.2, 0, 0.045), (0.24, 0, 0.045), 0.02, mat="M_WoodDark", n=6)
    elif kind == "mushrooms":
        for (x, y, h, r) in ((0, 0, 0.22, 0.13), (0.17, 0.08, 0.14, 0.09), (-0.1, 0.14, 0.1, 0.07)):
            tube(g, (x, y, 0), (x, y, h), r * 0.35, r * 0.3, "M_MushroomStem", n=6)
            lathe(g, [(r * 0.4, h - 0.01), (r, h + 0.01), (r * 0.8, h + r * 0.45), (0.0, h + r * 0.6)],
                  "M_Mushroom", n=9, cap_top=False)
    elif kind == "candles":
        for (x, y, h) in ((0, 0, 0.26), (0.1, 0.06, 0.18), (-0.07, 0.09, 0.13)):
            tube(g, (x, y, 0), (x, y, h), 0.035, mat="M_Wax", n=7)
            tube(g, (x, y, h + 0.005), (x, y, h + 0.07), 0.018, 0.0, "M_FlameCore", n=5)
    elif kind == "bell":
        lathe(g, [(0.2, 0.0), (0.21, 0.02), (0.15, 0.12), (0.13, 0.28), (0.09, 0.33), (0.0, 0.34)], "M_Gold",
              n=12, cap_bottom=False)
        tube(g, (0, 0, 0.33), (0, 0, 0.4), 0.03, mat="M_Gold", n=6)
        ball(g, (0, 0, 0.05), 0.04, "M_Iron")
    elif kind == "crab":
        ball(g, (0, 0, 0.07), 0.12, "M_Crab", squash=(1.2, 0.85, 0.5))
        for s in (-1, 1):
            ball(g, (s * 0.04, -0.1, 0.13), 0.018, "M_Dark")
            tube(g, (s * 0.1, -0.06, 0.06), (s * 0.18, -0.15, 0.06), 0.022, mat="M_Crab", n=5)
            ball(g, (s * 0.2, -0.19, 0.06), 0.05, "M_Crab", squash=(0.8, 1.2, 0.7))
            for k in range(3):
                y = -0.03 + 0.05 * k
                tube(g, (s * 0.12, y, 0.06), (s * 0.24, y + 0.03, 0.01), 0.012, mat="M_Crab", n=4)
    elif kind == "tankard":
        lathe(g, [(0.07, 0), (0.08, 0.18)], "M_WoodLight", n=9, cap_top=False)
        for z in (0.02, 0.15):
            lathe(g, [(0.078, z), (0.085, z + 0.025)], "M_Iron", n=9)
        tube(g, (0.08, 0, 0.15), (0.13, 0, 0.13), 0.012, mat="M_Iron", n=4)
        tube(g, (0.13, 0, 0.13), (0.13, 0, 0.05), 0.012, mat="M_Iron", n=4)
        tube(g, (0.13, 0, 0.05), (0.08, 0, 0.04), 0.012, mat="M_Iron", n=4)
    elif kind == "flag":
        tube(g, (0, 0, -0.2), (0, 0, 1.5), 0.025, mat="M_WoodDark", n=6)
        v = [[0, 0, 1.45], [0.32, 0.03, 1.4], [0.6, -0.02, 1.47], [0, 0, 1.1], [0.3, 0.04, 1.08], [0.58, 0.0, 1.13]]
        g.add(v, [[0, 3, 4, 1], [1, 4, 5, 2]], "M_FlagBlack")
        ball(g, (0.25, 0.04, 1.29), 0.06, "M_Bone")
    else:
        raise ValueError(kind)
    return g


ITEM_RADIUS = {"anchor": 0.8, "cannon": 0.8, "ship_wheel": 0.7, "driftwood": 0.8, "barrel": 0.4, "crate": 0.45,
               "stump": 0.5, "cannonballs": 0.35, "small_chest": 0.35, "flag": 0.4}
ZONE_ITEMS = {
    "beach": ["shell", "starfish", "bottle", "coconut", "driftwood", "crab", "bucket", "rope_coil", "shell",
              "starfish", "crab", "small_chest", "flag"],
    "palms": ["coconut", "coconuts", "sack", "coconut", "bucket", "coconuts"],
    "temple": ["amphora", "urn", "clay_pot", "idol", "skull", "gems", "candles", "book", "scroll", "bell",
               "helmet", "sword", "coin_pouch"],
    "camp": ["barrel", "crate", "sack", "lantern", "bucket", "rum_bottle", "cooking_pot", "rope_coil",
             "coin_pouch", "scroll", "book", "small_chest", "shovel", "tankard"],
    "wreck": ["cannonballs", "anchor", "ship_wheel", "cannon", "barrel", "crate", "rum_bottle", "bottle",
              "helmet", "sword", "rope_coil", "bell", "lantern"],
    "arch": ["skull", "lantern", "gems", "mushrooms", "candles", "clay_pot", "sword", "coin_pouch"],
    "dock": ["crate", "barrel", "rope_coil", "bucket", "lantern", "sack"],
    "islet": ["coconut", "shell", "bottle", "starfish"],
    "lighthouse": ["lantern", "rope_coil", "bucket", "crate", "barrel", "bottle", "book", "tankard"],
    "hut": ["bucket", "rope_coil", "sack", "barrel", "crate", "coconut", "bottle", "lantern", "tankard"],
    "ruin": ["clay_pot", "urn", "book", "candles", "skull", "coin_pouch", "cooking_pot", "scroll", "small_chest"],
    "graveyard": ["skull", "candles", "shovel", "lantern", "sword", "helmet", "coin_pouch", "mushrooms"],
    "tower": ["barrel", "crate", "lantern", "sack", "cannonballs"],
    "well": ["bucket", "coin_pouch"],
    "shed": ["barrel", "crate", "rum_bottle", "sack", "rope_coil", "small_chest"],
    "inland": ["mushrooms", "stump", "sack", "clay_pot", "shovel", "sword", "helmet", "coconut", "lantern",
               "skull", "gems", "small_chest", "tankard", "flag", "amphora"],
}
# (zone, count, centre, radius, height range)
ZONE_PLAN = [
    ("beach", 36, None, None, (-0.15, 0.85)),
    ("palms", 19, None, None, (0.1, 3.5)),
    ("temple", 27, TEMPLE, 8.5, (0.0, 20.0)),
    ("camp", 17, CAMP, 6.0, (0.3, 6.0)),
    ("wreck", 18, None, 8.5, (-0.1, 4.0)),
    ("arch", 10, ARCH, 6.0, (0.5, 12.0)),
    ("dock", 6, None, None, None),
    ("islet", 5, ISLET, 3.4, (0.0, 3.0)),
    ("lighthouse", 7, LIGHTHOUSE, 5.5, (0.2, 6.0)),
    ("hut", 8, HUT, 5.0, (-0.1, 4.0)),
    ("ruin", 8, RUIN, 4.5, (0.3, 8.0)),
    ("graveyard", 6, GRAVEYARD, 3.4, (0.3, 8.0)),
    ("tower", 4, TOWER, 4.0, (0.3, 8.0)),
    ("well", 2, WELL, 2.4, (0.3, 8.0)),
    ("shed", 5, SHED, 4.0, (0.0, 4.0)),
    ("inland", 22, np.zeros(2), 26.0, (0.9, 20.0)),
]
SOLIDS = []  # (xy, radius) footprints props must keep clear of
TEMPLE_TIERS = ((4.5, 0.45), (3.6, 1.05), (2.8, 1.6))  # (half size, top z above TEMPLE_Z)


def temple_surface(p):
    """Height of the temple platform under p (None on stairs, pillars and altar)."""
    q = p - TEMPLE
    cheb = max(abs(q[0]), abs(q[1]))
    if q[1] < -2.4 and abs(q[0]) < 1.9:
        return None
    if abs(q[0]) < 1.05 and abs(q[1] - 0.7) < 0.8:
        return None
    for px_, py_ in [(-2.2, 2.2), (0, 2.2), (2.2, 2.2), (-2.2, 0), (2.2, 0), (-2.2, -2.2), (2.2, -2.2)]:
        if math.hypot(q[0] - px_, q[1] - py_) < 0.75:
            return None
    z = None
    for half, top in TEMPLE_TIERS:
        if cheb < half - 0.25:
            z = TEMPLE_Z + top
        elif cheb < half + 0.1:
            return None  # too close to a tier edge
    return z


def prop_spot(p, rad, h_range, taken, wreck_ob):
    """Ground height for a prop at p, or None if the spot is blocked."""
    q = p - TEMPLE
    if max(abs(q[0]), abs(q[1])) < 4.6:
        z = temple_surface(p)
    else:
        z = gz(p)
        if not (h_range[0] <= z <= h_range[1]):
            return None
        e = 0.6
        sl = math.hypot(gz(p + [e, 0]) - gz(p - [e, 0]), gz(p + [0, e]) - gz(p - [0, e])) / (2 * e)
        if sl > 0.5:
            return None
    if z is None:
        return None
    # wreck hull (in its own frame) and the dock corridor
    inv = Euler(wreck_ob.rotation_euler).to_matrix().inverted()
    lp = inv @ Vector((p[0] - wreck_ob.location.x, p[1] - wreck_ob.location.y, z - wreck_ob.location.z))
    if abs(lp.x) < 5.6 + rad and abs(lp.y) < 2.4 + rad:
        return None
    d = p - DOCK_BASE
    along, across = d @ DOCK_DIR, d @ np.array([-DOCK_DIR[1], DOCK_DIR[0]])
    if -0.5 < along < DOCK_LEN + 0.5 and abs(across) < 1.6 + rad:
        return None
    for c, r in SOLIDS:
        if math.hypot(*(p - c)) < r + rad:
            return None
    for c, r in taken:
        if math.hypot(*(p - c)) < r + rad + 0.35:
            return None
    if float(path_dist(np.array([p[0]]), np.array([p[1]]))[0]) < 0.7 + rad:
        return None
    return z


def build_treasure_props(col, palms, wreck_ob):
    rng = np.random.default_rng(2024)
    kinds = sorted({k for v in ZONE_ITEMS.values() for k in v})
    meshes = {}
    for k in kinds:
        ob = emit(f"_item_{k}", item_geo(k), col)
        meshes[k] = ob.data
        meshes[k].name = f"Item_{k}"
        bpy.data.objects.remove(ob)
    taken = []  # (xy, radius)
    placed = []  # (kind, zone, (x, y, z), yaw)
    for zone, count, centre, radius, h_range in ZONE_PLAN:
        pool = list(ZONE_ITEMS[zone])
        rng.shuffle(pool)
        got, tries = 0, 0
        while got < count and tries < 20000:
            tries += 1
            kind = pool[got % len(pool)]
            rad = ITEM_RADIUS.get(kind, 0.3)
            yaw = float(rng.uniform(0, 2 * np.pi))
            if zone == "dock":
                along = rng.uniform(1.0, DOCK_LEN - 0.4)
                across = rng.uniform(-0.7, 0.7)
                p = DOCK_BASE + DOCK_DIR * along + np.array([-DOCK_DIR[1], DOCK_DIR[0]]) * across
                if any(math.hypot(*(p - c)) < r + rad + 0.3 for c, r in taken + SOLIDS):
                    continue
                z = DECK_Z + 0.03
            else:
                if zone == "beach":
                    th = rng.uniform(-np.pi, np.pi)
                    p = shore_point(th, inland=rng.uniform(-0.6, 3.5))
                elif zone == "palms":
                    c = palms[rng.integers(len(palms))]
                    a = rng.uniform(0, 2 * np.pi)
                    p = c + rng.uniform(0.7, 1.6) * np.array([math.cos(a), math.sin(a)])
                else:
                    c = WRECK_POS if zone == "wreck" else centre
                    a = rng.uniform(0, 2 * np.pi)
                    p = c + radius * math.sqrt(rng.random()) * np.array([math.cos(a), math.sin(a)])
                z = prop_spot(p, rad, h_range, taken, wreck_ob)
                if z is None:
                    continue
                if kind in ("shell", "starfish", "crab") and zone not in ("beach", "islet"):
                    continue
            taken.append((p, rad))
            placed.append((kind, zone, (float(p[0]), float(p[1]), float(z)), yaw))
            got += 1
        print(f"[props] {zone:7s} {got}/{count}")
    for i, (kind, zone, loc, yaw) in enumerate(placed):
        ob = bpy.data.objects.new(f"Prop_{i + 1:03d}_{kind}", meshes[kind])
        ob.location = loc
        ob.rotation_euler = (0, 0, yaw)
        ob["kind"] = "treasure_prop"
        ob["item"] = kind
        ob["zone"] = zone
        ob["index"] = i + 1
        col.objects.link(ob)
    return placed, meshes


def render_prop_catalog(rig, meshes, path, samples):
    """Every item type lined up on a sandy board far away from the island, labelled."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    from bpy_extras.object_utils import world_to_camera_view
    cat = K.get_collection("Prop_Catalog")
    kinds = sorted(meshes)
    cols, sp = 7, 1.5
    rows = math.ceil(len(kinds) / cols)
    origin = np.array([300.0, -300.0, 0.0])
    g = Geo()
    w, h = cols * sp + 1, rows * sp + 1
    box(g, origin + [(cols - 1) * sp / 2, (rows - 1) * sp / 2, -0.05], (w, h, 0.1), "M_CatalogFloor")
    emit("Catalog_Board", g, cat)
    spots = []
    for i, k in enumerate(kinds):
        r, c = divmod(i, cols)
        loc = origin + [c * sp, (rows - 1 - r) * sp, 0]
        ob = bpy.data.objects.new(f"Catalog_{k}", meshes[k])
        ob.location = loc
        ob.rotation_euler = (0, 0, math.radians(-20))
        cat.objects.link(ob)
        spots.append((k, loc))
    centre = origin + [(cols - 1) * sp / 2, (rows - 1) * sp / 2 - 0.4, 0.2]
    cam = camera(rig, "Cam_Catalog", centre + [0, -11.5, 9.0], centre, lens=36)
    res = (1600, 1200)
    render(cam, path, res, samples)
    scene = bpy.context.scene
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
    except OSError:
        f = ImageFont.load_default()
    for k, loc in spots:
        v = world_to_camera_view(scene, cam, Vector(loc - [0, 0.55, 0]))
        x, y = v.x * res[0], (1 - v.y) * res[1]
        tw = d.textlength(k, font=f)
        d.rectangle([x - tw / 2 - 5, y - 2, x + tw / 2 + 5, y + 22], fill=(255, 248, 230))
        d.text((x - tw / 2, y), k, font=f, fill=(60, 40, 20))
    im.save(path)
    print("[catalog]", path)


# ---------------------------------------------------------------------------
# more structures: lighthouse, fisherman's hut, ruined house, well, graveyard,
# watchtower and the smugglers' shed
# ---------------------------------------------------------------------------
# These are built in world coordinates (object origin at 0) so every post, stilt
# and wall course can be set on the ground under it.

class Site:
    """A local frame on the ground: +y is `front`, +x is to its right."""

    def __init__(self, c, front):
        self.c = np.asarray(c, float)
        f = np.asarray(front, float)
        self.f = f / np.linalg.norm(f)
        self.x = np.array([self.f[1], -self.f[0]])
        self.yaw = math.atan2(self.x[1], self.x[0])

    def w(self, lx, ly):
        return self.c + lx * self.x + ly * self.f

    def w3(self, lx, ly, z):
        p = self.w(lx, ly)
        return np.array([p[0], p[1], z])

    def g(self, lx, ly):
        return gz(self.w(lx, ly))


def plank_wall(g, site, a, b, z0, height, mat="M_Wood", door=None, window=None, plank=0.3):
    """Vertical planks from local a to b; door / window are (s0, s1) ranges along the wall."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    d = (b - a) / L
    wd = d[0] * site.x + d[1] * site.f
    wyaw = math.atan2(wd[1], wd[0])
    n = max(int(round(L / plank)), 1)
    for i in range(n):
        s = (i + 0.5) * L / n
        p = a + d * s
        if door and door[0] <= s <= door[1]:
            continue
        spans = [(0, height)]
        if window and window[0] <= s <= window[1]:
            spans = [(0, height * 0.45), (height * 0.78, height)]
        for h0, h1 in spans:
            box(g, site.w3(p[0], p[1], z0 + (h0 + h1) / 2), (L / n - 0.02, 0.08, h1 - h0),
                "M_Wood" if (i % 3) else "M_WoodLight", rot=(0, 0, wyaw))
            SOLIDS.append((site.w(p[0], p[1]), 0.2))


def pyramid_roof(g, site, half, z_eave, z_apex, mat="M_Thatch", cy=0.0, layers=3):
    """Thatched hip roof in overlapping layers with ragged, thick edges and a bound top knot."""
    m = 7
    sq = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    def ring(hf, z, jag):
        pts = []
        for i in range(4):
            (ax_, ay_), (bx_, by_) = sq[i], sq[(i + 1) % 4]
            for j in range(m):
                t = j / m
                lx, ly = (ax_ + (bx_ - ax_) * t) * hf, (ay_ + (by_ - ay_) * t) * hf
                pts.append(site.w3(lx, cy + ly, z - (jag if j % 2 else 0.0)))
        return np.array(pts)

    n = 4 * m
    alt = "M_ThatchDark" if mat == "M_Thatch" else mat
    for k in range(layers):
        t0, t1 = k / layers, min(1.0, (k + 1) / layers + 0.12)
        hf0, hf1 = half * (1 - t0) + 0.04, max(half * (1 - t1), 0.06)
        z0 = z_eave + (z_apex - z_eave) * t0 - 0.06 * k
        z1 = z_eave + (z_apex - z_eave) * t1
        outer, inner, lip = ring(hf0, z0, 0.06), ring(hf1, z1, 0.0), ring(hf0 - 0.05, z0 - 0.16, 0.06)
        v = np.concatenate([outer, inner, lip])
        f = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        f += [[2 * n + i, 2 * n + (i + 1) % n, (i + 1) % n, i] for i in range(n)]
        f += [[2 * n + (i + 1) % n, 2 * n + i, n + i, n + (i + 1) % n] for i in range(n)]
        g.add(v, f, mat if k % 2 == 0 else alt)
    apex = site.w3(0, cy, z_apex)
    ball(g, apex + [0, 0, 0.02], 0.16, alt, subdiv=1, squash=(1, 1, 1.4))
    tube(g, apex - [0, 0, 0.1], apex + [0, 0, 0.12], 0.13, 0.13, "M_Rope", n=8)


def ladder(g, bottom, top, width=0.6, mat="M_WoodDark"):
    bottom, top = np.asarray(bottom, float), np.asarray(top, float)
    a = top - bottom
    side = np.cross(a, [0, 0, 1.0])
    side = side / np.linalg.norm(side) * width / 2
    for s in (-1, 1):
        tube(g, bottom + s * side, top + s * side + a / np.linalg.norm(a) * 0.3, 0.035, mat=mat, n=5)
    n = int(np.linalg.norm(a) / 0.32)
    for k in range(1, n + 1):
        p = bottom + a * k / (n + 1)
        tube(g, p - side, p + side, 0.025, mat=mat, n=5)




def build_lighthouse(col, lights):
    g = Geo()
    c = LIGHTHOUSE
    z0 = gz(c)
    a_in = math.atan2(-c[1], -c[0])
    H0, H1 = z0 + 0.4, z0 + 7.4

    def r_at(z):
        return 1.45 - 0.45 * (z - H0) / (H1 - H0)

    def at(r, a, z):
        return np.array([c[0] + r * math.cos(a), c[1] + r * math.sin(a), z])

    tube(g, (c[0], c[1], z0 - 1.5), (c[0], c[1], H0), 2.0, 1.9, "M_StoneDark", n=12)
    bands = 6
    for k in range(bands):
        za, zb = H0 + (H1 - H0) * k / bands, H0 + (H1 - H0) * (k + 1) / bands
        tube(g, (c[0], c[1], za), (c[0], c[1], zb), r_at(za), r_at(zb),
             "M_LighthouseWhite" if k % 2 == 0 else "M_LighthouseRed", n=12)
    for k in range(1, 13):  # mortar courses
        zz = H0 + (H1 - H0) * k / 13
        tube(g, (c[0], c[1], zz - 0.015), (c[0], c[1], zz + 0.015), r_at(zz) + 0.012, mat="M_Mortar", n=12)
    tan = lambda aa: np.array([-math.sin(aa), math.cos(aa), 0.0])  # noqa: E731

    def framed(aa, zz, w, h, fill):
        base = at(r_at(zz) - 0.02, aa, zz)
        rot = (0, 0, aa + math.pi / 2)
        box(g, base, (w, 0.22, h), fill, rot=rot)
        out = at(0.06, aa, 0) - np.array([c[0], c[1], 0])
        box(g, base + out + [0, 0, h / 2 + 0.04], (w + 0.2, 0.2, 0.09), "M_StoneLight", rot=rot)
        box(g, base + out - [0, 0, h / 2 + 0.04], (w + 0.24, 0.24, 0.08), "M_StoneLight", rot=rot)
        for sd in (-1, 1):
            box(g, base + out + tan(aa) * sd * (w / 2 + 0.05), (0.09, 0.2, h), "M_StoneLight", rot=rot)

    framed(a_in, H0 + 0.95, 0.9, 1.9, "M_WoodDark")
    for k in range(4):  # door planks and studs
        q = at(r_at(H0 + 1) + 0.02, a_in, H0 + 0.95) + tan(a_in) * (-0.33 + 0.22 * k)
        box(g, q, (0.2, 0.06, 1.85), "M_Wood", rot=(0, 0, a_in + math.pi / 2))
    ball(g, at(r_at(H0 + 1) + 0.08, a_in, H0 + 0.95) + tan(a_in) * 0.3, 0.04, "M_Brass", subdiv=1)
    for k in range(3):  # door steps
        box(g, at(r_at(H0) + 0.35 + 0.3 * k, a_in, H0 - 0.15 - 0.2 * k), (1.3, 0.35, 0.2), "M_Stone",
            rot=(0, 0, a_in + math.pi / 2))
    for zz, aa in ((H0 + 3.0, a_in + 2.2), (H0 + 5.0, a_in - 1.3), (H0 + 4.0, a_in + math.pi)):
        framed(aa, zz, 0.35, 0.6, "M_Dark")
    # gallery, railing, lamp room, roof
    tube(g, (c[0], c[1], H1 - 0.18), (c[0], c[1], H1), 1.9, mat="M_StoneDark", n=14)
    for k in range(16):  # corbels under the gallery
        a = 2 * np.pi * k / 16
        box(g, at(1.35, a, H1 - 0.38), (0.5, 0.18, 0.3), "M_StoneLight", rot=(0, 0, a))
    tube(g, (c[0], c[1], H1), (c[0], c[1], H1 + 0.25), 0.85, mat="M_StoneDark", n=8)
    tube(g, (c[0], c[1], H1 + 0.25), (c[0], c[1], H1 + 1.35), 0.72, mat="M_LanternGlow", n=8)
    for k in range(8):
        a = 2 * np.pi * k / 8 + np.pi / 8
        tube(g, at(0.74, a, H1 + 0.25), at(0.74, a, H1 + 1.35), 0.035, mat="M_Iron", n=4)
    tube(g, (c[0], c[1], H1 + 1.35), (c[0], c[1], H1 + 1.5), 0.98, mat="M_LighthouseRed", n=10)
    tube(g, (c[0], c[1], H1 + 1.5), (c[0], c[1], H1 + 2.4), 0.98, 0.0, "M_LighthouseRed", n=10)
    ball(g, (c[0], c[1], H1 + 2.5), 0.12, "M_Iron")
    ball(g, (c[0], c[1], H1 + 0.8), 0.38, "M_LanternGlow", subdiv=2)  # the lamp itself
    for zz in (H1 + 0.27, H1 + 1.33):
        tube(g, (c[0], c[1], zz - 0.03), (c[0], c[1], zz + 0.03), 0.76, mat="M_Brass", n=8)
    tube(g, (c[0], c[1], H1 + 2.5), (c[0], c[1], H1 + 3.1), 0.025, mat="M_Iron", n=5)  # weather vane
    beam(g, (c[0] - 0.45, c[1], H1 + 3.0), (c[0] + 0.45, c[1], H1 + 3.0), 0.02, 0.02, "M_Iron")
    g.add([(c[0] + 0.45, c[1], H1 + 3.12), (c[0] + 0.62, c[1], H1 + 3.0), (c[0] + 0.45, c[1], H1 + 2.88)],
          [[0, 1, 2]], "M_Iron")
    g.add([(c[0] - 0.45, c[1], H1 + 3.12), (c[0] - 0.3, c[1], H1 + 3.0), (c[0] - 0.45, c[1], H1 + 2.88)],
          [[0, 1, 2]], "M_Iron")
    lights.append(np.array([c[0], c[1], H1 + 0.8]))
    # spiral stair around the outside, arriving at the gallery edge
    N = 26
    rise = (H1 - H0) / N
    a0 = a_in + 0.75
    da = 0.42
    for k in range(N):
        zt = H0 + (k + 1) * rise
        a = a0 + k * da
        box(g, at(2.35, a, zt - 0.06), (0.9, 0.55, 0.12), "M_Wood" if k % 2 else "M_WoodLight", rot=(0, 0, a))
        beam(g, at(r_at(zt) - 0.05, a, zt - 0.2), at(1.95, a, zt - 0.12), 0.08, 0.08, "M_Iron")
    a_top = a0 + (N - 1) * da
    for k in range(18):
        a = 2 * np.pi * k / 18
        if abs(wrap(a - a_top)) < 0.4:
            continue
        tube(g, at(1.82, a, H1), at(1.82, a, H1 + 0.95), 0.03, mat="M_Iron", n=4)
        b = 2 * np.pi * (k + 1) / 18
        if abs(wrap(b - a_top)) >= 0.4:
            beam(g, at(1.82, a, H1 + 0.92), at(1.82, b, H1 + 0.92), 0.05, 0.05, "M_Iron")
    for k in range(N - 1):  # outer stair rail
        a, b = a0 + k * da, a0 + (k + 1) * da
        beam(g, at(2.78, a, H0 + (k + 1) * rise + 0.9), at(2.78, b, H0 + (k + 2) * rise + 0.9), 0.05, 0.05,
             "M_Iron")
        tube(g, at(2.78, a, H0 + (k + 1) * rise), at(2.78, a, H0 + (k + 1) * rise + 0.9), 0.025, mat="M_Iron", n=4)
    i = [k for k, (n, _) in enumerate(TREASURE_SPAWNS) if n == "Lighthouse gallery"][0]
    p = at(1.3, a_top + math.pi, H1)
    TREASURE_SPAWNS[i] = ("Lighthouse gallery", p[:2])
    TREASURE_Z[i] = H1
    SOLIDS.append((c, 2.9))
    return emit("Lighthouse", g, col)


def build_fisher_hut(col):
    g = Geo()
    s = Site(HUT, -HUT)
    pz = max(s.g(x, y) for x in (-1.6, 1.6) for y in (-1.6, 2.6)) + 0.95
    box(g, s.w3(0, 0.5, pz - 0.05), (3.4, 4.3, 0.1), "M_WoodLight", rot=(0, 0, s.yaw))
    for lx in (-1.55, 1.55):
        for ly in (-1.55, 0.5, 2.55):
            tube(g, s.w3(lx, ly, s.g(lx, ly) - 0.4), s.w3(lx, ly, pz - 0.1), 0.1, mat="M_WoodDark", n=6)
            SOLIDS.append((s.w(lx, ly), 0.25))
    plank_wall(g, s, (-1.6, -1.6), (1.6, -1.6), pz, 2.0, window=(1.0, 2.2))
    plank_wall(g, s, (1.6, -1.6), (1.6, 1.6), pz, 2.0, window=(1.2, 2.0))
    plank_wall(g, s, (1.6, 1.6), (-1.6, 1.6), pz, 2.0, door=(1.15, 2.05))
    plank_wall(g, s, (-1.6, 1.6), (-1.6, -1.6), pz, 2.0, window=(1.2, 2.0))
    pyramid_roof(g, s, 2.15, pz + 1.95, pz + 3.3)
    for lx in (-1.55, 1.55):
        tube(g, s.w3(lx, 2.55, pz), s.w3(lx, 2.55, pz + 1.0), 0.06, mat="M_WoodDark", n=5)
        beam(g, s.w3(lx, 1.6, pz + 0.95), s.w3(lx, 2.55, pz + 0.95), 0.07, 0.07, "M_WoodDark")
    ladder(g, s.w3(0, 3.7, s.g(0, 3.7) - 0.05), s.w3(0, 2.62, pz + 0.05))
    SOLIDS.append((s.w(0, 3.6), 0.5))
    # fishing net on a drying rack, and a boat turned over on the sand
    for ly in (-0.7, 1.7):
        tube(g, s.w3(2.9, ly, s.g(2.9, ly) - 0.3), s.w3(2.9, ly, s.g(2.9, ly) + 1.9), 0.06, mat="M_WoodDark", n=5)
        SOLIDS.append((s.w(2.9, ly), 0.3))
    zr = s.g(2.9, 0.5)
    beam(g, s.w3(2.9, -0.8, zr + 1.8), s.w3(2.9, 1.8, zr + 1.8), 0.08, 0.08, "M_WoodDark")
    for k in range(9):
        ly = -0.6 + 2.2 * k / 8
        beam(g, s.w3(2.9, ly, zr + 1.78), s.w3(2.92, ly + 0.05, zr + 0.45), 0.02, 0.02, "M_Rope")
    for k in range(5):
        z = zr + 0.55 + 0.28 * k
        beam(g, s.w3(2.91, -0.6, z), s.w3(2.91, 1.6, z), 0.02, 0.02, "M_Rope")
    boat, _ = hull_geo(3.0, 1.15, 0.5, 9, 7, mat="M_WoodLight")
    p = s.w(-3.2, 0.6)
    g.merge(boat, loc=(p[0], p[1], gz(p) + 0.42), rot=(math.pi, 0, s.yaw + math.pi / 2 + 0.15))
    SOLIDS.extend([(s.w(-3.2, -0.6), 0.8), (s.w(-3.2, 0.6), 0.8), (s.w(-3.2, 1.8), 0.8)])
    i = [k for k, (n, _) in enumerate(TREASURE_SPAWNS) if n == "Fisherman's hut"][0]
    TREASURE_SPAWNS[i] = ("Fisherman's hut", s.w(-0.8, -0.9))
    TREASURE_Z[i] = pz
    return emit("Fisherman_Hut", g, col)


def build_ruined_house(col):
    g = Geo()
    s = Site(RUIN, WELL - RUIN)
    zc = gz(RUIN)
    base = zc - 0.8
    ch, bl, th = 0.42, 0.85, 0.45
    corner = np.array([3.0, -2.25])
    rng = np.random.default_rng(31)

    def height(lx, ly):
        d = math.hypot(lx - corner[0], ly - corner[1])
        return 0.5 + 2.2 * float(ss(0.6, 4.8, d)) + 0.25 * float(fbm(lx * 0.9, ly * 0.9, 2, seed=8))

    walls = [((-3.0, -2.25), (3.0, -2.25), None, None),        # back
             ((-3.0, 2.25), (3.0, 2.25), (-0.55, 0.55), None),  # front, with the doorway
             ((-3.0, -1.8), (-3.0, 1.8), None, (-0.5, 0.5)),    # left, with a window
             ((3.0, -1.8), (3.0, 1.8), None, None)]             # right
    for a, b, door, window in walls:
        a, b = np.array(a), np.array(b)
        L = np.linalg.norm(b - a)
        d = (b - a) / L
        wd = d[0] * s.x + d[1] * s.f
        wyaw = math.atan2(wd[1], wd[0])
        for k in range(int(3.0 / ch)):
            z_bot = base + k * ch
            off = 0.5 * bl if k % 2 else 0.0
            t = -off
            while t < L - 0.05:
                t0, t1 = max(t, 0), min(t + bl, L)
                t += bl
                m = a + d * (t0 + t1) / 2
                along = (m[0] if abs(d[0]) > 0.5 else m[1])
                above = z_bot + ch / 2 - zc
                if above > height(*m):
                    continue
                if door and door[0] - 0.2 < along < door[1] + 0.2 and 0 < above + ch / 2 and above < 2.0:
                    continue
                if window and window[0] - 0.2 < along < window[1] + 0.2 and 0.9 < above < 1.8:
                    continue
                box(g, s.w3(m[0], m[1], z_bot + ch / 2), (t1 - t0 - 0.04, th, ch - 0.03),
                    "M_Stone" if (k + int(t / bl)) % 3 else "M_StoneDark", rot=(0, 0, wyaw))
        for t in np.arange(0.3, L, 0.7):
            p = a + d * t
            if not (door and door[0] - 0.1 < (p[0] if abs(d[0]) > 0.5 else p[1]) < door[1] + 0.1):
                SOLIDS.append((s.w(*p), 0.38))
    # door lintel, rubble at the collapsed corner, fallen roof beams
    box(g, s.w3(0, 2.25, zc + 2.05), (1.6, 0.5, 0.16), "M_WoodDark", rot=(0, 0, s.yaw))
    for k in range(10):
        p = corner + rng.uniform(-1.6, 1.6, 2) * [1, 0.8] + [-0.6, 0.6]
        sz = rng.uniform(0.3, 0.55)
        box(g, s.w3(p[0], p[1], s.g(*p) + sz * 0.25), (sz * 1.6, sz, sz * 0.8), "M_StoneDark",
            rot=(rng.uniform(-.3, .3), rng.uniform(-.3, .3), s.yaw + rng.uniform(0, 3)))
        SOLIDS.append((s.w(*p), 0.45))
    for (ax_, ay_), (bx_, by_) in (((-2.8, -2.0), (0.6, -0.6)), ((-2.8, 1.9), (1.2, 0.2))):
        beam(g, s.w3(ax_, ay_, zc + 2.3), s.w3(bx_, by_, s.g(bx_, by_) + 0.1), 0.16, 0.16, "M_WoodDark")
        SOLIDS.append((s.w(bx_, by_), 0.4))
    i = [k for k, (n, _) in enumerate(TREASURE_SPAWNS) if n == "Ruined house"][0]
    TREASURE_SPAWNS[i] = ("Ruined house", s.w(-2.0, -1.3))
    return emit("Ruined_House", g, col)


def build_well(col):
    g = Geo()
    c = WELL
    z = gz(c)
    for k in range(12):
        a = 2 * np.pi * k / 12
        box(g, (c[0] + 0.75 * math.cos(a), c[1] + 0.75 * math.sin(a), z + 0.15), (0.3, 0.42, 1.1),
            "M_Stone" if k % 2 else "M_StoneDark", rot=(0, 0, a))
    tube(g, (c[0], c[1], z + 0.15), (c[0], c[1], z + 0.2), 0.62, mat="M_WellWater", n=12)
    for sx in (-1, 1):
        tube(g, (c[0] + sx * 0.95, c[1], z - 0.3), (c[0] + sx * 0.95, c[1], z + 1.95), 0.07, mat="M_WoodDark", n=6)
    tube(g, (c[0] - 1.05, c[1], z + 1.55), (c[0] + 1.05, c[1], z + 1.55), 0.06, mat="M_Wood", n=6)
    tube(g, (c[0] + 1.05, c[1], z + 1.55), (c[0] + 1.05, c[1] - 0.3, z + 1.35), 0.03, mat="M_Iron", n=4)
    for sy in (-1, 1):
        beam(g, (c[0], c[1], z + 2.35), (c[0], c[1] + sy * 0.95, z + 1.75), 2.3, 0.06, "M_WoodDark")
    tube(g, (c[0], c[1], z + 1.5), (c[0], c[1], z + 1.0), 0.012, mat="M_Rope", n=4)
    g.merge(item_geo("bucket"), loc=(c[0], c[1], z + 0.55), scale=0.85)
    SOLIDS.append((c, 1.15))
    return emit("Well", g, col)


def build_graveyard(col):
    g = Geo()
    s = Site(GRAVEYARD, np.array([12.5, 9.8]) - GRAVEYARD)
    rng = np.random.default_rng(41)
    W, D = 2.6, 2.0
    ring = [(-W, -D), (W, -D), (W, D), (0.6, D), (-0.6, D), (-W, D)]
    segs = [(ring[0], ring[1]), (ring[1], ring[2]), (ring[2], ring[3]), (ring[4], ring[5]), (ring[5], ring[0])]
    for a, b in segs:
        a, b = np.array(a), np.array(b)
        n = max(int(np.linalg.norm(b - a) / 0.85), 1)
        pts = [a + (b - a) * k / n for k in range(n + 1)]
        for k, p in enumerate(pts):
            gate = abs(p[0]) < 0.7 and p[1] > 0
            tube(g, s.w3(p[0], p[1], s.g(*p) - 0.3), s.w3(p[0], p[1], s.g(*p) + (1.3 if gate else 0.95)),
                 0.05 if not gate else 0.08, mat="M_WoodDark", n=5)
            SOLIDS.append((s.w(*p), 0.2))
            if k:
                q = pts[k - 1]
                for h in (0.4, 0.8):
                    beam(g, s.w3(q[0], q[1], s.g(*q) + h), s.w3(p[0], p[1], s.g(*p) + h), 0.05, 0.08, "M_Wood")
    for lx, ly in ((-1.7, -1.1), (-0.5, -1.2), (0.7, -1.1), (1.8, -1.0)):
        z = s.g(lx, ly)
        rot = (rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), s.yaw + rng.uniform(-0.15, 0.15))
        box(g, s.w3(lx, ly, z + 0.3), (0.55, 0.16, 0.9), "M_StoneDark", rot=rot)
        box(g, s.w3(lx, ly, z + 0.78), (0.42, 0.14, 0.12), "M_StoneDark", rot=rot)
        SOLIDS.append((s.w(lx, ly), 0.4))
    for lx, ly in ((-1.9, 0.9), (2.0, 1.2)):
        z = s.g(lx, ly)
        p0, p1 = s.w3(lx, ly, z - 0.2), s.w3(lx, ly, z + 1.1)
        beam(g, p0, p1 + [0.0, 0.0, 0.0], 0.08, 0.08, "M_WoodLight")
        beam(g, s.w3(lx - 0.3, ly, z + 0.8), s.w3(lx + 0.3, ly, z + 0.8), 0.07, 0.07, "M_WoodLight")
        SOLIDS.append((s.w(lx, ly), 0.3))
    zg = s.g(0.3, 0.6)
    box(g, s.w3(0.3, 0.6, zg - 0.2), (1.8, 0.8, 0.44), "M_Dark", rot=(0, 0, s.yaw))
    rock(g, s.w3(0.4, 1.35, zg), (0.95, 0.35, 0.35), "M_Dirt", seed=77, yaw=s.yaw, rough=0.1, sink=0.5)
    g.merge(item_geo("shovel"), loc=s.w3(1.45, 0.3, zg), rot=(0, 0, s.yaw))
    SOLIDS.extend([(s.w(-0.3, 0.6), 0.5), (s.w(0.9, 0.6), 0.5), (s.w(0.4, 1.35), 0.6), (s.w(1.45, 0.3), 0.3)])
    # a dead tree in the corner
    t = s.w3(-2.0, -0.1, s.g(-2.0, -0.1))
    pts = [t + [0, 0, -0.2], t + [0.1, 0.05, 1.0], t + [-0.05, 0.12, 1.9], t + [0.2, 0.0, 2.7]]
    for k in range(3):
        tube(g, pts[k], pts[k + 1], 0.16 - 0.04 * k, 0.12 - 0.035 * k, "M_DeadWood", n=6)
    for p0, d, L in ((pts[1], (0.8, 0.3, 0.6), 1.0), (pts[2], (-0.7, -0.2, 0.7), 0.9), (pts[2], (0.3, -0.8, 0.5), 0.8)):
        d = np.array(d) / np.linalg.norm(d)
        tube(g, p0, p0 + d * L, 0.07, 0.02, "M_DeadWood", n=5)
    SOLIDS.append((s.w(-2.0, -0.1), 0.45))
    i = [k for k, (n, _) in enumerate(TREASURE_SPAWNS) if n == "Graveyard, by the open grave"][0]
    TREASURE_SPAWNS[i] = ("Graveyard, by the open grave", s.w(-0.9, -0.15))
    return emit("Graveyard", g, col)


def build_watchtower(col):
    g = Geo()
    s = Site(TOWER, -TOWER)
    pz = gz(TOWER) + 4.2
    for sx in (-1, 1):
        for sy in (-1, 1):
            b = s.w3(sx * 1.15, sy * 1.15, s.g(sx * 1.15, sy * 1.15) - 0.4)
            t = s.w3(sx * 0.98, sy * 0.98, pz + 2.25)
            tube(g, b, t, 0.12, 0.1, "M_WoodDark", n=6)
            SOLIDS.append((s.w(sx * 1.15, sy * 1.15), 0.3))
    for (ax_, ay_), (bx_, by_) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
        za, zb = s.g(ax_ * 1.12, ay_ * 1.12) + 0.5, s.g(bx_ * 1.12, by_ * 1.12) + 0.5
        beam(g, s.w3(ax_ * 1.12, ay_ * 1.12, za), s.w3(bx_ * 1.0, by_ * 1.0, pz - 0.3), 0.08, 0.1, "M_Wood")
        beam(g, s.w3(bx_ * 1.12, by_ * 1.12, zb), s.w3(ax_ * 1.0, ay_ * 1.0, pz - 0.3), 0.08, 0.1, "M_Wood")
    box(g, s.w3(0, 0, pz - 0.06), (2.6, 2.6, 0.12), "M_WoodLight", rot=(0, 0, s.yaw))
    for (ax_, ay_), (bx_, by_) in (((-1.25, -1.25), (-0.4, -1.25)), ((0.4, -1.25), (1.25, -1.25)),
                                   ((1.25, -1.25), (1.25, 1.25)), ((1.25, 1.25), (-1.25, 1.25)),
                                   ((-1.25, 1.25), (-1.25, -1.25))):
        for h in (0.5, 1.0):
            beam(g, s.w3(ax_, ay_, pz + h), s.w3(bx_, by_, pz + h), 0.06, 0.08, "M_Wood")
    for lx in (-0.4, 0.4):
        tube(g, s.w3(lx, -1.25, pz), s.w3(lx, -1.25, pz + 1.05), 0.05, mat="M_WoodDark", n=5)
    pyramid_roof(g, s, 1.65, pz + 2.2, pz + 3.25)
    ladder(g, s.w3(0, -2.15, s.g(0, -2.15) - 0.05), s.w3(0, -1.3, pz + 0.05))
    SOLIDS.append((s.w(0, -2.1), 0.5))
    tube(g, s.w3(0, 0, pz + 3.2), s.w3(0, 0, pz + 4.1), 0.03, mat="M_WoodDark", n=4)
    g.add([s.w3(0, 0, pz + 4.05), s.w3(0.55, 0.05, pz + 3.95), s.w3(0, 0, pz + 3.75)], [[0, 1, 2]], "M_ClothRed")
    i = [k for k, (n, _) in enumerate(TREASURE_SPAWNS) if n == "Watchtower platform"][0]
    TREASURE_SPAWNS[i] = ("Watchtower platform", s.w(0.3, 0.35))
    TREASURE_Z[i] = pz
    return emit("Watchtower", g, col)


def build_shed(col):
    g = Geo()
    s = Site(SHED, -SHED)
    z = max(s.g(x, y) for x in (-1.6, 1.6) for y in (-1.2, 1.2))
    box(g, s.w3(0, 0, z - 0.1), (3.4, 2.6, 0.2), "M_WoodLight", rot=(0, 0, s.yaw))
    plank_wall(g, s, (-1.6, -1.2), (1.6, -1.2), z, 2.4)
    plank_wall(g, s, (-1.6, 1.2), (-1.6, -1.2), z, 2.1)
    plank_wall(g, s, (1.6, -1.2), (1.6, 1.2), z, 2.1, window=(0.8, 1.6))
    for lx in (-1.6, 1.6):
        tube(g, s.w3(lx, 1.35, z - 0.3), s.w3(lx, 1.35, z + 2.05), 0.08, mat="M_WoodDark", n=6)
        SOLIDS.append((s.w(lx, 1.35), 0.25))
    for k in range(9):
        lx = -1.9 + 3.8 * (k + 0.5) / 9
        beam(g, s.w3(lx, -1.5, z + 2.5), s.w3(lx, 1.7, z + 2.0), 0.4, 0.05, "M_WoodDark" if k % 2 else "M_Wood")
    for k, (lx, ly) in enumerate(((-1.1, -0.7), (-0.5, -0.75), (-1.15, -0.1))):
        barrel(g, s.w3(lx, ly, z), seed=k * 0.7)
        SOLIDS.append((s.w(lx, ly), 0.35))
    crate(g, s.w3(1.0, -0.65, z), 0.65, s.yaw)
    crate(g, s.w3(1.0, -0.65, z + 0.65), 0.5, s.yaw + 0.3)
    crate(g, s.w3(1.05, 0.1, z), 0.5, s.yaw - 0.2)
    SOLIDS.extend([(s.w(1.0, -0.65), 0.45), (s.w(1.05, 0.1), 0.35)])
    i = [k for k, (n, _) in enumerate(TREASURE_SPAWNS) if n == "Smuggler's shed"][0]
    TREASURE_SPAWNS[i] = ("Smuggler's shed", s.w(-0.1, 0.35))
    TREASURE_Z[i] = z
    return emit("Smugglers_Shed", g, col)


# ---------------------------------------------------------------------------
# preview: lights, cameras, renders
# ---------------------------------------------------------------------------

def setup_world(rig, fire_points):
    scene = bpy.context.scene
    world = bpy.data.worlds.new("SunsetSky")
    scene.world = world
    if not world.node_tree:
        world.use_nodes = True
    nt = world.node_tree
    bg = next(n for n in nt.nodes if n.type == 'BACKGROUND')
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    els = ramp.color_ramp.elements
    els[0].position, els[1].position = 0.02, 0.3
    els[0].color = (*srgb_to_linear(hexc("#ffcf9e")), 1)
    els[1].color = (*srgb_to_linear(hexc("#5f93d6")), 1)
    bg.inputs["Strength"].default_value = 0.9

    sd = bpy.data.lights.new("Sun", 'SUN')
    sd.energy = 4.2
    sd.angle = math.radians(2.5)
    sd.color = tuple(srgb_to_linear(hexc("#ffe0b8")))
    sun = bpy.data.objects.new("Sun", sd)
    sun.rotation_euler = (math.radians(55), 0, math.radians(-35))
    rig.objects.link(sun)
    for k, p in enumerate(fire_points):
        ld = bpy.data.lights.new(f"FireLight_{k}", 'POINT')
        ld.energy = 140
        ld.shadow_soft_size = 0.2
        ld.color = tuple(srgb_to_linear(hexc("#ff9a4a")))
        ob = bpy.data.objects.new(f"FireLight_{k}", ld)
        ob.location = p
        rig.objects.link(ob)


def camera(rig, name, loc, target, lens=35.0, ortho=None):
    cd = bpy.data.cameras.new(name)
    if ortho:
        cd.type = 'ORTHO'
        cd.ortho_scale = ortho
    else:
        cd.lens = lens
    cd.clip_start = 0.1
    cd.clip_end = 600
    ob = bpy.data.objects.new(name, cd)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    rig.objects.link(ob)
    return ob


def render(cam, path, res, samples):
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    try:
        scene.view_settings.view_transform = 'AgX'
    except TypeError:
        scene.view_settings.view_transform = 'Filmic'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print("[render]", path)


# ---------------------------------------------------------------------------
# parchment treasure map (top-down layout drawn like an old pirate chart)
# ---------------------------------------------------------------------------

def draw_parchment_map(path, spawns, players, palms, props=None):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[map] Pillow not available, skipping the parchment map")
        return
    S, E = 1400, 40.0
    xs = np.linspace(-E, E, S)
    X, Y = np.meshgrid(xs, xs[::-1])
    H = ground(X, Y)

    def px(p):
        return ((p[0] + E) / (2 * E) * (S - 1), (E - p[1]) / (2 * E) * (S - 1))

    noise = fbm(X / 2.0, Y / 2.0, 4, seed=9)
    paper = hexc("#efdcac")[None, None, :] * (0.93 + 0.07 * noise[..., None])
    land = H > 0
    sea_col = hexc("#d9c995")[None, None, :] * (0.95 + 0.05 * noise[..., None])
    img = np.where(land[..., None], paper, sea_col)
    shallow = (H <= 0) & (H > -1.0)
    img = np.where(shallow[..., None], img * 1.03, img)
    # hill shading from the north-west
    gy, gx = np.gradient(H)
    shade = np.clip(1 + 0.9 * (gx * -0.7 + gy * 0.7), 0.82, 1.12)
    img = np.where(land[..., None], img * shade[..., None], img)
    # contour every 1.2 m, coastline in heavy ink
    q = np.floor(np.where(land, H, 0) / 1.2)
    contour = (np.diff(q, axis=0, prepend=q[:1]) != 0) | (np.diff(q, axis=1, prepend=q[:, :1]) != 0)
    img = np.where((contour & land)[..., None], hexc("#9c7c4c")[None, None, :], img)
    coast = (np.diff(land.astype(int), axis=0, prepend=land[:1].astype(int)) != 0) | \
            (np.diff(land.astype(int), axis=1, prepend=land[:, :1].astype(int)) != 0)
    coast = K.box_blur(coast.astype(np.float32), 1, 1) > 0.05
    img = np.where(coast[..., None], hexc("#3f2c1a")[None, None, :], img)
    # burnt edges
    r = np.maximum(abs(X), abs(Y)) / E
    burn = ss(0.82, 1.0, r + 0.06 * noise)
    img = img * (1 - 0.55 * burn[..., None]) + hexc("#6b4422")[None, None, :] * 0.55 * burn[..., None] * 0.6
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    try:
        title_f, label_f, small_f = (ImageFont.truetype(font_path, s) for s in (54, 26, 20))
    except OSError:
        title_f = label_f = small_f = ImageFont.load_default()
    ink, red = (63, 44, 26), (178, 40, 30)

    # waves in the sea
    rng = np.random.default_rng(12)
    for _ in range(140):
        p = rng.uniform(-E * 0.95, E * 0.95, 2)
        if gz(p) > -1.2:
            continue
        x, y = px(p)
        d.arc([x - 12, y - 6, x + 12, y + 6], 200, 340, fill=(140, 120, 80), width=2)
    # dashed paths
    for pl in PATHS:
        for a, b in zip(pl[:-1], pl[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            n = max(int(np.linalg.norm(b - a) / 0.9), 1)
            for k in range(0, n, 2):
                p0, p1 = a + (b - a) * k / n, a + (b - a) * min(k + 1, n) / n
                d.line([px(p0), px(p1)], fill=(139, 60, 36), width=4)
    # palms as little stars
    for p in palms:
        x, y = px(p)
        for k in range(5):
            a = 2 * np.pi * k / 5
            d.line([(x, y), (x + 9 * math.cos(a), y + 9 * math.sin(a))], fill=(70, 100, 45), width=3)
    # landmarks
    tx, ty = px(TEMPLE)
    d.rectangle([tx - 30, ty - 30, tx + 30, ty + 30], outline=ink, width=4)
    for k in range(-2, 3):
        d.line([(tx + k * 11, ty - 18), (tx + k * 11, ty + 18)], fill=ink, width=3)
    wx, wy = px(WRECK_POS)
    d.arc([wx - 34, wy - 34, wx + 34, wy + 8], 20, 160, fill=ink, width=5)
    d.line([(wx, wy - 10), (wx + 8, wy - 52)], fill=ink, width=4)
    d.polygon([(wx + 8, wy - 50), (wx + 30, wy - 36), (wx + 5, wy - 26)], outline=ink, width=3)
    a0 = px(DOCK_BASE)
    a1 = px(DOCK_BASE + DOCK_DIR * DOCK_LEN)
    d.line([a0, a1], fill=ink, width=16)
    d.line([a0, a1], fill=(176, 136, 86), width=10)
    ax, ay = px(ARCH)
    d.arc([ax - 34, ay - 30, ax + 34, ay + 38], 180, 360, fill=ink, width=8)
    cx, cy = px(CAMP)
    d.polygon([(cx - 40, cy - 12), (cx - 24, cy - 44), (cx - 8, cy - 12)], fill=(178, 56, 44), outline=ink)
    d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(230, 120, 40), outline=ink)
    ix, iy = px(ISLET)
    for name, (lx, ly) in (("Tapınak", (tx - 52, ty - 74)), ("Batık Gemi", (wx - 70, wy + 22)),
                           ("İskele", (a1[0] - 40, a1[1] + 18)), ("Kaya Kemeri", (ax - 80, ay - 76)),
                           ("Korsan Kampı", (cx - 96, cy + 18)), ("Adacık", (ix - 44, iy - 66))):
        d.text((lx, ly), name, font=label_f, fill=ink)
    # the smaller structures
    hx, hy = px(LIGHTHOUSE)
    d.ellipse([hx - 13, hy - 13, hx + 13, hy + 13], fill=(194, 59, 46), outline=ink, width=3)
    for k in range(8):
        a = 2 * np.pi * k / 8
        d.line([(hx + 17 * math.cos(a), hy + 17 * math.sin(a)), (hx + 27 * math.cos(a), hy + 27 * math.sin(a))],
               fill=ink, width=3)
    for c, w_, h_ in ((HUT, 22, 22), (SHED, 22, 16)):
        x, y = px(c)
        d.rectangle([x - w_ / 2, y - h_ / 2 + 6, x + w_ / 2, y + h_ / 2 + 6], fill=(176, 136, 86), outline=ink, width=3)
        d.polygon([(x - w_ / 2 - 5, y - h_ / 2 + 6), (x, y - h_ / 2 - 10), (x + w_ / 2 + 5, y - h_ / 2 + 6)],
                  fill=(201, 162, 79), outline=ink)
    x, y = px(RUIN)
    for seg in (((-28, -20), (28, -20)), ((28, -20), (28, 2)), ((-28, -20), (-28, 20)), ((-28, 20), (-6, 20)),
                ((6, 20), (14, 20))):
        d.line([(x + seg[0][0], y + seg[0][1]), (x + seg[1][0], y + seg[1][1])], fill=ink, width=5)
    x, y = px(WELL)
    d.ellipse([x - 10, y - 10, x + 10, y + 10], fill=(40, 70, 90), outline=ink, width=3)
    x, y = px(GRAVEYARD)
    d.rectangle([x - 30, y - 24, x + 30, y + 24], outline=ink, width=2)
    for dx, dy in ((-15, -6), (0, 8), (15, -6)):
        d.line([(x + dx, y + dy - 9), (x + dx, y + dy + 9)], fill=ink, width=3)
        d.line([(x + dx - 6, y + dy - 3), (x + dx + 6, y + dy - 3)], fill=ink, width=3)
    x, y = px(TOWER)
    d.rectangle([x - 14, y - 14, x + 14, y + 14], outline=ink, width=3)
    d.line([(x - 14, y - 14), (x + 14, y + 14)], fill=ink, width=3)
    d.line([(x - 14, y + 14), (x + 14, y - 14)], fill=ink, width=3)
    for name, c, (ox, oy) in (("Deniz Feneri", LIGHTHOUSE, (-178, -36)), ("Balıkçı Kulübesi", HUT, (-70, 40)),
                              ("Yıkık Ev", RUIN, (-46, 30)), ("Kuyu", WELL, (16, -12)),
                              ("Mezarlık", GRAVEYARD, (-44, 30)), ("Gözcü Kulesi", TOWER, (-64, 20)),
                              ("Kaçakçı Barakası", SHED, (-80, 22))):
        x, y = px(c)
        d.text((x + ox, y + oy), name, font=small_f, fill=ink)
    # treasure X marks, numbered
    for i, p in enumerate(spawns):
        x, y = px(p)
        for s in (1, -1):
            d.line([(x - 13, y - 13 * s), (x + 13, y + 13 * s)], fill=red, width=6)
        d.text((x + 14, y - 2), str(i + 1), font=small_f, fill=red)
    for p in players:
        x, y = px(p)
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(40, 70, 130), outline=ink)
    for _, _, p, _ in props or []:
        x, y = px(p)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(214, 160, 40), outline=ink)
    # title cartouche, legend, compass
    title = "Kayıp Korsan Koyu"
    tw = d.textlength(title, font=title_f)
    d.rectangle([S / 2 - tw / 2 - 26, 46, S / 2 + tw / 2 + 26, 124], fill=(236, 218, 170), outline=ink, width=4)
    d.text((S / 2 - tw / 2, 54), title, font=title_f, fill=ink)
    lx, ly = 70, S - 190
    d.rectangle([lx - 16, ly - 16, lx + 380, ly + 104], fill=(236, 218, 170), outline=ink, width=3)
    for s in (1, -1):
        d.line([(lx, ly + 14 - 10 * s), (lx + 20, ly + 14 + 10 * s)], fill=red, width=5)
    d.text((lx + 34, ly + 2), f"Hazine noktası (1–{len(spawns)})", font=small_f, fill=ink)
    d.ellipse([lx + 3, ly + 40, lx + 17, ly + 54], fill=(40, 70, 130), outline=ink)
    d.text((lx + 34, ly + 36), "Oyuncu başlangıcı", font=small_f, fill=ink)
    d.line([(lx, ly + 82), (lx + 22, ly + 82)], fill=(139, 60, 36), width=4)
    d.text((lx + 34, ly + 70), "Patika", font=small_f, fill=ink)
    if props:
        d.rectangle([lx - 16, ly + 104, lx + 380, ly + 140], fill=(236, 218, 170), outline=ink, width=3)
        d.ellipse([lx + 5, ly + 115, lx + 15, ly + 125], fill=(214, 160, 40), outline=ink)
        d.text((lx + 34, ly + 108), f"Hazine olabilecek eşya ({len(props)})", font=small_f, fill=ink)
    ccx, ccy, R = S - 150, S - 160, 70
    d.ellipse([ccx - R, ccy - R, ccx + R, ccy + R], outline=ink, width=3)
    for k, lab in enumerate(("K", "D", "G", "B")):
        a = -math.pi / 2 + k * math.pi / 2
        tip = (ccx + R * 0.95 * math.cos(a), ccy + R * 0.95 * math.sin(a))
        l = (ccx + 14 * math.cos(a + math.pi / 2), ccy + 14 * math.sin(a + math.pi / 2))
        rr = (ccx + 14 * math.cos(a - math.pi / 2), ccy + 14 * math.sin(a - math.pi / 2))
        d.polygon([tip, l, (ccx, ccy), rr], fill=red if k == 0 else ink)
        tx2, ty2 = ccx + (R + 22) * math.cos(a) - 9, ccy + (R + 22) * math.sin(a) - 13
        d.text((tx2, ty2), lab, font=small_f, fill=ink)
    # scale bar: 10 m
    sb = 10 / (2 * E) * S
    d.line([(S - 330, S - 60), (S - 330 + sb, S - 60)], fill=ink, width=5)
    d.text((S - 330, S - 92), "10 m", font=small_f, fill=ink)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)
    print("[map]", path)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export_glb(objs, path):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_apply=True,
                              export_extras=True, export_yup=True, export_cameras=False, export_lights=False)
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in objs if o.type == 'MESH')
    print("[export]", path, "triangles:", tris)


def setup_viewport():
    scene = bpy.context.scene
    for eng in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            continue


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    make_materials()
    world_col = K.get_collection("TreasureCove")
    land_col = K.get_collection("Terrain", world_col)
    mark_col = K.get_collection("Landmarks", world_col)
    veg_col = K.get_collection("Vegetation", world_col)
    game_col = K.get_collection("Gameplay", world_col)
    rig = K.get_collection("Preview_Rig")

    print(f"[layout] temple z={TEMPLE_Z:.2f} camp z={CAMP_Z:.2f} arch z={gz(ARCH):.2f} islet={ISLET.round(1)}")
    print(f"[layout] dock base={DOCK_BASE.round(1)} wreck={WRECK_POS.round(1)}")
    build_terrain(land_col, 100 if FAST else 190)
    fire = []
    build_temple(mark_col, fire)
    wreck = build_shipwreck(mark_col)
    build_dock(mark_col)
    build_arch(mark_col)
    build_camp(mark_col, fire)
    build_x_mark(mark_col, TREASURE_SPAWNS[5][1])
    build_lighthouse(mark_col, fire)
    build_fisher_hut(mark_col)
    build_ruined_house(mark_col)
    build_well(mark_col)
    build_graveyard(mark_col)
    build_watchtower(mark_col)
    build_shed(mark_col)
    palms, _, _ = build_vegetation(veg_col)
    spawns, players = build_spawns(game_col)
    props, item_meshes = build_treasure_props(K.get_collection("TreasureProps", world_col), palms, wreck)
    build_ground_cover(veg_col)
    for label, p in spawns:
        print(f"[spawn] {label:36s} ({p[0]:6.1f}, {p[1]:6.1f}, {p[2]:5.2f})")

    setup_world(rig, fire)
    hero = camera(rig, "Cam_Hero", (36, -40, 25), (-1, 0, 0.5), lens=34)
    top = camera(rig, "Cam_Top", (0, 0, 80), (0, 0, 0), ortho=72)
    top.rotation_euler = (0, 0, 0)
    near = camera(rig, "Cam_Temple", (9, -10, TEMPLE_Z + 5.5), (0, 5, TEMPLE_Z + 1.4), lens=28)
    east = camera(rig, "Cam_East", (36, -26, 13), (14, -1, 2.5), lens=30)
    west = camera(rig, "Cam_West", (-34, -30, 13), (-15, -5, 2.0), lens=30)

    draw_parchment_map(os.path.join(RENDER_DIR, "treasure_map.png"), [p for _, p in spawns], players, palms)
    draw_parchment_map(os.path.join(RENDER_DIR, "treasure_props_map.png"), [p for _, p in spawns], players, palms,
                       props=props)
    if not NO_RENDER:
        s = 12 if FAST else 64
        res_w = (960, 540) if FAST else (1920, 1080)
        render(hero, os.path.join(RENDER_DIR, "cove_hero.png"), res_w, s)
        render(near, os.path.join(RENDER_DIR, "cove_temple.png"), res_w, s)
        render(east, os.path.join(RENDER_DIR, "cove_east.png"), res_w, s)
        render(west, os.path.join(RENDER_DIR, "cove_west.png"), res_w, s)
        render(top, os.path.join(RENDER_DIR, "cove_top.png"), (800, 800) if FAST else (1600, 1600), s)
        render_prop_catalog(rig, item_meshes, os.path.join(RENDER_DIR, "props_catalog.png"), s)
    scene.camera = hero

    setup_viewport()
    blend_path = os.path.join(HERE, "treasure_cove.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True, relative_remap=True)
    objs = [o for o in world_col.all_objects if o.type in ('MESH', 'EMPTY')]
    export_glb(objs, os.path.join(EXPORT_DIR, "treasure_cove.glb"))


if __name__ == "__main__":
    main()
