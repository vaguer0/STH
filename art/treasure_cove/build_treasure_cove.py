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
    h = h + 0.10 * fbm(x / 2.2, y / 2.2, 2, seed=3) * ss(-0.05, 0.05, shore)
    # the islet and the sandbar that leads to it (knee-deep at -0.45 m)
    h = np.maximum(h, -0.3 + 1.5 * np.exp(-((x - ISLET[0]) ** 2 + (y - ISLET[1]) ** 2) / (2 * 2.4 ** 2)))
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

PATHS = [
    [DOCK_BASE - DOCK_DIR * 0.5, DOCK_BASE - DOCK_DIR * 6 + [0, 1.5], np.array([8.0, -0.5]),
     np.array([3.0, -0.5]), TEMPLE + [0, -5.8]],
    [WRECK_POS - WRECK_DIR * 4.5, np.array([-3.5, -10.0]), np.array([-1.5, -2.5]), TEMPLE + [0, -5.8]],
    [CAMP + [3.0, 0.5], np.array([-10.0, -2.5]), np.array([-4.0, -1.6]), TEMPLE + [-0.8, -5.8]],
    [TEMPLE + [-5.2, 1.5], np.array([-10.0, 9.0]), ARCH + [1.2, -2.6]],
    [TEMPLE + [5.2, 2.0], np.array([11.0, 10.5]), shore_point(ISLET_TH, inland=1.5)],
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


def box(g, center, size, mat, rot=(0, 0, 0)):
    sx, sy, sz = np.asarray(size, float) / 2
    v = np.array([[x, y, z] for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)])
    f = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    return g.add(xf(v, center, rot), f, mat)


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
    flat_mat("M_Flame", "#ff7a1f", emit="#ff7a1f", emit_strength=8.0)
    flat_mat("M_FlameCore", "#ffe08a", emit="#ffe08a", emit_strength=12.0)
    for m in MATS.values():
        m.use_backface_culling = False


def emit(name, g, col, loc=(0, 0, 0)):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in g.v], [], g.f)
    used = list(dict.fromkeys(g.m))
    for m in used:
        mesh.materials.append(MATS[m])
    idx = {m: i for i, m in enumerate(used)}
    mesh.polygons.foreach_set("material_index", [idx[m] for m in g.m])
    mesh.validate()
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

    deep = ss(-0.2, -3.0, hc)[:, None]
    col_rgb = c("#dcc48c") * (1 - deep) + c("#2a6d86") * deep
    sand_line = 0.95 + 0.25 * var
    sand = np.where((hc < 0.18)[:, None], c("#d8bd84"), c("#eed79c"))
    col_rgb = np.where((hc >= -0.15)[:, None], sand, col_rgb)
    grass = c("#7dba50") * (0.5 + 0.5 * var[:, None]) + c("#5b9c3d") * (0.5 - 0.5 * var[:, None])
    on_grass = hc >= sand_line
    col_rgb = np.where(on_grass[:, None], grass, col_rgb)
    col_rgb = np.where((on_grass & (nz < 0.86))[:, None], c("#938a7c"), col_rgb)
    col_rgb = np.where((on_grass & (nz < 0.74))[:, None], c("#716a60"), col_rgb)
    pd = path_dist(cx, cy)
    dirt = ss(1.25, 0.85, pd + 0.25 * var) * (hc > 0.25)
    col_rgb = col_rgb * (1 - dirt[:, None]) + c("#ae8853") * dirt[:, None]
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
    box(g, (0, 0, H / 2), (L, W, H), "M_Wood")
    for x in (-L / 2 + 0.07, L / 2 - 0.07):
        box(g, (x, 0, H / 2), (0.06, W + 0.03, H + 0.02), "M_Iron")
    # rounded lid: half cylinder along x
    n = 7
    ang = np.linspace(0, np.pi, n)
    prof = np.stack([np.zeros(n), (W / 2) * np.cos(ang), 0.22 * np.sin(ang)], axis=1)
    rings = [prof + [x, 0, 0] for x in (-L / 2, L / 2)]
    lid = Geo()
    loft(lid, rings, "M_WoodDark")
    for x in (-L / 2 + 0.07, L / 2 - 0.07):
        band = [prof * [1, 1.04, 1.06] + [xx, 0, 0] for xx in (x - 0.035, x + 0.035)]
        loft(lid, band, "M_Iron")
    if open_lid:
        g.merge(lid, loc=(0, W / 2, H), rot=(math.radians(-105), 0, 0), scale=1.0)
    else:
        g.merge(lid, loc=(0, 0, H))
    box(g, (0, -W / 2 - 0.02, H - 0.02), (0.12, 0.05, 0.16), "M_Gold")
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
    ts = np.linspace(0, 1, 9)
    for i, t in enumerate(ts):
        c = ld * bend * t * t + [0, 0, height * t - 0.2]
        r = (0.24 + (0.14 - 0.24) * t) * (1.08 if i % 2 else 1.0)
        a = np.linspace(0, 2 * np.pi, 7, endpoint=False) + 0.3 * i
        rings.append(c + r * np.stack([np.cos(a), np.sin(a), np.zeros(7)], axis=1))
    loft(g, rings, "M_PalmTrunk")
    top = ld * bend + [0, 0, height - 0.2]
    nf = rng.integers(7, 10)
    for k in range(nf):
        yaw = 2 * np.pi * k / nf + rng.uniform(-0.25, 0.25)
        d = np.array([math.cos(yaw), math.sin(yaw), 0])
        side = np.array([-d[1], d[0], 0])
        L = rng.uniform(2.3, 3.0)
        lift = rng.uniform(0.3, 0.6)
        st = np.linspace(0, 1, 7)
        Lv, Cv, Rv = [], [], []
        for s in st:
            p = top + d * L * s + [0, 0, L * (lift * s - 0.85 * s * s)]
            w = 0.55 * math.sin(math.pi * min(s * 1.15, 1.0)) ** 0.6 + 0.02
            Cv.append(p + [0, 0, 0.05])
            Lv.append(p + side * w - [0, 0, 0.16 * w])
            Rv.append(p - side * w - [0, 0, 0.16 * w])
        n = len(st)
        v = np.array(Lv + Cv + Rv)
        f = []
        for i in range(n - 1):
            f.append([i, i + 1, n + i + 1, n + i])
            f.append([n + i, n + i + 1, 2 * n + i + 1, 2 * n + i])
        g.add(v, f, "M_PalmLeaf" if k % 2 else "M_PalmLeafDark")
    v, f = ico(1)
    for k in range(3):
        a = 2 * np.pi * k / 3 + rng.uniform(0, 1)
        g.add(v * 0.13 + top + [0.17 * math.cos(a), 0.17 * math.sin(a), -0.18], f, "M_Coconut")
    return g


def bush_geo(seed):
    rng = np.random.default_rng(seed)
    g = Geo()
    for k in range(rng.integers(3, 5)):
        a = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(0.1, 0.55)
        s = rng.uniform(0.45, 0.8)
        rock(g, (d * math.cos(a), d * math.sin(a), s * 0.45), (s, s, s * 0.75),
             "M_Bush" if k % 2 else "M_BushLight", seed=seed * 7 + k, rough=0.15, sink=0.5)
    return g


def barrel(g, center, seed=0, lying=False):
    t = np.linspace(0, 1, 6)
    rings = []
    for tt in t:
        r = 0.27 + 0.05 * math.sin(math.pi * tt)
        a = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        rings.append(np.stack([r * np.cos(a), r * np.sin(a), np.full(10, 0.85 * tt)], axis=1))
    b = Geo()
    loft(b, rings, "M_Wood")
    for z in (0.12, 0.73):
        tube(b, (0, 0, z), (0, 0, z + 0.05), 0.31, mat="M_Iron", n=10)
    rot = (math.radians(90), 0, seed) if lying else (0, 0, seed)
    loc = np.asarray(center) + ([0, 0, 0.29] if lying else [0, 0, 0])
    g.merge(b, loc=loc, rot=rot)


def crate(g, center, size=0.7, yaw=0.0):
    c = Geo()
    box(c, (0, 0, size / 2), (size, size, size), "M_WoodLight")
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(c, (sx * size / 2, sy * size / 2, size / 2), (0.08, 0.08, size + 0.01), "M_WoodDark")
    box(c, (0, 0, size), (size + 0.04, 0.1, 0.04), "M_WoodDark")
    g.merge(c, loc=center, rot=(0, 0, yaw))


def torch(g, base, lights):
    base = np.asarray(base, float)
    tube(g, base - [0, 0, 0.3], base + [0, 0, 1.6], 0.06, 0.05, "M_WoodDark", n=6)
    tube(g, base + [0, 0, 1.55], base + [0, 0, 1.75], 0.09, 0.16, "M_Iron", n=8)
    tube(g, base + [0, 0, 1.7], base + [0, 0, 2.15], 0.12, 0.0, "M_Flame", n=6)
    tube(g, base + [0, 0, 1.72], base + [0, 0, 2.0], 0.065, 0.0, "M_FlameCore", n=6)
    lights.append(base + [0, 0, 2.0])


# ---------------------------------------------------------------------------
# landmarks
# ---------------------------------------------------------------------------

def build_temple(col, lights):
    g = Geo()
    z0 = -0.3  # object origin sits at TEMPLE_Z
    tiers = [(9.0, 0.75), (7.2, 0.6), (5.6, 0.55)]
    z = z0
    for k, (s, h) in enumerate(tiers):
        box(g, (0, 0, z + h / 2), (s, s, h), "M_Stone" if k % 2 == 0 else "M_StoneDark")
        z += h
    top = z
    # stair run down the south face
    steps = 7
    for k in range(steps):
        zt = top - (k + 0.5) * top / steps
        box(g, (0, -2.8 - k * 0.42, (zt - 1.5) / 2), (2.6, 0.42, zt + 1.5), "M_StoneDark" if k % 2 else "M_Stone")
    for sx in (-1, 1):
        box(g, (sx * 1.5, -2.8 - 3 * 0.42, (top - 1.5) / 2 - 0.2), (0.4, 7 * 0.42, top + 1.1), "M_StoneDark")
    # pillars; the back row still carries its lintel
    rng = np.random.default_rng(5)
    for px, py in [(-2.2, 2.2), (0, 2.2), (2.2, 2.2), (-2.2, 0), (2.2, 0), (-2.2, -2.2), (2.2, -2.2)]:
        full = py > 2
        h = 3.0 if full else rng.uniform(0.6, 1.9)
        box(g, (px, py, top + 0.12), (0.9, 0.9, 0.24), "M_StoneDark")
        tube(g, (px, py, top + 0.24), (px, py, top + 0.24 + h), 0.33, 0.3 if full else 0.31, "M_Stone", n=8)
        if full:
            box(g, (px, py, top + 0.34 + h), (0.86, 0.86, 0.2), "M_StoneDark")
        else:
            box(g, (px + 0.05, py, top + 0.24 + h), (0.5, 0.42, 0.18), "M_Stone", rot=(0.3, 0.2, rng.uniform(0, 3)))
    box(g, (0, 2.2, top + 3.64), (5.4, 0.8, 0.45), "M_Stone")
    box(g, (0, 2.2, top + 3.9), (5.0, 0.6, 0.12), "M_Moss")
    # altar
    box(g, (0, 0.7, top + 0.4), (1.5, 0.95, 0.8), "M_StoneDark")
    box(g, (0, 0.7, top + 0.85), (1.75, 1.15, 0.1), "M_Stone")
    TREASURE_Z[0] = TEMPLE_Z + top + 0.9
    # fallen pillars and loose blocks behind the temple
    for p0, p1 in [((1.8, 5.4), (4.6, 6.6)), ((5.0, 4.8), (6.4, 2.6)), ((-4.8, 5.2), (-3.0, 6.6))]:
        a = TEMPLE + np.array(p0)
        b = TEMPLE + np.array(p1)
        za, zb = gz(a), gz(b)
        tube(g, (a[0] - TEMPLE[0], a[1] - TEMPLE[1], za + 0.25 - TEMPLE_Z),
             (b[0] - TEMPLE[0], b[1] - TEMPLE[1], zb + 0.25 - TEMPLE_Z), 0.32, mat="M_Stone", n=8, phase=0.4)
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
        rock(g, (p[0], p[1], gz(p) + 1.2), (1.5, 1.7, 2.6), "M_Rock", seed=40 + k, yaw=0.5, sink=0.1)
    lz = gz(ARCH) + 3.6
    rock(g, (ARCH[0], ARCH[1], lz), (3.9, 1.6, 1.15), "M_RockDark", seed=44, yaw=0.5, subdiv=2, rough=0.12)
    rock(g, (ARCH[0] + 0.4, ARCH[1] + 0.2, lz + 0.9), (1.8, 1.0, 0.6), "M_Moss", seed=45, yaw=0.6, rough=0.15)
    rng = np.random.default_rng(46)
    for k in range(6):
        a = rng.uniform(0, 2 * np.pi)
        p = ARCH + rng.uniform(3.8, 5.5) * np.array([math.cos(a), math.sin(a)])
        s = rng.uniform(0.5, 1.1)
        rock(g, (p[0], p[1], gz(p) + s * 0.3), (s * 1.2, s, s * 0.8), "M_Rock", seed=50 + k, yaw=a)
        SOLIDS.append((p, s * 1.3))
    return emit("Rock_Arch", g, col)


def build_camp(col, lights):
    g = Geo()
    c = np.array([CAMP[0], CAMP[1], CAMP_Z])
    rng = np.random.default_rng(9)
    for k in range(9):
        a = 2 * np.pi * k / 9
        rock(g, c + [0.75 * math.cos(a), 0.75 * math.sin(a), 0.05], (0.2, 0.17, 0.14), "M_RockDark", seed=60 + k, yaw=a)
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
    for c, r in ((ARCH, 4.5), (CAMP, 3.6), (WRECK_POS, 6.5), (DOCK_BASE, 3.5)):
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
    taken += palms
    bushes = scatter(26, (1.2, 9.0), 2.0, taken)
    g = Geo()
    for k, p in enumerate(bushes):
        g.merge(bush_geo(200 + k), loc=(p[0], p[1], gz(p)), rot=(0, 0, RNG.uniform(0, 6.3)),
                scale=RNG.uniform(0.9, 1.4))
    emit("Bushes", g, col)
    taken += bushes
    rocks = scatter(22, (-0.8, 9.0), 2.2, taken)
    g = Geo()
    for k, p in enumerate(rocks):
        s = RNG.uniform(0.35, 1.0)
        rock(g, (p[0], p[1], gz(p)), (s * RNG.uniform(1.0, 1.6), s, s * 0.7), "M_Rock" if k % 3 else "M_RockDark",
             seed=300 + k, yaw=RNG.uniform(0, 6.3))
    # boulders around the north beach spawn, big enough to hide behind
    nb = TREASURE_SPAWNS[7][1]
    for k, (dx, dy, s) in enumerate([(-1.4, 0.9, 1.4), (1.3, 1.2, 1.1), (0.2, 2.0, 0.9)]):
        p = nb + [dx, dy]
        rock(g, (p[0], p[1], gz(p) + 0.2), (s * 1.3, s, s), "M_Rock", seed=400 + k, yaw=k)
        SOLIDS.append((p, s * 1.4))
    emit("Rocks", g, col)
    SOLIDS.extend([(p, 0.45) for p in palms] + [(p, 0.8) for p in bushes] + [(p, 1.25) for p in rocks])
    return palms, bushes, rocks


# ---------------------------------------------------------------------------
# gameplay markers and chests
# ---------------------------------------------------------------------------

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

def lathe(g, prof, mat, n=10, cap_top=True, cap_bottom=True):
    """prof: list of (radius, z); a surface of revolution around +z."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rings = [np.stack([max(r, 1e-3) * np.cos(a), max(r, 1e-3) * np.sin(a), np.full(n, z)], axis=1) for r, z in prof]
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
    "inland": ["mushrooms", "stump", "sack", "clay_pot", "shovel", "sword", "helmet", "coconut", "lantern",
               "skull", "gems", "small_chest", "tankard", "flag", "amphora"],
}
# (zone, count, centre, radius, height range)
ZONE_PLAN = [
    ("beach", 44, None, None, (-0.15, 0.85)),
    ("palms", 22, None, None, (0.1, 3.5)),
    ("temple", 30, TEMPLE, 8.5, (0.0, 20.0)),
    ("camp", 20, CAMP, 6.0, (0.3, 6.0)),
    ("wreck", 20, None, 8.5, (-0.1, 4.0)),
    ("arch", 12, ARCH, 6.0, (0.5, 12.0)),
    ("dock", 6, None, None, None),
    ("islet", 5, ISLET, 3.4, (0.0, 3.0)),
    ("inland", 41, np.zeros(2), 26.0, (0.9, 20.0)),
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
    d.text((lx + 34, ly + 2), "Hazine noktası (1–8)", font=small_f, fill=ink)
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
    build_terrain(land_col, 80 if FAST else 132)
    fire = []
    build_temple(mark_col, fire)
    wreck = build_shipwreck(mark_col)
    build_dock(mark_col)
    build_arch(mark_col)
    build_camp(mark_col, fire)
    build_x_mark(mark_col, TREASURE_SPAWNS[5][1])
    palms, _, _ = build_vegetation(veg_col)
    spawns, players = build_spawns(game_col)
    props, item_meshes = build_treasure_props(K.get_collection("TreasureProps", world_col), palms, wreck)
    for label, p in spawns:
        print(f"[spawn] {label:36s} ({p[0]:6.1f}, {p[1]:6.1f}, {p[2]:5.2f})")

    setup_world(rig, fire)
    hero = camera(rig, "Cam_Hero", (36, -40, 25), (-1, 0, 0.5), lens=34)
    top = camera(rig, "Cam_Top", (0, 0, 80), (0, 0, 0), ortho=72)
    top.rotation_euler = (0, 0, 0)
    near = camera(rig, "Cam_Temple", (9, -10, TEMPLE_Z + 5.5), (0, 5, TEMPLE_Z + 1.4), lens=28)

    draw_parchment_map(os.path.join(RENDER_DIR, "treasure_map.png"), [p for _, p in spawns], players, palms)
    draw_parchment_map(os.path.join(RENDER_DIR, "treasure_props_map.png"), [p for _, p in spawns], players, palms,
                       props=props)
    if not NO_RENDER:
        s = 12 if FAST else 64
        res_w = (960, 540) if FAST else (1920, 1080)
        render(hero, os.path.join(RENDER_DIR, "cove_hero.png"), res_w, s)
        render(near, os.path.join(RENDER_DIR, "cove_temple.png"), res_w, s)
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
