"""
Blossom Valley -- SeekThenHide
==============================

A second small treasure-hunt map, in spring Japan: a valley about 50 x 60 m walled in
by pine-covered mountains.  A five-storey pagoda stands on the northern terrace,
reached by a stair lined with vermilion torii gates.  A spring on the terrace feeds a
stream that drops over the terrace edge as a waterfall, winds under an arched red
bridge and ends in a koi pond with a tea house and a lantern island.  The east side
holds a little village street with market stalls and rice paddies; the west a bamboo
grove with a fox shrine against the cliff.  Players enter through the south gate.

Fourteen treasure spawns (``TreasureSpawn_##``, each holding a lacquered chest), six
player spawns and 200 treasure props (``Prop_###_<item>``) are exported as glTF nodes
with ``kind`` extras, like the Lost Pirate Cove map.

Reuses the toolkit in ../treasure_cove/build_treasure_cove.py.

    blender --background --python build_blossom_valley.py
    blender --background --python build_blossom_valley.py -- --fast
    blender --background --python build_blossom_valley.py -- --no-render
"""

import math
import os
import sys

import bpy  # noqa: E402  (bpy must load before bmesh when run as a module)
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "treasure_cove"))
import build_treasure_cove as C  # noqa: E402

K = C.K
EXPORT_DIR = os.path.join(HERE, "export")
RENDER_DIR = os.path.join(HERE, "renders")

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
FAST = "--fast" in ARGS
NO_RENDER = "--no-render" in ARGS

ss, fbm, seg_dist, hexc = C.ss, C.fbm, C.seg_dist, C.hexc
Geo, box, tube, beam, loft, lathe, ball, rock = C.Geo, C.box, C.tube, C.beam, C.loft, C.lathe, C.ball, C.rock
emit, Site = C.emit, C.Site
SOLIDS = C.SOLIDS  # prop footprints, shared with the toolkit's plank_wall()
RNG = np.random.default_rng(3141)


# ---------------------------------------------------------------------------
# layout (metres, Blender Z up, +Y north)
# ---------------------------------------------------------------------------

TERRACE_H = 5.2
PAGODA = np.array([6.0, 21.0])
BELL_TOWER = np.array([14.5, 19.5])
ZEN = np.array([-1.5, 21.5])
SPRING = np.array([-9.0, 22.0])
GATE = np.array([6.0, -28.6])
POND = np.array([-4.0, -15.5])
POND_R = 5.0
POND_ISLAND = np.array([-3.0, -14.6])
TEA_HOUSE = np.array([-10.6, -15.8])
HOUSES = [(np.array([9.3, 2.2]), (0, -1)), (np.array([17.4, 1.8]), (0, -1)), (np.array([13.0, -8.4]), (0, 1))]
STALLS = [(np.array([7.6, -8.6]), (0, 1)), (np.array([19.6, -7.6]), (0, 1))]
PADDY_HUT = np.array([22.2, -13.4])
SHRINE = np.array([-21.6, 6.0])
BAMBOO = np.array([-17.0, 0.0])
STAIR_X, STAIR_Y0, STAIR_Y1 = 6.0, 8.2, 14.0

STREAM = [np.array(p, float) for p in
          [(-9.0, 21.0), (-10.0, 17.5), (-10.2, 14.0), (-9.6, 10.6), (-7.2, 6.8), (-3.8, 3.2), (-4.4, -2.0),
           (-2.6, -7.0), (-3.4, -11.2)]]
PADDIES = [(np.array([cx, cy]), np.array([2.7, 1.75])) for cy in (-15.6, -19.6, -23.4) for cx in (12.4, 18.4)]

PATHS = [
    [GATE + [0, 2.0], np.array([5.0, -18.0]), np.array([4.0, -10.0]), np.array([3.0, -4.0]), np.array([3.6, 3.0]),
     np.array([STAIR_X, STAIR_Y0 - 0.6])],                                            # main road
    [np.array([3.0, -4.0]), np.array([-1.0, -2.6]), np.array([-8.0, -2.0]), np.array([-12.5, -1.0]),
     np.array([-16.5, 1.5]), SHRINE + [2.2, -0.6]],                                   # bridge, bamboo, shrine
    [np.array([3.0, -3.6]), np.array([9.0, -3.2]), np.array([16.0, -3.0]), np.array([21.2, 0.5]),
     np.array([20.6, 6.5]), np.array([18.0, 12.0]), np.array([15.2, 17.0])],           # village street, ramp
    [np.array([5.0, -18.0]), np.array([0.0, -21.4]), np.array([-7.0, -21.4]), TEA_HOUSE + [-0.4, -3.0]],
    [np.array([STAIR_X, STAIR_Y1 + 0.8]), PAGODA + [0, -3.4]],
    [np.array([STAIR_X - 1.4, STAIR_Y1 + 1.2]), np.array([0.0, 16.6]), ZEN + [-0.5, -3.0],
     SPRING + [2.2, -2.2]],
    [np.array([STAIR_X + 1.4, STAIR_Y1 + 1.2]), np.array([12.0, 16.6]), BELL_TOWER + [-0.6, -2.0]],
    [np.array([16.0, -3.0]), np.array([20.0, -9.0]), PADDY_HUT + [-1.6, 1.2]],
    [np.array([0.0, -21.4]), np.array([-2.6, -20.4])],                               # to the stepping stones
]


def path_dist(x, y):
    d = np.full(np.shape(x), 1e9)
    for pl in PATHS:
        for a, b in zip(pl[:-1], pl[1:]):
            d = np.minimum(d, seg_dist(x, y, np.asarray(a, float), np.asarray(b, float)))
    return d


def stream_dist(x, y):
    d = np.full(np.shape(x), 1e9)
    for a, b in zip(STREAM[:-1], STREAM[1:]):
        d = np.minimum(d, seg_dist(x, y, a, b))
    return d


def rim_q(x, y):
    return ((np.abs(x) / 27.0) ** 4 + (np.abs(y + 1.0) / 31.0) ** 4) ** 0.25


def ramp(x):
    return ss(13.0, 16.0, x) * ss(24.0, 21.0, x)


def _floor(x, y):
    """Valley floor and northern terrace, before any carving."""
    floor = 0.3 + 0.022 * (y + 28) + 0.25 * fbm(x / 7, y / 7, 3, seed=21)
    r = ramp(x)
    e0 = 10.5 - 6.0 * r
    t = ss(e0, e0 + 3.0 + 9.0 * r, y)
    return floor * (1 - t) + (TERRACE_H + 0.2 * fbm(x / 9, y / 9, 2, seed=22)) * t


PADS = []  # (centre, inner radius, outer radius, level); filled below


def ground(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = _floor(x, y)
    # mountains all round, with a gap for the south gate
    q = rim_q(x, y) + 0.035 * fbm(x / 6, y / 6, 3, seed=23)
    m = ss(0.9, 1.18, q)
    gap = ss(3.6, 2.2, np.abs(x - GATE[0])) * ss(-34.0, -31.5, y)
    m = m * (1 - gap)
    h = h + 0.05 * fbm(x / 0.9, y / 0.9, 2, seed=27) * (1 - m)  # small bumps underfoot
    h = h + m * (11.0 + 5.0 * fbm(x / 9, y / 9, 3, seed=24) + 1.2 * fbm(x / 2.5, y / 2.5, 2, seed=25)
                  + 0.45 * fbm(x / 1.0, y / 1.0, 2, seed=30))
    # flat pads under the buildings
    for c, r0, r1, lvl in PADS:
        w = ss(r1, r0, np.hypot(x - c[0], y - c[1]))
        h = h * (1 - w) + lvl * w
    for c, hs in PADDIES:
        lvl = float(_floor(*c)) - 0.2
        w = ss(0.5, 0.0, np.maximum(np.abs(x - c[0]) - hs[0], np.abs(y - c[1]) - hs[1]))
        h = h * (1 - w) + lvl * w
    # stream channel, spring pool, koi pond and its island
    h = h - 0.85 * ss(1.7, 0.5, stream_dist(x, y))
    h = h - 0.9 * ss(2.6, 1.6, np.hypot(x - SPRING[0], y - SPRING[1]))
    w = ss(POND_R + 0.7, POND_R - 0.8, np.hypot(x - POND[0], y - POND[1]))
    h = h * (1 - w) - 1.1 * w
    h = np.maximum(h, -1.5 + 1.85 * ss(1.7, 0.6, np.hypot(x - POND_ISLAND[0], y - POND_ISLAND[1])))
    return h


def gz(p):
    return float(ground(p[0], p[1]))


for c, r0, r1 in [(PAGODA, 4.6, 6.4), (BELL_TOWER, 2.4, 3.6), (ZEN, 4.0, 5.4), (SHRINE, 2.2, 3.4),
                  (PADDY_HUT, 2.0, 3.2), (GATE, 3.0, 4.0)] + [(c, 3.6, 5.0) for c, _ in HOUSES] + \
        [(c, 2.0, 3.0) for c, _ in STALLS]:
    PADS.append((c, r0, r1, float(_floor(*c))))

# the toolkit's Site.g() and friends sample the ground through C.gz
C.gz, C.ground = gz, ground


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def make_materials():
    C.make_materials()
    fm = C.flat_mat
    fm("M_Vermilion", "#d4432c", rough=0.6)
    fm("M_BlackLacquer", "#1d1a1c", rough=0.3)
    fm("M_RoofTile", "#46505e", rough=0.7)
    fm("M_RoofTileDark", "#343c48", rough=0.7)
    fm("M_Plaster", "#f0e9dc", rough=0.95)
    fm("M_Shoji", "#f7f1e3", rough=0.9, emit="#fff4dc", emit_strength=0.25)
    fm("M_Tatami", "#c9c27a", rough=0.95)
    fm("M_Indigo", "#2c3f72", rough=0.9)
    fm("M_Gravel", "#e2ddd1", rough=1.0)
    fm("M_Bronze", "#7b6a3c", rough=0.4, metal=0.8)
    fm("M_LilyPad", "#4f8f3c", rough=0.6)
    fm("M_Sakura", "#f5b3c6", rough=0.9)
    fm("M_SakuraWhite", "#fde9ef", rough=0.9)
    fm("M_SakuraLight", "#fcd6e1", rough=0.9)
    fm("M_Maple", "#d64a2a", rough=0.9)
    fm("M_MapleLight", "#ec7a32", rough=0.9)
    fm("M_Pine", "#2f5a3a", rough=0.9)
    fm("M_PineLight", "#3f7347", rough=0.9)
    fm("M_Bamboo", "#7fae45", rough=0.6)
    fm("M_BambooDark", "#5b8a33", rough=0.6)
    fm("M_BambooLeaf", "#6aa63c", rough=0.8)
    fm("M_Straw", "#d8bb6a", rough=1.0)
    fm("M_Snow", "#f3f6f8", rough=0.8)
    fm("M_StreamWater", "#62bfd0", rough=0.05, alpha=0.7)
    fm("M_PondWater", "#3f97a6", rough=0.05, alpha=0.78)
    fm("M_PaddyWater", "#a7d3cb", rough=0.03, alpha=0.6)
    fm("M_Foam", "#ffffff", rough=0.6, alpha=0.85)
    fm("M_Koi", "#f06a2a", rough=0.5)
    fm("M_White", "#f5f2ea", rough=0.7)
    fm("M_PaperRed", "#d9392b", rough=0.9, emit="#ff6a3a", emit_strength=0.6)
    fm("M_Celadon", "#9cc4a8", rough=0.3)
    fm("M_Porcelain", "#eef0f2", rough=0.3)
    fm("M_Gourd", "#c99a45", rough=0.6)
    fm("M_Pumpkin", "#4f7a3a", rough=0.7)
    fm("M_Purple", "#6b3f8f", rough=0.9)
    for m in C.MATS.values():
        m.use_backface_culling = False


def remap(g, table):
    g.m = [table.get(m, m) for m in g.m]
    return g


# ---------------------------------------------------------------------------
# vegetation positions come first: the terrain is tinted with fallen petals
# ---------------------------------------------------------------------------

STRUCTURES = [(PAGODA, 4.8), (BELL_TOWER, 2.6), (ZEN, 4.2), (SHRINE, 2.8), (PADDY_HUT, 2.4), (GATE, 4.5),
              (TEA_HOUSE, 3.4), (POND, POND_R + 0.6), (SPRING, 2.6)] + [(c, 3.6) for c, _ in HOUSES] + \
             [(c, 2.0) for c, _ in STALLS]


def wet(p, margin=0.0):
    """True over the stream, pond, spring or a paddy."""
    if float(stream_dist(np.array([p[0]]), np.array([p[1]]))[0]) < 1.6 + margin:
        return True
    if math.hypot(*(p - POND)) < POND_R + 0.6 + margin or math.hypot(*(p - SPRING)) < 2.6 + margin:
        return True
    return any(abs(p[0] - c[0]) < hs[0] + 0.6 + margin and abs(p[1] - c[1]) < hs[1] + 0.6 + margin
               for c, hs in PADDIES)


def stair_zone(p, margin=0.0):
    return abs(p[0] - STAIR_X) < 2.2 + margin and STAIR_Y0 - 1.5 - margin < p[1] < STAIR_Y1 + 1.5 + margin


def site_clear(p, extra=0.0, path_gap=1.4):
    if wet(p, extra) or stair_zone(p, extra):
        return False
    if any(math.hypot(*(p - c)) < r + extra for c, r in STRUCTURES):
        return False
    return float(path_dist(np.array([p[0]]), np.array([p[1]]))[0]) > path_gap + extra


def scatter(count, ok, min_gap, existing, sampler, tries=6000):
    pts = []
    for _ in range(tries):
        if len(pts) >= count:
            break
        p = sampler()
        if not ok(p) or any(math.hypot(*(p - q)) < min_gap for q in pts + existing):
            continue
        pts.append(p)
    return pts


def in_valley(lo=-28, hi=26, qmax=0.86):
    def f():
        while True:
            p = RNG.uniform([-26, lo], [26, hi])
            if float(rim_q(*p)) < qmax:
                return p
    return f


def near(c, r):
    def f():
        a, d = RNG.uniform(0, 2 * np.pi), r * math.sqrt(RNG.random())
        return np.asarray(c, float) + d * np.array([math.cos(a), math.sin(a)])
    return f


def plan_vegetation():
    veg = {}
    taken = []
    veg["bamboo"] = scatter(150, lambda p: site_clear(p, 0, 1.0) and math.hypot(*(p - BAMBOO)) > 1.9, 0.75, [],
                            near(BAMBOO, 7.5))
    veg["bamboo"] += scatter(30, lambda p: site_clear(p, 0, 1.0) and float(rim_q(*p)) < 0.9, 0.8, veg["bamboo"],
                             near(SHRINE + [1, -2], 4.0))
    taken += veg["bamboo"]
    veg["sakura"] = scatter(22, lambda p: site_clear(p, 1.2), 4.2, taken, in_valley())
    veg["sakura"] += [p for p in [np.array([0.5, -18.6]), np.array([-9.4, -20.5]), np.array([10.5, 15.5]),
                                  np.array([1.0, 13.0]), np.array([-6.0, 10.5])] if site_clear(p, 0.8)]
    taken += veg["sakura"]
    veg["maple"] = scatter(7, lambda p: site_clear(p, 1.0), 4.0, taken, in_valley())
    taken += veg["maple"]
    veg["pine"] = scatter(110, lambda p: 0.9 < float(rim_q(*p)) < 1.32 and gz(p) < 14.5 and
                          not (abs(p[0] - GATE[0]) < 4.5 and p[1] < -26), 2.6, [],
                          lambda: RNG.uniform(-38, 38, 2))
    veg["pine"] += scatter(8, lambda p: site_clear(p, 1.0) and gz(p) > 4.5, 4.0, taken, in_valley(10, 27))
    taken += veg["pine"][-8:]
    veg["bush"] = scatter(40, lambda p: site_clear(p, 0.4), 2.2, taken, in_valley())
    taken += veg["bush"]
    veg["rock"] = scatter(26, lambda p: site_clear(p, 0.4, 1.0), 2.4, taken, in_valley(qmax=0.95))
    veg["rock"] += scatter(14, lambda p: 1.4 < float(stream_dist(np.array([p[0]]), np.array([p[1]]))[0]) < 2.4
                           and not stair_zone(p) and math.hypot(*(p - POND)) > POND_R + 0.4,
                           1.6, veg["rock"], in_valley(-14, 23))
    return veg


# ---------------------------------------------------------------------------
# terrain and water
# ---------------------------------------------------------------------------

EXTENT = 44.0


def build_terrain(col, n, veg):
    xs = np.linspace(-EXTENT, EXTENT, n + 1)
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
    cx, cy, hc = cen[:, 0], cen[:, 1], cen[:, 2]
    var = fbm(cx / 3.0, cy / 3.0, 2, seed=26)

    def c(h):
        return hexc(h)[None, :]

    def mix(base, col_, w):
        w = np.clip(w, 0, 1)[:, None]
        return base * (1 - w) + col_ * w

    var2 = fbm(cx / 1.1, cy / 1.1, 2, seed=28)
    var3 = fbm(cx / 6.0, cy / 6.0, 2, seed=29)
    rgb = c("#93c96a") * (0.5 + 0.5 * var[:, None]) + c("#6fae4c") * (0.5 - 0.5 * var[:, None])
    rgb = rgb * (0.93 + 0.09 * var2[:, None])
    rgb = mix(rgb, c("#5d9c45"), ss(-0.25, -0.55, var3) * 0.75)   # clover hollows
    rgb = mix(rgb, c("#b7c574"), ss(0.35, 0.6, var3) * 0.6)       # dry, sunny patches
    q = rim_q(cx, cy)
    mtn = ss(0.9, 1.05, q)
    rgb = mix(rgb, c("#5f8f45") * (0.9 + 0.1 * var[:, None]), mtn)
    band = np.floor(hc * 1.6 + 0.6 * var2) % 3  # bedding in the cliffs
    rock_c = np.where((band == 0)[:, None], c("#8d857a"), np.where((band == 1)[:, None], c("#7b746a"), c("#9d968a")))
    rgb = mix(rgb, rock_c, (nz < 0.8) * 1.0)
    rgb = mix(rgb, rock_c * 0.8, (nz < 0.62) * 1.0)
    rgb = mix(rgb, c("#a8a296") * (0.95 + 0.08 * var2[:, None]), (nz > 0.8) * ss(0.88, 0.95, q) * ss(0.3, 0.0, var2))  # scree
    rgb = mix(rgb, c("#f1f3f6"), ss(12.5, 14.0, hc + 0.8 * var))
    # bamboo grove floor, petals under the cherry trees
    rgb = mix(rgb, c("#8a9a4c") * (0.9 + 0.1 * var[:, None]), ss(8.5, 6.0, np.hypot(cx - BAMBOO[0], cy - BAMBOO[1])))
    petals = np.zeros(len(F))
    for t in veg["sakura"]:
        petals = np.maximum(petals, ss(3.4, 1.2, np.hypot(cx - t[0], cy - t[1])))
    rgb = mix(rgb, c("#f3c3d1"), petals * (0.55 + 0.35 * var))
    # stream banks: pebbles; beds under water: dark
    sd = stream_dist(cx, cy)
    pd_ = np.hypot(cx - POND[0], cy - POND[1])
    rgb = mix(rgb, c("#aaa293"), ss(2.3, 1.6, sd) + ss(POND_R + 1.4, POND_R + 0.7, pd_))
    rgb = mix(rgb, c("#62715d"), ss(1.2, 0.6, sd) + ss(POND_R, POND_R - 0.8, pd_))
    rgb = mix(rgb, c("#aaa293"), ss(1.7, 1.2, np.hypot(cx - POND_ISLAND[0], cy - POND_ISLAND[1])))
    for cc, hs in PADDIES:
        inside = (np.abs(cx - cc[0]) < hs[0]) & (np.abs(cy - cc[1]) < hs[1])
        rgb = np.where(inside[:, None], c("#7d6b48"), rgb)
    pdist = path_dist(cx, cy)
    path_c = c("#d2c19b") * (0.94 + 0.07 * var2[:, None])
    path_c = mix(path_c, c("#bfae88"), ss(0.7, 0.25, pdist) * 0.5)
    rgb = mix(rgb, path_c, ss(1.15, 0.8, pdist + 0.2 * var + 0.1 * var2))
    for cc, r0, _, _ in PADS:
        rgb = mix(rgb, c("#cbb994"), ss(r0 + 0.4, r0 - 0.2, np.hypot(cx - cc[0], cy - cc[1])) * 0.85)
    rgb = np.clip(rgb * (1 + 0.04 * (2 * RNG.random(len(F))[:, None] - 1)), 0, 1)

    mesh = bpy.data.meshes.new("Terrain")
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.materials.append(C.MATS["M_Terrain"])
    attr = mesh.color_attributes.new("Col", 'BYTE_COLOR', 'CORNER')
    rgba = np.concatenate([np.repeat(rgb, 3, axis=0), np.ones((len(F) * 3, 1))], axis=1)
    attr.data.foreach_set("color_srgb", rgba.astype(np.float32).ravel())
    mesh.polygons.foreach_set("use_smooth", (nz >= 0.8).tolist())  # soft ground, faceted rock faces
    mesh.update()
    ob = bpy.data.objects.new("Terrain", mesh)
    col.objects.link(ob)
    return ob


def resample(poly, step):
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(int(np.linalg.norm(b - a) / step), 1)
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


POND_Z = -0.2


def build_water(col):
    pts = resample(STREAM + [POND + (STREAM[-1] - POND) * 0.6], 0.35)
    bed = np.array([gz(p) for p in pts])
    wz = np.minimum.accumulate(bed + 0.5)
    wz = np.maximum(wz, POND_Z)
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    side = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    g = Geo()
    hw = 1.25
    v = []
    for p, s, z in zip(pts, side, wz):
        v += [[*(p - s * hw), z], [*(p + s * hw), z]]
    f = [[2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2] for i in range(len(pts) - 1)]
    g.add(v, f, "M_StreamWater")
    # the spring pool, the koi pond and the paddies
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for c, r, z, mat in ((SPRING, 2.3, wz[0], "M_StreamWater"), (POND, POND_R + 0.5, POND_Z, "M_PondWater")):
        g.add([[c[0] + r * math.cos(t), c[1] + r * math.sin(t), z] for t in a], [list(range(24))], mat)
    for c, hs in PADDIES:
        z = float(_floor(*c)) - 0.2 + 0.12
        g.add([[c[0] - hs[0], c[1] - hs[1], z], [c[0] + hs[0], c[1] - hs[1], z], [c[0] + hs[0], c[1] + hs[1], z],
               [c[0] - hs[0], c[1] + hs[1], z]], [[0, 1, 2, 3]], "M_PaddyWater")
        for s in (-1, 1):  # mud ridges between the paddies
            box(g, (c[0], c[1] + s * (hs[1] + 0.12), z + 0.02), (2 * hs[0] + 0.5, 0.3, 0.3), "M_Dirt")
            box(g, (c[0] + s * (hs[0] + 0.12), c[1], z + 0.02), (0.3, 2 * hs[1] + 0.5, 0.3), "M_Dirt")
        rng = np.random.default_rng(int(c[0] * 10 + c[1]))
        for _ in range(26):  # rice seedlings
            p = c + rng.uniform(-hs + 0.3, hs - 0.3)
            tube(g, (p[0], p[1], z - 0.1), (p[0] + 0.03, p[1], z + 0.28), 0.05, 0.0, "M_BambooLeaf", n=4)
    ob = emit("Water", g, col)
    # foam where the stream drops off the terrace
    drop = np.diff(wz)
    k = int(np.argmin(drop))
    j = k + int(np.argmax(drop[k:] > -0.05))
    foam = Geo()
    base = pts[j]
    for t in range(7):
        a_ = RNG.uniform(0, 2 * np.pi)
        p = base + RNG.uniform(0, 1.1) * np.array([math.cos(a_), math.sin(a_)])
        rock(foam, (p[0], p[1], wz[j] + 0.05), (0.45, 0.4, 0.18), "M_Foam", seed=500 + t, rough=0.2, sink=0.5)
    emit("Waterfall_Foam", foam, col)
    return ob, pts, wz, j


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def flared_roof(g, c3, yaw, half_out, half_in, z_eave, z_top, lift=0.35, mat="M_RoofTile", thick=0.16,
                ribs=True):
    """Hipped roof whose corners curl up, centred on c3 (x, y) and turned by yaw; tile ribs run down
    each slope, a fascia board runs round the eaves and rafter ends peek out underneath."""
    def ring(h, z, up):
        pts = []
        for sx, sy in ((-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)):
            corner = sx != 0 and sy != 0
            pts.append([sx * h, sy * h, z + (up if corner else 0.0)])
        return np.array(pts)

    outer = ring(half_out, z_eave, lift)
    inner = ring(half_in, z_top, 0.0)
    under = outer - [0, 0, thick]
    v = np.concatenate([outer, inner, under])
    f = [[i, (i + 1) % 8, 8 + (i + 1) % 8, 8 + i] for i in range(8)]
    f += [[8 + i for i in range(8)]]
    f += [[16 + i, 16 + (i + 1) % 8, (i + 1) % 8, i] for i in range(8)]
    f += [[16 + i for i in range(8)][::-1]]
    loc = (c3[0], c3[1], 0)
    rot = (0, 0, yaw)
    g.add(C.xf(v, loc, rot), f, mat)

    def W(q):
        return C.xf([q], loc, rot)[0]

    def along(rng_, u):  # point on a side, u in [-1, 1] from one corner through the mid to the other
        a, m, b = rng_
        return m + (a - m) * (-u) if u < 0 else m + (b - m) * u

    for side in range(4):
        o = (outer[2 * side], outer[2 * side + 1], outer[(2 * side + 2) % 8])
        i_ = (inner[2 * side], inner[2 * side + 1], inner[(2 * side + 2) % 8])
        n = max(int(2 * half_out / 0.27), 4)
        for k in range(n + 1):
            u = -0.92 + 1.84 * k / n
            po, pi_ = along(o, u), along(i_, u * half_in / half_out * 1.0)
            if ribs:
                beam(g, W(pi_ + [0, 0, 0.04]), W(po + [0, 0, 0.045]), 0.07, 0.05, "M_RoofTileDark")
            if k % 2 == 0:  # rafter ends
                dirn = po - np.array([0, 0, po[2]])
                dirn = dirn / (np.linalg.norm(dirn[:2]) + 1e-6) * 0.35
                q = po - [0, 0, thick + 0.04]
                beam(g, W(q - dirn), W(q - dirn * 0.1), 0.07, 0.07, "M_WoodDark")
        for a_, b_ in ((o[0], o[1]), (o[1], o[2])):  # fascia
            beam(g, W(a_ - [0, 0, thick * 0.5]), W(b_ - [0, 0, thick * 0.5]), 0.06, thick + 0.04, "M_WoodDark")
    top0, top1 = W([0, 0, z_top]), W([0, 0, z_top + 0.12])
    beam(g, top0, top1, half_in * 2, half_in * 2, "M_RoofTileDark")
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):  # hip ridges
        beam(g, W([sx * half_in, sy * half_in, z_top + 0.05]), W([sx * half_out, sy * half_out, z_eave + lift + 0.06]),
             0.13, 0.1, "M_RoofTileDark")


def gable_roof(g, s, cx, cy, hw, hd, z_eave, z_ridge, mat="M_RoofTile", oh=0.55, end_mat="M_Plaster"):
    """Gabled roof over a local rectangle, ridge along local x, with tile ribs, barge boards and
    onigawara end tiles on the ridge."""
    slope = (z_ridge - z_eave) / hd
    ze = z_eave - slope * oh
    for sy in (-1, 1):
        for dz, m in ((0.0, mat), (-0.12, "M_RoofTileDark")):
            v = [s.w3(cx - hw - oh, cy, z_ridge + dz), s.w3(cx + hw + oh, cy, z_ridge + dz),
                 s.w3(cx + hw + oh, cy + sy * (hd + oh), ze + dz), s.w3(cx - hw - oh, cy + sy * (hd + oh), ze + dz)]
            g.add(v, [[0, 1, 2, 3]], m)
        for x in np.arange(-hw - oh + 0.14, hw + oh - 0.1, 0.27):
            beam(g, s.w3(cx + x, cy + sy * 0.05, z_ridge + 0.04), s.w3(cx + x, cy + sy * (hd + oh), ze + 0.04),
                 0.08, 0.06, "M_RoofTileDark")
        beam(g, s.w3(cx - hw - oh, cy + sy * (hd + oh), ze - 0.05), s.w3(cx + hw + oh, cy + sy * (hd + oh), ze - 0.05),
             0.08, 0.16, "M_WoodDark")
        for sx in (-1, 1):  # barge boards along the gable edges
            beam(g, s.w3(cx + sx * (hw + oh), cy, z_ridge - 0.02), s.w3(cx + sx * (hw + oh), cy + sy * (hd + oh), ze - 0.02),
                 0.08, 0.2, "M_WoodDark")
    beam(g, s.w3(cx - hw - oh - 0.1, cy, z_ridge + 0.1), s.w3(cx + hw + oh + 0.1, cy, z_ridge + 0.1), 0.3, 0.24,
         "M_RoofTileDark")
    for sx in (-1, 1):
        box(g, s.w3(cx + sx * (hw + oh + 0.12), cy, z_ridge + 0.25), (0.14, 0.42, 0.42), "M_RoofTileDark",
            rot=(0, 0, s.yaw))
        g.add([s.w3(cx + sx * hw, cy - hd, z_eave), s.w3(cx + sx * hw, cy + hd, z_eave),
               s.w3(cx + sx * hw, cy, z_ridge)], [[0, 1, 2]], end_mat)
        beam(g, s.w3(cx + sx * hw, cy - hd, z_eave + 0.05), s.w3(cx + sx * hw, cy + hd, z_eave + 0.05), 0.1, 0.1,
             "M_WoodDark")


def torii(g, base, yaw, width=3.4, height=3.6, z=None):
    base = np.asarray(base, float)
    z = gz(base) if z is None else z
    d = np.array([math.cos(yaw), math.sin(yaw)])  # along the crossbeams

    def p(t, h):
        q = base + d * t
        return np.array([q[0], q[1], z + h])

    for s_ in (-1, 1):
        tube(g, p(s_ * width / 2, -0.5), p(s_ * width / 2 * 0.97, height), 0.17, 0.15, "M_Vermilion", n=12)
        SOLIDS.append(((base + d * s_ * width / 2), 0.3))
        kam = Geo()
        lathe(kam, [(0.24, -0.05), (0.24, 0.32), (0.19, 0.4)], "M_BlackLacquer", n=12)
        g.merge(kam, loc=p(s_ * width / 2, 0))
        for zz in (height * 0.78 - 0.18, height * 0.78 + 0.18):  # wedges at the nuki joint
            box(g, p(s_ * (width / 2 + 0.22), zz), (0.1, 0.22, 0.1), "M_Vermilion", rot=(0, 0, yaw))
    beam(g, p(-width / 2 - 0.5, height * 0.78), p(width / 2 + 0.5, height * 0.78), 0.18, 0.22, "M_Vermilion")
    beam(g, p(-width / 2 - 0.7, height + 0.05), p(width / 2 + 0.7, height + 0.05), 0.3, 0.2, "M_Vermilion")
    n = 10
    ext = width / 2 + 1.1
    pts = [p(t, height + 0.27 + 0.22 * abs(t / ext) ** 2.6) for t in np.linspace(-ext, ext, n + 1)]
    for a_, b_ in zip(pts[:-1], pts[1:]):  # kasagi with upswept ends
        beam(g, a_, b_, 0.36, 0.2, "M_BlackLacquer")
    beam(g, p(0, height * 0.78), p(0, height), 0.16, 0.16, "M_Vermilion")
    box(g, p(0, height * 0.89), (0.42, 0.08, 0.36), "M_BlackLacquer", rot=(0, 0, yaw))  # name plaque
    box(g, p(0, height * 0.89) + [*(np.array([-d[1], d[0]]) * 0.045), 0], (0.3, 0.02, 0.24), "M_Gold",
        rot=(0, 0, yaw), bevel=False)


def stone_lantern(g, base3, scale=1.0, yaw=0.0):
    """Kasuga-style toro: hexagonal base, shaft, platform, lit firebox, upturned roof, hoju."""
    b = np.asarray(base3, float)
    k = scale
    t = Geo()
    lathe(t, [(0.34, 0), (0.34, 0.1), (0.24, 0.18), (0.2, 0.2)], "M_StoneDark", n=6)
    lathe(t, [(0.1, 0.2), (0.09, 0.75), (0.12, 0.78), (0.12, 0.82), (0.09, 0.85)], "M_Stone", n=12)
    lathe(t, [(0.18, 0.85), (0.31, 0.92), (0.31, 0.98)], "M_StoneDark", n=6, cap_bottom=False)
    box(t, (0, 0, 1.15), (0.26, 0.26, 0.32), "M_Shoji", bevel=False)
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(t, (sx * 0.14, sy * 0.14, 1.15), (0.07, 0.07, 0.34), "M_Stone")
    lathe(t, [(0.26, 1.31), (0.44, 1.33), (0.47, 1.37), (0.3, 1.47), (0.1, 1.56)], "M_StoneDark", n=6,
          groove=0.0)
    lathe(t, [(0.05, 1.56), (0.08, 1.6), (0.09, 1.66), (0.06, 1.71), (0.0, 1.76)], "M_Stone", n=10)
    g.merge(t, loc=b, rot=(0, 0, yaw + math.pi / 6), scale=k)


def solid_rect(s, cx, cy, hx, hy, r=0.7):
    for x in np.arange(-hx, hx + 0.01, 1.0):
        for y in np.arange(-hy, hy + 0.01, 1.0):
            SOLIDS.append((s.w(cx + x, cy + y), r))


def spawn_slot(name, xy, z=None):
    i = [k for k, (n, _) in enumerate(SPAWNS) if n == name][0]
    SPAWNS[i] = (name, np.asarray(xy, float))
    if z is not None:
        SPAWN_Z[i] = z


SPAWNS = [(n, None) for n in (
    "Pagoda, ground floor", "Under the red bridge", "Foot of the waterfall", "Tea house veranda",
    "Lantern island", "Bamboo grove clearing", "Fox shrine", "Bell tower", "Village house", "Market stall",
    "Paddy hut", "Zen garden", "Spring on the terrace", "South gate")]
SPAWN_Z = {}


# ---------------------------------------------------------------------------
# landmarks
# ---------------------------------------------------------------------------

def build_pagoda(col, lights):
    g = Geo()
    c = PAGODA
    z0 = gz(c)
    box(g, (c[0], c[1], z0 - 0.25), (6.6, 6.6, 0.9), "M_StoneDark")
    for k in range(3):  # front steps
        box(g, (c[0], c[1] - 3.55 - 0.35 * k, z0 - 0.45 - 0.0 * k + 0.25 - 0.22 * k), (2.4, 0.4, 0.45), "M_Stone")
    zf = z0 + 0.2
    sizes = [4.2, 3.7, 3.25, 2.8, 2.35]
    z = zf
    for t, sz in enumerate(sizes):
        h = 2.0 if t == 0 else 1.25
        hz = sz / 2
        if t == 0:
            for (ax_, ay_), (bx_, by_) in (((-hz, hz), (hz, hz)), ((-hz, -hz), (-hz, hz)), ((hz, -hz), (hz, hz)),
                                           ((-hz, -hz), (-0.75, -hz)), ((0.75, -hz), (hz, -hz))):
                a = np.array([c[0] + ax_, c[1] + ay_])
                b = np.array([c[0] + bx_, c[1] + by_])
                m = (a + b) / 2
                L = np.linalg.norm(b - a)
                yaw = math.atan2(b[1] - a[1], b[0] - a[0])
                box(g, (m[0], m[1], z + h / 2), (L, 0.22, h), "M_Plaster", rot=(0, 0, yaw))
            box(g, (c[0], c[1] - hz, z + h - 0.2), (1.5, 0.24, 0.4), "M_Vermilion")
            box(g, (c[0], c[1] + hz - 0.6, z + 0.35), (1.2, 0.6, 0.7), "M_BlackLacquer")  # altar
            ball(g, (c[0], c[1] + hz - 0.6, z + 0.95), 0.18, "M_Gold")
        else:
            box(g, (c[0], c[1], z + h / 2), (sz, sz, h), "M_Plaster")
            box(g, (c[0], c[1], z + 0.05), (sz + 0.75, sz + 0.75, 0.1), "M_WoodDark")  # balcony deck
            rh = hz + 0.33
            for k in range(4):  # balcony railing
                a0 = np.array([[-rh, -rh], [rh, -rh], [rh, rh], [-rh, rh]][k])
                a1 = np.array([[-rh, -rh], [rh, -rh], [rh, rh], [-rh, rh]][(k + 1) % 4])
                for u in np.linspace(0, 1, max(int(2 * rh / 0.45), 2), endpoint=False):
                    q = a0 + (a1 - a0) * u
                    tube(g, (c[0] + q[0], c[1] + q[1], z + 0.1), (c[0] + q[0], c[1] + q[1], z + 0.55), 0.03,
                         mat="M_Vermilion", n=6)
                beam(g, (c[0] + a0[0], c[1] + a0[1], z + 0.55), (c[0] + a1[0], c[1] + a1[1], z + 0.55), 0.06, 0.06,
                     "M_Vermilion")
                beam(g, (c[0] + a0[0], c[1] + a0[1], z + 0.32), (c[0] + a1[0], c[1] + a1[1], z + 0.32), 0.03, 0.03,
                     "M_Vermilion")
            for k in range(4):  # door panels with gold studs on each face
                a = k * math.pi / 2
                n_ = np.array([math.cos(a), math.sin(a)])
                q = c + n_ * (hz + 0.01)
                box(g, (q[0], q[1], z + h * 0.48), (0.08, 0.75, 0.85), "M_WoodDark", rot=(0, 0, a))
                for zz in (z + h * 0.48 - 0.25, z + h * 0.48 + 0.25):
                    for u in (-0.2, 0.2):
                        t_ = np.array([-n_[1], n_[0]]) * u
                        ball(g, (q[0] + n_[0] * 0.05 + t_[0], q[1] + n_[1] * 0.05 + t_[1], zz), 0.025, "M_Gold")
        for sx in (-1, 1):
            for sy in (-1, 1):
                tube(g, (c[0] + sx * hz, c[1] + sy * hz, z), (c[0] + sx * hz, c[1] + sy * hz, z + h), 0.13,
                     mat="M_Vermilion", n=10)
        box(g, (c[0], c[1], z + h - 0.1), (sz + 0.1, sz + 0.1, 0.2), "M_Vermilion")
        for k in range(4):  # bracket sets (tokyo) carrying the eaves
            a = k * math.pi / 2
            n_ = np.array([math.cos(a), math.sin(a)])
            t_ = np.array([-n_[1], n_[0]])
            for u in (-0.62, 0.0, 0.62):
                q0 = c + n_ * (hz + 0.12) + t_ * u * hz
                q1 = c + n_ * (hz + 0.34) + t_ * u * hz
                box(g, (q0[0], q0[1], z + h - 0.2), (0.22, 0.2, 0.16), "M_WoodDark", rot=(0, 0, a))
                box(g, (q1[0], q1[1], z + h - 0.06), (0.5, 0.18, 0.12), "M_WoodDark", rot=(0, 0, a))
                box(g, (q1[0], q1[1], z + h + 0.03), (0.26, 0.42, 0.08), "M_Vermilion", rot=(0, 0, a))
        flared_roof(g, c, 0.0, hz + 0.95, hz * 0.62, z + h, z + h + 0.55, lift=0.38)
        for sx in (-1, 1):  # wind bells hanging from the upturned corners
            for sy in (-1, 1):
                q = c + np.array([sx, sy]) * (hz + 0.85)
                zt = z + h + 0.3
                tube(g, (q[0], q[1], zt), (q[0], q[1], zt - 0.3), 0.01, mat="M_Iron", n=4)
                bell = Geo()
                lathe(bell, [(0.09, 0.0), (0.08, 0.06), (0.05, 0.14), (0.0, 0.16)], "M_Bronze", n=10,
                      cap_bottom=False)
                g.merge(bell, loc=(q[0], q[1], zt - 0.46))
        z += h + 0.5
    tube(g, (c[0], c[1], z), (c[0], c[1], z + 2.4), 0.07, mat="M_Bronze", n=6)
    for k in range(6):
        zz = z + 0.4 + 0.28 * k
        tube(g, (c[0], c[1], zz), (c[0], c[1], zz + 0.06), 0.2, mat="M_Bronze", n=8)
    ball(g, (c[0], c[1], z + 2.5), 0.16, "M_Gold")
    for sx in (-1, 1):
        stone_lantern(g, (c[0] + sx * 2.3, c[1] - 4.6, gz(c + [sx * 2.3, -4.6])))
        SOLIDS.append((c + [sx * 2.3, -4.6], 0.45))
    solid_rect(Site(c, (0, 1)), 0, 0, 3.0, 3.0, 0.8)
    spawn_slot("Pagoda, ground floor", c + [0.6, 0.2], zf)
    return emit("Pagoda", g, col)


def build_stairs(col):
    g = Geo()
    z0, z1 = gz((STAIR_X, STAIR_Y0)), TERRACE_H
    n = int(math.ceil((z1 - z0) / 0.26))
    depth = (STAIR_Y1 - STAIR_Y0) / n
    for k in range(n):
        y = STAIR_Y0 + (k + 0.5) * depth
        top = z0 + (k + 1) * (z1 - z0) / n
        bot = min(gz((STAIR_X, y)), top) - 1.2
        box(g, (STAIR_X, y, (top + bot) / 2), (3.0, depth + 0.02, top - bot), "M_Stone" if k % 2 else "M_StoneDark")
    for s in (-1, 1):
        for k in range(n):
            y = STAIR_Y0 + (k + 0.5) * depth
            top = z0 + (k + 1) * (z1 - z0) / n + 0.55
            bot = gz((STAIR_X + s * 1.7, y)) - 1.0
            box(g, (STAIR_X + s * 1.7, y, (top + bot) / 2), (0.4, depth + 0.02, top - bot), "M_StoneDark")
    box(g, (STAIR_X, STAIR_Y1 + 0.6, z1 - 0.4), (3.8, 1.4, 0.8), "M_Stone")
    for y in (5.2, STAIR_Y0 + 0.2 * (STAIR_Y1 - STAIR_Y0)):
        z = gz((STAIR_X, y)) if y < STAIR_Y0 else z0 + 0.25 * (z1 - z0)
        torii(g, (STAIR_X, y), 0.0, width=3.9, height=3.6, z=z)
    torii(g, (STAIR_X, STAIR_Y1 + 0.6), 0.0, width=3.9, height=3.8, z=z1)
    for s in (-1, 1):
        for y in (6.2, STAIR_Y1 + 1.6):
            p = np.array([STAIR_X + s * 2.5, y])
            stone_lantern(g, (p[0], p[1], gz(p)), scale=0.9)
            SOLIDS.append((p, 0.45))
    return emit("Torii_Stairs", g, col)


def build_bridge(col):
    # where the bamboo path crosses the stream
    path = [np.asarray(p, float) for p in PATHS[1]]
    best = None
    for a, b in zip(path[:-1], path[1:]):
        for t in np.linspace(0, 1, 60):
            p = a + (b - a) * t
            d = float(stream_dist(np.array([p[0]]), np.array([p[1]]))[0])
            if best is None or d < best[0]:
                best = (d, p, (b - a) / np.linalg.norm(b - a))
    _, mid, ax = best
    side = np.array([-ax[1], ax[0]])
    L, W, H = 7.0, 2.0, 1.25
    zb = max(gz(mid - ax * L / 2), gz(mid + ax * L / 2)) + 0.05
    g = Geo()
    n = 16
    pts = [np.array([*(mid + ax * (t - 0.5) * L), zb + H * math.sin(math.pi * t)]) for t in np.linspace(0, 1, n + 1)]
    for k in range(n):
        beam(g, pts[k], pts[k + 1], W, 0.12, "M_Wood" if k % 2 else "M_WoodLight")
        for s in (-1, 1):
            beam(g, pts[k] + [*(side * s * W * 0.45), -0.18], pts[k + 1] + [*(side * s * W * 0.45), -0.18],
                 0.16, 0.22, "M_Vermilion")
    for s in (-1, 1):
        off = np.array([*(side * s * (W / 2 + 0.02)), 0])
        posts = list(range(0, n + 1, 2))
        for k in posts:
            tube(g, pts[k] + off, pts[k] + off + [0, 0, 0.95], 0.07, mat="M_Vermilion", n=6)
        for k0, k1 in zip(posts[:-1], posts[1:]):
            beam(g, pts[k0] + off + [0, 0, 0.9], pts[k1] + off + [0, 0, 0.9], 0.1, 0.1, "M_Vermilion")
            beam(g, pts[k0] + off + [0, 0, 0.45], pts[k1] + off + [0, 0, 0.45], 0.06, 0.06, "M_Vermilion")
        for k in (0, n):
            ball(g, pts[k] + off + [0, 0, 1.03], 0.1, "M_Bronze")
        for t in (0.2, 0.8):
            p = mid + ax * (t - 0.5) * L + side * s * W * 0.45
            tube(g, (p[0], p[1], gz(p) - 1.0), (p[0], p[1], zb + H * math.sin(math.pi * t) - 0.2), 0.12,
                 mat="M_WoodDark", n=6)
            SOLIDS.append((p, 0.3))
    for k in (0, n):
        SOLIDS.append((pts[k][:2], 1.0))
    under = mid + ax * 2.0
    spawn_slot("Under the red bridge", under, gz(under))
    return emit("Red_Bridge", g, col), mid, ax


def build_tea_house(col):
    g = Geo()
    s = Site(TEA_HOUSE, POND - TEA_HOUSE)
    pz = gz(TEA_HOUSE) + 0.55
    for lx in (-2.0, 0.0, 2.0):
        for ly in (-2.0, 0.0, 2.0, 3.0):
            p = s.w(lx, ly)
            tube(g, (p[0], p[1], min(gz(p), POND_Z) - 0.8), (p[0], p[1], pz), 0.1, mat="M_WoodDark", n=6)
            SOLIDS.append((p, 0.25))
    box(g, s.w3(0, 0.5, pz - 0.06), (4.4, 5.4, 0.14), "M_WoodLight", rot=(0, 0, s.yaw))
    box(g, s.w3(0, -0.4, pz + 0.04), (3.8, 3.2, 0.06), "M_Tatami", rot=(0, 0, s.yaw))
    for lx in np.arange(-2.0, 2.01, 0.8):  # shoji wall at the back
        box(g, s.w3(lx, -2.05, pz + 1.0), (0.06, 0.08, 2.0), "M_WoodDark", rot=(0, 0, s.yaw))
    box(g, s.w3(0, -2.05, pz + 1.0), (4.0, 0.04, 1.9), "M_Shoji", rot=(0, 0, s.yaw))
    for sx in (-1, 1):
        box(g, s.w3(sx * 2.05, -1.2, pz + 1.0), (0.04, 1.7, 1.9), "M_Shoji", rot=(0, 0, s.yaw))
        for ly in (-2.0, -1.2, -0.4):
            box(g, s.w3(sx * 2.05, ly, pz + 1.0), (0.08, 0.06, 2.0), "M_WoodDark", rot=(0, 0, s.yaw))
    for lx in (-2.05, 2.05):
        for ly in (-2.05, 2.0):
            tube(g, s.w3(lx, ly, pz), s.w3(lx, ly, pz + 2.25), 0.09, mat="M_WoodDark", n=6)
    beam(g, s.w3(-2.4, 2.0, pz + 0.55), s.w3(2.4, 2.0, pz + 0.55), 0.06, 0.08, "M_WoodDark")  # veranda rail
    flared_roof(g, s.w(0, -0.1), s.yaw, 3.2, 1.0, pz + 2.25, pz + 3.4, lift=0.3)
    box(g, s.w3(0, -0.6, pz + 0.22), (1.0, 0.7, 0.3), "M_BlackLacquer", rot=(0, 0, s.yaw))  # low table
    for lx, ly in ((-0.9, -0.6), (0.9, -0.6), (0, -1.3)):
        box(g, s.w3(lx, ly, pz + 0.1), (0.5, 0.5, 0.1), "M_Purple", rot=(0, 0, s.yaw))
    for k in range(3):  # stone steps up from the path
        p = s.w(2.7 + 0.45 * k, -1.0)
        tube(g, (p[0], p[1], gz(p) - 0.3), (p[0], p[1], pz - 0.18 * (k + 1)), 0.3, mat="M_Stone", n=7)
    solid_rect(s, 0, 0.5, 2.2, 2.7, 0.75)
    spawn_slot("Tea house veranda", s.w(0.9, 1.5), pz)
    return emit("Tea_House", g, col)


def build_pond_details(col):
    g = Geo()
    stone_lantern(g, (POND_ISLAND[0] + 0.2, POND_ISLAND[1] + 0.35, gz(POND_ISLAND + [0.2, 0.35])), 0.85)
    for k in range(4):
        a = 2 * np.pi * k / 4 + 0.5
        p = POND_ISLAND + 1.0 * np.array([math.cos(a), math.sin(a)])
        C.boulder(g, (p[0], p[1], gz(p)), (0.4, 0.35, 0.3), seed=600 + k, yaw=a, moss=0.4)
    for k, y in enumerate(np.linspace(-20.0, -16.2, 6)):  # stepping stones
        x = -2.7 - 0.25 * math.sin(k * 1.3)
        tube(g, (x, y, -1.2), (x, y, POND_Z + 0.14), 0.36, 0.33, "M_StoneDark", n=7, phase=k)
    rng = np.random.default_rng(66)
    for k in range(9):  # koi just under the surface
        a = rng.uniform(0, 2 * np.pi)
        p = POND + rng.uniform(1.8, 4.2) * np.array([math.cos(a), math.sin(a)])
        h = a + math.pi / 2
        d = np.array([math.cos(h), math.sin(h), 0])
        c3 = np.array([p[0], p[1], POND_Z - 0.18])
        fish = Geo()
        ball(fish, (0, 0, 0), 0.2, "M_Koi" if k % 3 else "M_White", squash=(0.45, 1.0, 0.3))
        tube(fish, (0, -0.18, 0), (0, -0.38, 0), 0.1, 0.0, "M_Koi", n=4)
        g.merge(fish, loc=c3, rot=(0, 0, h - math.pi / 2))
    spawn_slot("Lantern island", POND_ISLAND + [-0.35, -0.2], gz(POND_ISLAND + [-0.35, -0.2]))
    return emit("Pond_Details", g, col)


def build_house(col, c, front, idx):
    g = Geo()
    s = Site(c, front)
    z0 = gz(c)
    hx, hy, H = 3.0, 2.3, 2.6
    box(g, s.w3(0, 0, z0 + 0.0), (2 * hx + 0.3, 2 * hy + 0.3, 0.6), "M_StoneDark", rot=(0, 0, s.yaw))
    zf = z0 + 0.3
    walls = [((-hx, -hy), (hx, -hy), (-0.7, 0.7), (1.6, 2.6)),   # front: door and window
             ((-hx, hy), (hx, hy), None, (-1.0, 1.0)),
             ((-hx, -hy), (-hx, hy), None, (-0.6, 0.6)),
             ((hx, -hy), (hx, hy), None, None)]
    for a, b, door, window in walls:
        a, b = np.array(a), np.array(b)
        L = np.linalg.norm(b - a)
        d = (b - a) / L
        wd = d[0] * s.x + d[1] * s.f
        wyaw = math.atan2(wd[1], wd[0])
        cuts = sorted([r for r in (door, window) if r])
        spans, t = [], -L / 2
        for r in cuts:
            spans.append((t, r[0], 0, H))
            if r is door:
                spans.append((r[0], r[1], 2.05, H))
            else:
                spans.append((r[0], r[1], 0, 1.0))
                spans.append((r[0], r[1], 1.9, H))
                for k in np.arange(r[0] + 0.12, r[1], 0.16):  # lattice bars
                    m = a + d * (L / 2 + k)
                    box(g, s.w3(m[0], m[1], zf + 1.45), (0.04, 0.06, 0.9), "M_WoodDark", rot=(0, 0, wyaw))
            t = r[1]
        spans.append((t, L / 2, 0, H))
        for t0, t1, h0, h1 in spans:
            if t1 - t0 < 0.02:
                continue
            m = a + d * (L / 2 + (t0 + t1) / 2)
            box(g, s.w3(m[0], m[1], zf + (h0 + h1) / 2), (t1 - t0, 0.2, h1 - h0), "M_Plaster", rot=(0, 0, wyaw))
            if h0 == 0:
                box(g, s.w3(m[0], m[1], zf + min(h1, 0.9) / 2), (t1 - t0, 0.26, min(h1, 0.9)), "M_WoodDark",
                    rot=(0, 0, wyaw))
        for t_ in np.arange(0, L + 0.01, L / 4):
            m = a + d * t_
            SOLIDS.append((s.w(m[0], m[1]), 0.45))
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(g, s.w3(sx * hx, sy * hy, zf + H / 2), (0.24, 0.24, H), "M_WoodDark", rot=(0, 0, s.yaw))
    # timber frame over the plaster: studs, a rail above the doors and a top plate
    for (ax_, ay_), (bx_, by_), out in (((-hx, -hy), (hx, -hy), (0, -1)), ((-hx, hy), (hx, hy), (0, 1)),
                                         ((-hx, -hy), (-hx, hy), (-1, 0)), ((hx, -hy), (hx, hy), (1, 0))):
        a_, b_ = np.array([ax_, ay_]), np.array([bx_, by_])
        o = np.array(out) * 0.12
        for zz in (zf + 2.0, zf + H - 0.06):
            beam(g, s.w3(*(a_ + o), zz), s.w3(*(b_ + o), zz), 0.06, 0.12, "M_WoodDark")
        L = np.linalg.norm(b_ - a_)
        for t_ in np.arange(1.0, L - 0.5, 1.0):
            q = a_ + (b_ - a_) / L * t_ + o
            along = q[0] if out[1] else q[1]
            if out == (0, -1) and (-0.8 < along < 0.8 or 1.5 < along < 2.7):
                continue
            box(g, s.w3(q[0], q[1], zf + 2.0 + (H - 2.0) / 2), (0.1, 0.06, H - 2.0), "M_WoodDark",
                rot=(0, 0, s.yaw + (0 if out[1] else math.pi / 2)))
    for lx in (-0.75, 0.75):  # door frame
        box(g, s.w3(lx, -hy - 0.11, zf + 1.02), (0.1, 0.1, 2.05), "M_WoodDark", rot=(0, 0, s.yaw))
    box(g, s.w3(0, -hy - 0.12, zf + 2.1), (1.7, 0.12, 0.12), "M_WoodDark", rot=(0, 0, s.yaw))
    box(g, s.w3(2.1, -hy - 0.16, zf + 2.15), (1.3, 0.04, 0.32), "M_Straw", rot=(0, 0, s.yaw))  # bamboo blind
    for z_ in np.arange(zf + 2.0, zf + 2.3, 0.06):
        beam(g, s.w3(1.45, -hy - 0.19, z_), s.w3(2.75, -hy - 0.19, z_), 0.01, 0.01, "M_WoodDark")
    box(g, s.w3(0, -hy - 0.24, zf + 2.42), (1.1, 0.06, 0.32), "M_BlackLacquer", rot=(0, 0, s.yaw))  # sign
    box(g, s.w3(0, -hy - 0.28, zf + 2.42), (0.85, 0.02, 0.18), "M_Gold", rot=(0, 0, s.yaw), bevel=False)
    pot = Geo()
    lathe(pot, [(0.14, 0), (0.2, 0.25), (0.22, 0.32)], "M_Indigo", n=12)
    g.merge(pot, loc=s.w3(-1.3, -hy - 0.45, zf - 0.3))
    rock(g, s.w3(-1.3, -hy - 0.45, zf + 0.25), (0.28, 0.28, 0.3), "M_BushLight", seed=idx * 11 + 3, rough=0.12,
         sink=0.3, subdiv=2)
    for k in range(9):  # firewood stacked against the side wall
        lx_, lz_ = (k % 3) * 0.16, (k // 3) * 0.15
        tube(g, s.w3(hx + 0.32, -1.2 + lx_ * 2.4, zf - 0.22 + lz_), s.w3(hx + 0.32, -0.4 + lx_ * 2.4, zf - 0.22 + lz_),
             0.07, mat="M_Wood", n=7)
    SOLIDS.append((s.w(hx + 0.35, -0.6), 0.6))
    gable_roof(g, s, 0, 0, hx, hy, zf + H, zf + H + 1.25)
    for k, lx in enumerate((-0.45, 0.0, 0.45)):  # noren over the door
        box(g, s.w3(lx, -hy - 0.14, zf + 1.75), (0.42, 0.03, 0.6), "M_Indigo", rot=(0, 0, s.yaw))
    lamp = Geo()
    lathe(lamp, [(0.05, 1.55), (0.17, 1.65), (0.17, 1.95), (0.05, 2.05)], "M_PaperRed", n=8)
    g.merge(lamp, loc=(*s.w(-1.3, -hy - 0.35), zf))
    box(g, s.w3(0.8, 1.2, zf + 0.2), (3.0, 1.6, 0.4), "M_Tatami", rot=(0, 0, s.yaw))
    if idx == 0:
        spawn_slot("Village house", s.w(-1.6, -0.8), zf)
    return emit(f"Village_House_{idx + 1}", g, col)


def build_stall(col, c, front, idx):
    g = Geo()
    s = Site(c, front)
    z0 = gz(c)
    for lx in (-1.4, 1.4):
        for ly, h in ((-0.9, 2.1), (0.9, 2.4)):
            tube(g, s.w3(lx, ly, z0 - 0.2), s.w3(lx, ly, z0 + h), 0.07, mat="M_WoodDark", n=6)
            SOLIDS.append((s.w(lx, ly), 0.25))
    box(g, s.w3(0, 0.8, z0 + 0.45), (2.9, 0.6, 0.9), "M_Wood", rot=(0, 0, s.yaw))
    box(g, s.w3(0, 0.8, z0 + 0.92), (3.0, 0.7, 0.06), "M_WoodLight", rot=(0, 0, s.yaw))
    for k in range(5):
        lx0, lx1 = -1.6 + 3.2 * k / 5, -1.6 + 3.2 * (k + 1) / 5
        g.add([s.w3(lx0, -1.1, z0 + 2.15), s.w3(lx1, -1.1, z0 + 2.15), s.w3(lx1, 1.5, z0 + 2.35),
               s.w3(lx0, 1.5, z0 + 2.35)], [[0, 1, 2, 3]], "M_Indigo" if k % 2 else "M_White")
    rng = np.random.default_rng(80 + idx)
    for k in range(5):  # goods on the counter
        p = s.w3(-1.2 + 0.6 * k, 0.8, z0 + 0.95)
        if k % 2:
            jar = Geo()
            lathe(jar, [(0.1, 0), (0.16, 0.12), (0.12, 0.28), (0.06, 0.32)], "M_Porcelain" if k == 1 else "M_Celadon",
                  n=8)
            g.merge(jar, loc=p)
        else:
            for j in range(3):
                ball(g, p + [rng.uniform(-.12, .12), rng.uniform(-.1, .1), 0.08 + 0.05 * j], 0.08,
                     "M_Pumpkin" if idx else "M_Koi")
    for lx in (-1.2, 1.2):
        lamp = Geo()
        lathe(lamp, [(0.04, 0), (0.13, 0.08), (0.13, 0.3), (0.04, 0.38)], "M_PaperRed", n=8)
        g.merge(lamp, loc=s.w3(lx, 1.45, z0 + 1.7))
    solid_rect(s, 0, 0.5, 1.2, 0.5, 0.6)
    if idx == 0:
        spawn_slot("Market stall", s.w(0.4, -0.35), z0)
    return emit(f"Market_Stall_{idx + 1}", g, col)


def build_bell_tower(col):
    g = Geo()
    c = BELL_TOWER
    z0 = gz(c)
    box(g, (c[0], c[1], z0 + 0.0), (3.6, 3.6, 0.8), "M_StoneDark")
    zt = z0 + 0.4
    for sx in (-1, 1):
        for sy in (-1, 1):
            tube(g, (c[0] + sx * 1.35, c[1] + sy * 1.35, zt), (c[0] + sx * 1.25, c[1] + sy * 1.25, zt + 3.0), 0.14,
                 mat="M_Vermilion", n=6)
    for sy in (-1, 1):
        beam(g, (c[0] - 1.5, c[1] + sy * 1.25, zt + 2.8), (c[0] + 1.5, c[1] + sy * 1.25, zt + 2.8), 0.2, 0.22,
             "M_Vermilion")
    beam(g, (c[0], c[1] - 1.4, zt + 2.75), (c[0], c[1] + 1.4, zt + 2.75), 0.2, 0.2, "M_WoodDark")
    flared_roof(g, c, 0.0, 2.3, 0.7, zt + 3.0, zt + 4.1, lift=0.4)
    bell = Geo()
    lathe(bell, [(0.62, 0.95), (0.6, 1.05), (0.5, 1.6), (0.46, 2.3), (0.3, 2.5), (0.0, 2.55)], "M_Bronze", n=16,
          cap_bottom=False)
    for zz in (1.25, 1.9, 2.15):  # cast bands
        r_ = float(np.interp(zz, [0.95, 1.05, 1.6, 2.3, 2.5], [0.62, 0.6, 0.5, 0.46, 0.3])) + 0.012
        lathe(bell, [(r_, zz - 0.02), (r_, zz + 0.02)], "M_Bronze", n=16, cap_top=False, cap_bottom=False)
    g.merge(bell, loc=(c[0], c[1], zt))
    tube(g, (c[0], c[1], zt + 2.5), (c[0], c[1], zt + 2.75), 0.06, mat="M_Bronze", n=6)
    beam(g, (c[0] + 0.75, c[1] - 1.1, zt + 1.55), (c[0] + 0.75, c[1] + 1.1, zt + 1.55), 0.22, 0.22, "M_Wood")
    for y in (-0.7, 0.7):
        tube(g, (c[0] + 0.75, c[1] + y, zt + 1.6), (c[0] + 0.75, c[1] + y, zt + 2.75), 0.015, mat="M_Rope", n=4)
    SOLIDS.append((c, 2.4))
    spawn_slot("Bell tower", c + [-0.7, -0.75], zt)
    return emit("Bell_Tower", g, col)


def build_zen_garden(col):
    g = Geo()
    s = Site(ZEN, (0, -1))
    z0 = gz(ZEN)
    hx, hy = 3.6, 2.5
    box(g, s.w3(0, 0, z0 + 0.0), (2 * hx, 2 * hy, 0.2), "M_Gravel", rot=(0, 0, s.yaw))
    for sy in (-1, 1):
        beam(g, s.w3(-hx - 0.1, sy * (hy + 0.08), z0 + 0.1), s.w3(hx + 0.1, sy * (hy + 0.08), z0 + 0.1), 0.18, 0.24,
             "M_WoodDark")
    for sx in (-1, 1):
        beam(g, s.w3(sx * (hx + 0.08), -hy - 0.1, z0 + 0.1), s.w3(sx * (hx + 0.08), hy + 0.1, z0 + 0.1), 0.18, 0.24,
             "M_WoodDark")
    rocks = [(-1.6, 0.5, 0.75), (1.3, -0.6, 0.55), (2.4, 1.2, 0.4)]
    for y in np.arange(-hy + 0.25, hy - 0.1, 0.3):  # raked lines
        beam(g, s.w3(-hx + 0.15, y, z0 + 0.11), s.w3(hx - 0.15, y, z0 + 0.11), 0.05, 0.03, "M_White")
    for k, (lx, ly, r) in enumerate(rocks):
        C.boulder(g, s.w3(lx, ly, z0 + 0.1), (r * 1.3, r, r * 1.1), seed=700 + k, yaw=k, sink=0.3, subdiv=3,
                  mats=("M_RockDark", "M_Rock", "M_RockDark"))
        for rr in (r * 1.6, r * 2.0):
            ring = Geo()
            C.ring_path(ring, rr, 0.03, "M_White", n=18, m=3, z=z0 + 0.11)
            g.merge(ring, loc=(*s.w(lx, ly), 0))
        rock(g, s.w3(lx + 0.3, ly - 0.4, z0 + 0.1), (r * 0.9, r * 0.7, 0.12), "M_Moss", seed=710 + k, sink=0.5)
        SOLIDS.append((s.w(lx, ly), r * 1.3))
    solid_rect(s, 0, 0, hx - 0.2, hy - 0.2, 0.8)
    spawn_slot("Zen garden", s.w(-2.9, -1.9), z0 + 0.1)
    return emit("Zen_Garden", g, col)


def build_spring(col):
    g = Geo()
    for k in range(13):
        a = 2 * np.pi * k / 13
        p = SPRING + 2.45 * np.array([math.cos(a), math.sin(a)])
        if float(stream_dist(np.array([p[0]]), np.array([p[1]]))[0]) < 1.3 and p[1] < SPRING[1]:
            continue
        C.boulder(g, (p[0], p[1], gz(p)), (0.55, 0.45, 0.4), seed=800 + k, yaw=a, moss=0.5)
        SOLIDS.append((p, 0.55))
    # bamboo spout feeding the pool
    p = SPRING + [-1.0, 2.6]
    z = gz(p)
    tube(g, (p[0], p[1], z - 0.2), (p[0], p[1], z + 0.9), 0.06, mat="M_Bamboo", n=6)
    tube(g, (p[0], p[1] + 0.2, z + 0.8), (p[0] + 0.3, p[1] - 1.1, z + 0.55), 0.06, mat="M_Bamboo", n=6)
    spawn_slot("Spring on the terrace", SPRING + [-2.9, 0.9])
    return emit("Spring", g, col)


def build_waterfall_spot(col, pts, wz, j):
    base = pts[j]
    tang = pts[min(j + 2, len(pts) - 1)] - pts[max(j - 2, 0)]
    tang /= np.linalg.norm(tang)
    side = np.array([-tang[1], tang[0]])
    g = Geo()
    for k, s_ in enumerate((-1, 1)):
        for t in range(3):
            p = base + side * s_ * (1.9 + 0.5 * t) - tang * (0.8 + 0.9 * t)
            C.boulder(g, (p[0], p[1], gz(p)), (0.8, 0.7, 0.9 + 0.3 * t), seed=900 + k * 5 + t, yaw=t, moss=0.6, subdiv=3)
            SOLIDS.append((p, 0.8))
    p = base + side * 2.4 + tang * 0.6
    spawn_slot("Foot of the waterfall", p, gz(p))
    return emit("Waterfall_Rocks", g, col)


def fox(g, base3, yaw):
    b = np.asarray(base3, float)
    box(g, b + [0, 0, 0.2], (0.45, 0.45, 0.4), "M_StoneDark", rot=(0, 0, yaw))
    ball(g, b + [0, 0, 0.62], 0.2, "M_White", squash=(0.8, 1.0, 1.3))
    ball(g, b + [*(np.array([math.cos(yaw), math.sin(yaw)]) * 0.05), 0.95], 0.13, "M_White", squash=(1, 1.1, 1))
    for s_ in (-1, 1):
        q = b + [*(np.array([-math.sin(yaw), math.cos(yaw)]) * s_ * 0.07), 1.05]
        tube(g, q, q + [0, 0, 0.14], 0.045, 0.0, "M_White", n=4)
    box(g, b + [*(np.array([math.cos(yaw), math.sin(yaw)]) * 0.12), 0.78], (0.08, 0.22, 0.12), "M_Vermilion",
        rot=(0, 0, yaw))


def build_shrine(col):
    g = Geo()
    s = Site(SHRINE, (1, -0.15))
    z0 = gz(SHRINE)
    box(g, s.w3(0, -0.5, z0 + 0.15), (1.6, 1.4, 0.6), "M_StoneDark", rot=(0, 0, s.yaw))
    box(g, s.w3(0, -0.5, z0 + 0.9), (1.0, 0.9, 0.9), "M_Wood", rot=(0, 0, s.yaw))
    box(g, s.w3(0, -0.04, z0 + 0.9), (0.6, 0.04, 0.6), "M_BlackLacquer", rot=(0, 0, s.yaw))
    gable_roof(g, s, 0, -0.5, 0.5, 0.45, z0 + 1.35, z0 + 1.85, oh=0.3, end_mat="M_Wood")
    for k in range(3):
        C.boulder(g, s.w3(-1.0 + k, -2.2, z0 + 0.3), (1.2, 1.0, 1.5), seed=950 + k, yaw=k, moss=0.5, subdiv=3)
    torii(g, s.w(0, 2.0), s.yaw, width=1.9, height=2.3)
    for sx in (-1, 1):
        fox(g, s.w3(sx * 1.0, 0.8, z0), s.yaw + math.pi / 2 + sx * 0.25)
        SOLIDS.append((s.w(sx * 1.0, 0.8), 0.4))
    box(g, s.w3(0, 0.35, z0 + 0.25), (0.6, 0.35, 0.5), "M_Wood", rot=(0, 0, s.yaw))  # offering box
    solid_rect(s, 0, -0.9, 1.4, 1.2, 0.8)
    spawn_slot("Fox shrine", s.w(0.0, 1.05), z0)
    return emit("Fox_Shrine", g, col)


def build_paddy_hut(col):
    g = Geo()
    s = Site(PADDY_HUT, (-1, 0.2))
    z0 = gz(PADDY_HUT)
    for lx in (-1.2, 1.2):
        for ly in (-1.2, 1.2):
            tube(g, s.w3(lx, ly, z0 - 0.3), s.w3(lx, ly, z0 + 2.0), 0.08, mat="M_WoodDark", n=6)
            SOLIDS.append((s.w(lx, ly), 0.25))
    C.pyramid_roof(g, s, 1.75, z0 + 1.9, z0 + 3.2, mat="M_Straw")
    for k, (lx, ly, zz) in enumerate(((-0.6, -0.7, 0), (0.1, -0.7, 0), (-0.25, -0.7, 0.45))):
        bale = Geo()
        lathe(bale, [(0.0, -0.4), (0.2, -0.38), (0.26, 0.0), (0.2, 0.38), (0.0, 0.4)], "M_Straw", n=8)
        g.merge(bale, loc=s.w3(lx, ly, z0 + 0.26 + zz), rot=(math.pi / 2, 0, s.yaw))
    SOLIDS.append((s.w(-0.25, -0.7), 0.6))
    # scarecrow at the paddy edge
    p = s.w(1.2, 3.0)
    z = gz(p)
    beam(g, (p[0], p[1], z - 0.4), (p[0], p[1], z + 1.8), 0.08, 0.08, "M_Wood")
    beam(g, s.w3(1.2 - 0.7, 3.0, z + 1.35), s.w3(1.2 + 0.7, 3.0, z + 1.35), 0.07, 0.07, "M_Wood")
    box(g, (p[0], p[1], z + 1.2), (0.5, 0.25, 0.6), "M_Indigo", rot=(0, 0, s.yaw))
    ball(g, (p[0], p[1], z + 1.75), 0.17, "M_Paper")
    tube(g, (p[0], p[1], z + 1.85), (p[0], p[1], z + 2.15), 0.45, 0.0, "M_Straw", n=8)
    SOLIDS.append((p, 0.4))
    spawn_slot("Paddy hut", s.w(0.45, 0.4), z0)
    return emit("Paddy_Hut", g, col)


def build_gate(col):
    g = Geo()
    s = Site(GATE, (0, 1))
    z0 = gz(GATE)
    for sx in (-1, 1):
        box(g, s.w3(sx * 1.9, 0, z0 + 1.8), (0.45, 0.45, 4.6), "M_WoodDark", rot=(0, 0, s.yaw))
        box(g, s.w3(sx * 1.9, 0, z0 - 0.1), (0.7, 0.7, 0.6), "M_StoneDark", rot=(0, 0, s.yaw))
        for lx in np.arange(2.4, 7.5, 1.0):  # side walls into the mountain
            p = s.w(sx * lx, 0)
            box(g, (p[0], p[1], gz(p) + 0.9), (1.02, 0.5, 3.2), "M_Plaster", rot=(0, 0, s.yaw))
            box(g, (p[0], p[1], gz(p) + 0.0), (1.04, 0.56, 1.2), "M_StoneDark", rot=(0, 0, s.yaw))
            box(g, (p[0], p[1], gz(p) + 2.55), (1.1, 0.8, 0.12), "M_RoofTileDark", rot=(0, 0, s.yaw))
            SOLIDS.append((p, 0.6))
    for k, lx in enumerate(np.arange(-1.65, 1.66, 0.3)):  # closed plank doors
        box(g, s.w3(lx, 0, z0 + 1.6), (0.29, 0.14, 3.2), "M_Wood" if k % 2 else "M_WoodLight", rot=(0, 0, s.yaw))
    for z in (0.7, 2.6):
        box(g, s.w3(0, 0.1, z0 + z), (3.4, 0.08, 0.14), "M_Iron", rot=(0, 0, s.yaw))
    box(g, s.w3(0, 0, z0 + 3.9), (4.6, 0.5, 0.35), "M_WoodDark", rot=(0, 0, s.yaw))
    gable_roof(g, s, 0, 0, 2.6, 1.0, z0 + 4.15, z0 + 5.0, oh=0.6, end_mat="M_WoodDark")
    for sx in (-1, 1):
        p = s.w(sx * 3.2, 1.6)
        stone_lantern(g, (p[0], p[1], gz(p)), 0.9)
        SOLIDS.append((p, 0.45))
    SOLIDS.extend([(s.w(x, 0), 0.5) for x in np.arange(-2.0, 2.01, 0.8)])
    spawn_slot("South gate", s.w(2.5, 1.0), z0)
    return emit("South_Gate", g, col)


def build_bamboo_clearing(col):
    g = Geo()
    c = BAMBOO
    for k in range(4):  # log seats round a stone in the middle of the grove
        a = 2 * np.pi * k / 4 + 0.4
        p = c + 1.3 * np.array([math.cos(a), math.sin(a)])
        q = np.array([-math.sin(a), math.cos(a)]) * 0.5
        tube(g, (p[0] - q[0], p[1] - q[1], gz(p) + 0.15), (p[0] + q[0], p[1] + q[1], gz(p) + 0.15), 0.18,
             mat="M_Wood", n=6)
        SOLIDS.append((p, 0.5))
    spawn_slot("Bamboo grove clearing", c + [0.1, -0.2])
    return emit("Bamboo_Clearing", g, col)


# ---------------------------------------------------------------------------
# vegetation
# ---------------------------------------------------------------------------

def blossom_tree_geo(seed, mats=("M_Sakura", "M_SakuraLight", "M_SakuraWhite"), scale=1.0):
    """Gnarled trunk that forks twice; the crown is many small blossom clumps round the branch tips."""
    rng = np.random.default_rng(seed)
    g = Geo()
    lean = rng.uniform(0, 2 * np.pi)
    ld = np.array([math.cos(lean), math.sin(lean), 0])
    n = 10
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rings = []
    for i, t in enumerate(np.linspace(0, 1, 7)):
        c = ld * 0.45 * t * t + [0.06 * math.sin(4 * t + seed), 0.06 * math.cos(3 * t), -0.25 + 2.05 * t]
        r = 0.2 + 0.1 * max(0.0, 0.25 - t) / 0.25
        bumps = 1 + 0.12 * np.sin(a * 3 + i + seed)
        rings.append(c + (r * (1 - 0.25 * t) * bumps)[:, None] * np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1))
    loft(g, rings, "M_Bark")
    top = ld * 0.45 + [0, 0, 1.8]
    tips = []
    for k in range(3):
        a1 = lean + 2 * np.pi * k / 3 + rng.uniform(-0.4, 0.4)
        mid = top + [0.85 * math.cos(a1), 0.85 * math.sin(a1), rng.uniform(0.6, 0.9)]
        tube(g, top, mid, 0.15, 0.1, "M_Bark", n=7)
        for j in range(2):
            a2 = a1 + rng.uniform(-0.8, 0.8)
            tip = mid + [0.75 * math.cos(a2), 0.75 * math.sin(a2), rng.uniform(0.3, 0.8)]
            tube(g, mid, tip, 0.09, 0.04, "M_Bark", n=6)
            for _ in range(2):  # twigs
                tw = tip + rng.uniform(-0.45, 0.45, 3) * [1, 1, 0.5] + [0, 0, 0.2]
                tube(g, tip, tw, 0.035, 0.012, "M_Bark", n=4)
            tips.append(tip)
    for k, tip in enumerate(tips):
        for j in range(4):
            c = tip + rng.uniform(-0.55, 0.55, 3) * [1, 1, 0.6] + [0, 0, 0.25]
            rr = rng.uniform(0.42, 0.66)
            rock(g, c, (rr, rr, rr * 0.78), mats[(k + j) % len(mats)], seed=seed * 31 + k * 4 + j, rough=0.07,
                 sink=0.0, subdiv=1)
    for j in range(5):  # a few clumps closing the crown at the top
        c = top + [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), 1.6 + rng.uniform(0, 0.5)]
        rr = rng.uniform(0.55, 0.75)
        rock(g, c, (rr, rr, rr * 0.75), mats[j % len(mats)], seed=seed * 37 + j, rough=0.07, sink=0.0, subdiv=1)
    if scale != 1.0:
        g.v = (np.array(g.v) * scale).tolist()
    return g


def pine_geo(seed, scale=1.0):
    """Layered pine: ragged star-shaped tiers drooping at their tips, over a visible trunk."""
    rng = np.random.default_rng(seed)
    g = Geo()
    h = rng.uniform(3.4, 4.6)
    tube(g, (0, 0, -0.3), (0, 0, h * 0.85), 0.2, 0.06, "M_Bark", n=8)
    tiers = 5
    n = 18
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 1)
    for k in range(tiers):
        z0 = h * (0.22 + 0.15 * k)
        r = 1.45 * (1 - 0.17 * k)
        rr = r * np.where(np.arange(n) % 2 == 0, 1.0, 0.68) * rng.uniform(0.9, 1.1, n)
        droop = np.where(np.arange(n) % 2 == 0, 0.32, 0.12)
        rim = np.stack([rr * np.cos(a), rr * np.sin(a), z0 - droop], axis=1)
        apex = np.array([[0, 0, z0 + 0.95 - 0.08 * k]])
        under = np.array([[0, 0, z0 + 0.12]])
        v = np.concatenate([rim, apex, under])
        f = [[i, (i + 1) % n, n] for i in range(n)] + [[(i + 1) % n, i, n + 1] for i in range(n)]
        g.add(v, f, "M_Pine" if k % 2 else "M_PineLight")
    g.v = (np.array(g.v) * scale).tolist()
    return g


def bamboo_geo(seed):
    """Segmented culm with node rings; twigs near the top carry drooping leaf blades."""
    rng = np.random.default_rng(seed)
    g = Geo()
    h = rng.uniform(5.0, 7.5)
    lean = rng.uniform(0, 2 * np.pi)
    d = np.array([math.cos(lean), math.sin(lean), 0]) * rng.uniform(0.1, 0.5)
    segs = 7
    pts = [d * (t * t) + [0, 0, h * t - 0.2] for t in np.linspace(0, 1, segs + 1)]
    mat = "M_Bamboo" if seed % 3 else "M_BambooDark"
    for k in range(segs):
        tube(g, pts[k], pts[k + 1], 0.075 - 0.004 * k, 0.07 - 0.004 * k, mat, n=7)
        tube(g, pts[k + 1] - [0, 0, 0.03], pts[k + 1] + [0, 0, 0.03], 0.083 - 0.004 * k, mat="M_BambooDark", n=7)
    for k in range(segs - 3, segs + 1):
        for _ in range(2):
            a = rng.uniform(0, 2 * np.pi)
            dv = np.array([math.cos(a), math.sin(a), 0.35])
            base = pts[k]
            twig = base + dv * rng.uniform(0.4, 0.8)
            tube(g, base, twig, 0.015, 0.008, "M_BambooDark", n=3)
            for j in range(5):
                b_ = rng.uniform(0, 2 * np.pi)
                ld = np.array([math.cos(b_), math.sin(b_), 0])
                L = rng.uniform(0.3, 0.45)
                tip = twig + ld * L - [0, 0, L * 0.45]
                side = np.array([-ld[1], ld[0], 0]) * 0.045
                mid = twig + ld * L * 0.45 - [0, 0, L * 0.1]
                g.add([twig, mid + side, tip, mid - side], [[0, 1, 2, 3]], "M_BambooLeaf" if j % 2 else "M_PineLight")
    return g


def build_vegetation(col, veg):
    for name, pts, make, r in (
            ("Sakura_Trees", veg["sakura"], lambda k: blossom_tree_geo(1000 + k), 0.5),
            ("Maple_Trees", veg["maple"], lambda k: blossom_tree_geo(1100 + k, ("M_Maple", "M_MapleLight"), 0.85), 0.45),
            ("Bamboo", veg["bamboo"], lambda k: bamboo_geo(1200 + k), 0.3)):
        g = Geo()
        for k, p in enumerate(pts):
            g.merge(make(k), loc=(p[0], p[1], gz(p)), rot=(0, 0, RNG.uniform(0, 6.3)))
            SOLIDS.append((p, r))
        emit(name, g, col)
    g = Geo()
    for k, p in enumerate(veg["pine"]):
        g.merge(pine_geo(1300 + k, RNG.uniform(0.9, 1.5)), loc=(p[0], p[1], gz(p)), rot=(0, 0, RNG.uniform(0, 6.3)))
        SOLIDS.append((p, 0.5))
    emit("Pines", g, col)
    g = Geo()
    for k, p in enumerate(veg["bush"]):
        b = C.bush_geo(1400 + k, "M_FlowerWhite" if k % 3 == 0 else None)
        if k % 3 == 0:
            remap(b, {"M_Bush": "M_Sakura", "M_BushLight": "M_SakuraLight"})  # azaleas
        g.merge(b, loc=(p[0], p[1], gz(p)), rot=(0, 0, RNG.uniform(0, 6.3)), scale=RNG.uniform(0.8, 1.2))
        SOLIDS.append((p, 0.8))
    emit("Bushes", g, col)
    g = Geo()
    for k, p in enumerate(veg["rock"]):
        s = RNG.uniform(0.35, 0.95)
        C.boulder(g, (p[0], p[1], gz(p)), (s * RNG.uniform(1.0, 1.5), s, s * 0.7), moss=0.35,
             seed=1500 + k, yaw=RNG.uniform(0, 6.3))
        SOLIDS.append((p, s * 1.2))
    emit("Rocks", g, col)


# ---------------------------------------------------------------------------
# treasure props (200), Japanese household and temple things
# ---------------------------------------------------------------------------

LACQUER = {"M_Wood": "M_BlackLacquer", "M_WoodDark": "M_Vermilion", "M_Iron": "M_Gold"}


def jp_item_geo(kind):
    g = Geo()
    if kind == "paper_lantern":
        lathe(g, [(0.06, 0.05), (0.2, 0.15), (0.24, 0.35), (0.2, 0.55), (0.06, 0.65)], "M_PaperRed", n=10)
        tube(g, (0, 0, 0.0), (0, 0, 0.07), 0.09, mat="M_BlackLacquer", n=8)
        tube(g, (0, 0, 0.63), (0, 0, 0.7), 0.09, mat="M_BlackLacquer", n=8)
    elif kind == "stone_lantern_small":
        stone_lantern(g, (0, 0, 0), 0.5)
    elif kind == "sake_bottle":
        lathe(g, [(0.07, 0), (0.1, 0.08), (0.11, 0.17), (0.04, 0.27), (0.03, 0.32), (0.045, 0.34)], "M_Porcelain", n=9)
        tube(g, (0, 0, 0.1), (0, 0, 0.14), 0.112, mat="M_Indigo", n=9)
    elif kind == "sake_barrel":
        g.merge(remap(C.item_geo("barrel"), {"M_Wood": "M_Straw", "M_Iron": "M_Rope"}))
        box(g, (0, -0.3, 0.45), (0.3, 0.04, 0.35), "M_PaperRed")
    elif kind == "teapot":
        lathe(g, [(0.08, 0), (0.15, 0.05), (0.16, 0.13), (0.1, 0.2), (0.04, 0.22), (0.0, 0.24)], "M_Iron", n=10)
        tube(g, (0.12, 0, 0.1), (0.24, 0, 0.18), 0.03, 0.018, "M_Iron", n=5)
        tube(g, (-0.1, 0, 0.21), (0, 0, 0.32), 0.012, mat="M_Iron", n=4)
        tube(g, (0, 0, 0.32), (0.1, 0, 0.21), 0.012, mat="M_Iron", n=4)
    elif kind == "tea_set":
        box(g, (0, 0, 0.02), (0.55, 0.35, 0.04), "M_BlackLacquer")
        t = Geo()
        lathe(t, [(0.05, 0), (0.09, 0.04), (0.09, 0.1), (0.04, 0.15), (0.0, 0.16)], "M_Celadon", n=8)
        tube(t, (0.08, 0, 0.07), (0.15, 0, 0.11), 0.018, mat="M_Celadon", n=4)
        g.merge(t, loc=(-0.12, 0, 0.04))
        for x, y in ((0.1, -0.08), (0.18, 0.06), (0.04, 0.09)):
            cup = Geo()
            lathe(cup, [(0.03, 0), (0.045, 0.06)], "M_Porcelain", n=7, cap_top=False)
            g.merge(cup, loc=(x, y, 0.04))
    elif kind == "rice_bales":
        for p, ang in (((0, -0.22, 0.26), 0.0), ((0, 0.3, 0.26), 0.0), ((0, 0.04, 0.68), 0.0)):
            b = Geo()
            lathe(b, [(0.0, -0.42), (0.2, -0.4), (0.26, 0.0), (0.2, 0.4), (0.0, 0.42)], "M_Straw", n=8)
            for z in (-0.25, 0.25):
                tube(b, (0, 0, z - 0.02), (0, 0, z + 0.02), 0.255, mat="M_Rope", n=8)
            g.merge(b, loc=p, rot=(0, math.pi / 2, ang))
    elif kind == "umbrella":
        u = Geo()
        tube(u, (0, 0, 0), (0, 0, 0.35), 0.62, 0.05, "M_PaperRed", n=12)
        tube(u, (0, 0, 0.3), (0, 0, -0.55), 0.015, mat="M_Wood", n=4)
        g.merge(u, loc=(0, 0, 0.45), rot=(1.2, 0, 0))
    elif kind == "fan":
        a = np.linspace(math.radians(20), math.radians(160), 9)
        v = [[0, -0.12, 0.02]] + [[0.3 * math.cos(t), -0.12 + 0.3 * math.sin(t), 0.02 + 0.01 * (i % 2)]
                                  for i, t in enumerate(a)]
        g.add(v, [[0, i + 1, i + 2] for i in range(8)], "M_PaperRed")
        g.add(np.array(v) - [0, 0, 0.005], [[0, i + 2, i + 1] for i in range(8)], "M_Wood")
    elif kind == "daruma":
        ball(g, (0, 0, 0.2), 0.2, "M_Vermilion", subdiv=2, squash=(1, 0.95, 1.05))
        ball(g, (0, -0.13, 0.26), 0.11, "M_White", squash=(1, 0.45, 0.9))
        for s_ in (-1, 1):
            ball(g, (s_ * 0.04, -0.175, 0.28), 0.022, "M_BlackLacquer")
    elif kind == "kokeshi":
        tube(g, (0, 0, 0), (0, 0, 0.3), 0.07, 0.08, "M_Vermilion", n=9)
        ball(g, (0, 0, 0.39), 0.1, "M_WoodLight", subdiv=2)
        ball(g, (0, 0.015, 0.42), 0.1, "M_BlackLacquer", subdiv=1, squash=(1.02, 1.0, 0.8))
        tube(g, (0, 0, 0.12), (0, 0, 0.16), 0.085, mat="M_Gold", n=9)
    elif kind == "maneki_neko":
        ball(g, (0, 0, 0.16), 0.16, "M_White", squash=(1, 0.85, 1.05))
        ball(g, (0, -0.02, 0.38), 0.13, "M_White")
        for s_ in (-1, 1):
            tube(g, (s_ * 0.07, -0.02, 0.46), (s_ * 0.09, -0.02, 0.56), 0.045, 0.0, "M_White", n=4)
        tube(g, (0.12, -0.05, 0.2), (0.15, -0.08, 0.42), 0.04, mat="M_White", n=5)
        tube(g, (0, -0.02, 0.27), (0, -0.02, 0.3), 0.12, mat="M_Vermilion", n=8)
        ball(g, (0, -0.13, 0.26), 0.03, "M_Gold")
        ball(g, (0, -0.12, 0.08), 0.06, "M_Gold", squash=(1.2, 0.5, 1))
    elif kind == "bonsai":
        box(g, (0, 0, 0.07), (0.45, 0.3, 0.14), "M_Indigo")
        tube(g, (0, 0, 0.14), (0.08, 0, 0.3), 0.04, 0.03, "M_Bark", n=5)
        tube(g, (0.08, 0, 0.3), (-0.08, 0.02, 0.45), 0.03, 0.02, "M_Bark", n=5)
        for p, r in (((0.15, 0.0, 0.33), 0.12), ((-0.1, 0.03, 0.47), 0.14), ((0.03, -0.03, 0.52), 0.09)):
            ball(g, p, r, "M_Pine", squash=(1.3, 1, 0.5))
    elif kind == "bamboo_basket":
        lathe(g, [(0.14, 0), (0.22, 0.12), (0.25, 0.25)], "M_Straw", n=10, cap_top=False)
        rim = Geo()
        C.ring_path(rim, 0.25, 0.02, "M_Bamboo", n=14, m=4, z=0.25)
        g.merge(rim)
    elif kind == "geta":
        for x in (-0.07, 0.07):
            box(g, (x, 0, 0.07), (0.1, 0.24, 0.03), "M_WoodLight")
            for y in (-0.06, 0.06):
                box(g, (x, y, 0.03), (0.1, 0.02, 0.06), "M_WoodLight")
            beam(g, (x - 0.04, 0.02, 0.09), (x, 0.08, 0.11), 0.015, 0.015, "M_Vermilion")
            beam(g, (x + 0.04, 0.02, 0.09), (x, 0.08, 0.11), 0.015, 0.015, "M_Vermilion")
    elif kind == "kitsune_mask":
        m = Geo()
        ball(m, (0, 0, 0), 0.16, "M_White", squash=(0.9, 1.25, 0.45))
        tube(m, (0, 0.14, 0.0), (0, 0.27, -0.02), 0.07, 0.02, "M_White", n=5)
        for s_ in (-1, 1):
            tube(m, (s_ * 0.08, -0.1, 0.0), (s_ * 0.14, -0.22, 0.02), 0.045, 0.0, "M_White", n=4)
            box(m, (s_ * 0.05, 0.02, 0.07), (0.06, 0.02, 0.01), "M_Vermilion", rot=(0, 0, s_ * 0.4))
        g.merge(m, loc=(0, 0, 0.07))
    elif kind == "katana_stand":
        for x in (-0.3, 0.3):
            box(g, (x, 0, 0.18), (0.06, 0.18, 0.36), "M_BlackLacquer")
        box(g, (0, 0, 0.02), (0.75, 0.2, 0.04), "M_BlackLacquer")
        beam(g, (-0.55, 0.0, 0.34), (0.25, 0.0, 0.37), 0.035, 0.012, "M_Steel")
        tube(g, (0.25, 0, 0.37), (0.29, 0, 0.375), 0.05, mat="M_Gold", n=8)
        tube(g, (0.29, 0, 0.375), (0.55, 0, 0.385), 0.02, mat="M_Vermilion", n=6)
    elif kind == "kabuto":
        lathe(g, [(0.2, 0.05), (0.19, 0.13), (0.14, 0.22), (0.07, 0.27), (0.0, 0.28)], "M_BlackLacquer", n=12)
        lathe(g, [(0.2, 0.06), (0.3, 0.0), (0.32, -0.04)], "M_BlackLacquer", n=12, cap_top=False,
              cap_bottom=False)
        for s_ in (-1, 1):
            beam(g, (0, -0.17, 0.2), (s_ * 0.18, -0.24, 0.42), 0.03, 0.015, "M_Gold")
        g.v = (np.array(g.v) + [0, 0, 0.04]).tolist()
    elif kind == "wind_chime":
        tube(g, (0, 0, -0.1), (0, 0, 0.9), 0.02, mat="M_Bamboo", n=4)
        tube(g, (0, 0, 0.88), (0.25, 0, 0.88), 0.015, mat="M_Bamboo", n=4)
        bell = Geo()
        lathe(bell, [(0.07, 0), (0.06, 0.06), (0.0, 0.1)], "M_Celadon", n=8, cap_bottom=False)
        g.merge(bell, loc=(0.25, 0, 0.74))
        box(g, (0.25, 0, 0.6), (0.06, 0.005, 0.14), "M_Paper")
    elif kind == "ema_rack":
        for x in (-0.5, 0.5):
            tube(g, (x, 0, -0.1), (x, 0, 1.0), 0.04, mat="M_Wood", n=5)
        for z in (0.55, 0.9):
            beam(g, (-0.55, 0, z), (0.55, 0, z), 0.04, 0.04, "M_Wood")
            for k in range(5):
                box(g, (-0.4 + 0.2 * k, -0.03, z - 0.1), (0.15, 0.015, 0.1), "M_WoodLight",
                    rot=(0, 0, 0.1 * (k % 2)))
    elif kind == "incense_burner":
        lathe(g, [(0.12, 0.08), (0.2, 0.12), (0.21, 0.24), (0.18, 0.28)], "M_Bronze", n=10, cap_top=False)
        for k in range(3):
            a = 2 * np.pi * k / 3
            tube(g, (0.14 * math.cos(a), 0.14 * math.sin(a), 0.1), (0.17 * math.cos(a), 0.17 * math.sin(a), 0),
                 0.025, mat="M_Bronze", n=4)
        for k in range(3):
            tube(g, (0.04 * k - 0.04, 0, 0.2), (0.04 * k - 0.04, 0.02, 0.45), 0.006, mat="M_Vermilion", n=3)
    elif kind == "gourd":
        ball(g, (0, 0, 0.12), 0.12, "M_Gourd", subdiv=2)
        ball(g, (0, 0, 0.28), 0.08, "M_Gourd", subdiv=2)
        tube(g, (0, 0, 0.35), (0, 0, 0.4), 0.02, mat="M_Wood", n=5)
        tube(g, (0, 0, 0.2), (0, 0, 0.215), 0.065, mat="M_Vermilion", n=7)
    elif kind == "zabuton":
        box(g, (0, 0, 0.05), (0.5, 0.5, 0.1), "M_Purple", rot=(0, 0, 0.3))
        for sx in (-1, 1):
            ball(g, C.xf([[sx * 0.25, 0.25, 0.08]], rot=(0, 0, 0.3))[0], 0.03, "M_Gold")
    elif kind == "origami_crane":
        v = [[0, 0, 0.06], [0.12, 0, 0.1], [-0.12, 0, 0.12], [0, 0.18, 0.04], [0, -0.18, 0.04], [0, 0, 0.0]]
        g.add(v, [[0, 3, 1], [0, 1, 4], [0, 4, 2], [0, 2, 3], [5, 1, 3], [5, 4, 1], [5, 2, 4], [5, 3, 2]],
              "M_PaperRed")
        tube(g, (0.1, 0, 0.09), (0.17, 0, 0.18), 0.015, 0.0, "M_PaperRed", n=3)
        g.v = (np.array(g.v) * 1.3).tolist()
    elif kind == "taiko_drum":
        d = Geo()
        lathe(d, [(0.22, -0.2), (0.27, 0), (0.22, 0.2)], "M_Vermilion", n=12)
        for z in (-0.2, 0.2):
            tube(d, (0, 0, z - 0.01), (0, 0, z + 0.01), 0.225, mat="M_Paper", n=12)
        g.merge(d, loc=(0, 0, 0.42), rot=(math.pi / 2, 0, 0))
        for x in (-0.18, 0.18):
            beam(g, (x, -0.15, 0), (x, 0, 0.3), 0.04, 0.04, "M_WoodDark")
            beam(g, (x, 0.15, 0), (x, 0, 0.3), 0.04, 0.04, "M_WoodDark")
    elif kind == "koinobori":
        tube(g, (0, 0, -0.3), (0, 0, 2.6), 0.04, mat="M_Bamboo", n=5)
        for k, (mat, z) in enumerate((("M_BlackLacquer", 2.3), ("M_Vermilion", 1.75), ("M_Indigo", 1.25))):
            L = 1.1 - 0.15 * k
            tube(g, (0.05, 0, z), (0.05 + L, 0.08, z - 0.25), 0.16 - 0.02 * k, 0.03, mat, n=7)
            ball(g, (0.15, -0.1, z + 0.03), 0.035, "M_White")
    elif kind == "lacquer_box":
        g.merge(remap(C.item_geo("small_chest"), LACQUER))
    else:
        return C.item_geo(kind)
    return g


ITEM_RADIUS = {"koinobori": 0.6, "katana_stand": 0.45, "rice_bales": 0.55, "umbrella": 0.5, "ema_rack": 0.55,
               "sake_barrel": 0.4, "crate": 0.45, "taiko_drum": 0.35, "stone_lantern_small": 0.3}
ZONE_ITEMS = {
    "terrace": ["stone_lantern_small", "incense_burner", "ema_rack", "bonsai", "scroll", "candles", "wind_chime",
                "kitsune_mask", "kabuto", "taiko_drum", "origami_crane", "coin_pouch", "lacquer_box"],
    "stairs": ["stone_lantern_small", "paper_lantern", "origami_crane", "coin_pouch", "kitsune_mask"],
    "village": ["sake_barrel", "rice_bales", "crate", "sack", "bucket", "umbrella", "geta", "maneki_neko", "daruma",
                "kokeshi", "tea_set", "zabuton", "paper_lantern", "koinobori", "bamboo_basket", "sake_bottle"],
    "market": ["bamboo_basket", "sake_bottle", "teapot", "daruma", "kokeshi", "fan", "gourd", "crate", "sack",
               "umbrella", "maneki_neko"],
    "stream": ["bucket", "gourd", "rope_coil", "umbrella", "geta", "fan", "origami_crane", "bamboo_basket",
               "stone_lantern_small"],
    "pond": ["tea_set", "teapot", "zabuton", "fan", "umbrella", "origami_crane", "stone_lantern_small", "sake_bottle",
             "wind_chime"],
    "bamboo": ["bamboo_basket", "mushrooms", "kitsune_mask", "katana_stand", "gourd", "sack", "sake_bottle",
               "rope_coil"],
    "shrine": ["kitsune_mask", "ema_rack", "candles", "incense_burner", "sake_bottle", "coin_pouch", "origami_crane"],
    "paddies": ["rice_bales", "bamboo_basket", "bucket", "sack", "umbrella", "gourd", "geta"],
    "gate": ["katana_stand", "kabuto", "paper_lantern", "crate", "sake_barrel", "umbrella"],
    "meadow": ["mushrooms", "daruma", "gourd", "umbrella", "geta", "fan", "sake_bottle", "origami_crane", "rope_coil",
               "kokeshi", "lacquer_box", "bonsai", "coin_pouch", "kabuto", "koinobori"],
}


def zone_sampler(zone):
    if zone == "terrace":
        return lambda: RNG.uniform([-14, 14.5], [20, 26])
    if zone == "stairs":
        return lambda: np.array([STAIR_X, STAIR_Y0 - 1.6 if RNG.random() < 0.5 else STAIR_Y1 + 2.2]) + \
            RNG.uniform([-3.5, -1.3], [3.5, 1.3])
    if zone == "village":
        return lambda: RNG.uniform([6, -11], [22, 5])
    if zone == "market":
        return lambda: STALLS[RNG.integers(2)][0] + RNG.uniform(-3.2, 3.2, 2)
    if zone == "stream":
        pts = resample(STREAM[3:], 0.5)

        def f():
            p = pts[RNG.integers(len(pts))]
            a = RNG.uniform(0, 2 * np.pi)
            return p + RNG.uniform(1.9, 3.6) * np.array([math.cos(a), math.sin(a)])
        return f
    if zone == "pond":
        def f():
            a = RNG.uniform(0, 2 * np.pi)
            return POND + RNG.uniform(POND_R + 0.9, POND_R + 3.4) * np.array([math.cos(a), math.sin(a)])
        return f
    if zone == "bamboo":
        return near(BAMBOO, 7.5)
    if zone == "shrine":
        return near(SHRINE + [1.5, 0], 4.0)
    if zone == "paddies":
        return lambda: RNG.uniform([6.5, -27], [24, -10.5])
    if zone == "gate":
        return near(GATE + [0, 4.5], 4.5)
    return in_valley(qmax=0.84)


ZONE_PLAN = [("terrace", 26), ("stairs", 8), ("village", 26), ("market", 12), ("stream", 18), ("pond", 20),
             ("bamboo", 18), ("shrine", 8), ("paddies", 16), ("gate", 8), ("meadow", 40)]


def prop_ok(p, rad, taken, players):
    if float(rim_q(*p)) > 0.87 or wet(p, rad + 0.15) or stair_zone(p, rad):
        return None
    e = 0.5
    sl = math.hypot(gz(p + [e, 0]) - gz(p - [e, 0]), gz(p + [0, e]) - gz(p - [0, e])) / (2 * e)
    if sl > 0.45:
        return None
    for c, r in SOLIDS:
        if math.hypot(*(p - c)) < r + rad:
            return None
    for c, r in taken:
        if math.hypot(*(p - c)) < r + rad + 0.35:
            return None
    for c in players:
        if math.hypot(*(p - c)) < 1.0 + rad:
            return None
    if float(path_dist(np.array([p[0]]), np.array([p[1]]))[0]) < 0.6 + rad:
        return None
    return gz(p)


def build_props(col, players):
    kinds = sorted({k for v in ZONE_ITEMS.values() for k in v})
    meshes = {}
    for k in kinds:
        ob = emit(f"_item_{k}", jp_item_geo(k), col)
        meshes[k] = ob.data
        meshes[k].name = f"Item_{k}"
        bpy.data.objects.remove(ob)
    taken, placed = [], []
    for zone, count in ZONE_PLAN:
        pool = list(ZONE_ITEMS[zone])
        RNG.shuffle(pool)
        sample = zone_sampler(zone)
        got = tries = 0
        while got < count and tries < 30000:
            tries += 1
            kind = pool[got % len(pool)]
            rad = ITEM_RADIUS.get(kind, 0.3)
            p = sample()
            z = prop_ok(p, rad, taken, players)
            if z is None:
                continue
            taken.append((p, rad))
            placed.append((kind, zone, (float(p[0]), float(p[1]), z), float(RNG.uniform(0, 2 * np.pi))))
            got += 1
        print(f"[props] {zone:8s} {got}/{count}")
    for i, (kind, zone, loc, yaw) in enumerate(placed):
        ob = bpy.data.objects.new(f"Prop_{i + 1:03d}_{kind}", meshes[kind])
        ob.location = loc
        ob.rotation_euler = (0, 0, yaw)
        ob["kind"], ob["item"], ob["zone"], ob["index"] = "treasure_prop", kind, zone, i + 1
        col.objects.link(ob)
    return placed, meshes


def build_ground_cover(col):
    rng = np.random.default_rng(4242)
    sol = np.array([c for c, _ in SOLIDS])
    rad = np.array([r for _, r in SOLIDS])

    def clear(p, path_gap=0.9, qmax=0.93):
        if float(rim_q(*p)) > qmax or wet(p, 0.3) or stair_zone(p, 0.2):
            return None
        if any(math.hypot(*(p - c)) < r0 for c, r0, _, _ in PADS):
            return None
        if np.any(np.hypot(sol[:, 0] - p[0], sol[:, 1] - p[1]) < rad * 0.75):
            return None
        if float(path_dist(np.array([p[0]]), np.array([p[1]]))[0]) < path_gap:
            return None
        e = 0.5
        if math.hypot(gz(p + [e, 0]) - gz(p - [e, 0]), gz(p + [0, e]) - gz(p - [0, e])) / (2 * e) > 0.55:
            return None
        return gz(p)

    def anywhere():
        return rng.uniform([-27, -30], [27, 28])

    def petals(r):
        g = Geo()
        for _ in range(6):
            q = np.array([*r.uniform(-0.3, 0.3, 2), 0.03])
            a = r.uniform(0, 2 * np.pi)
            dv = np.array([math.cos(a), math.sin(a), 0]) * 0.035
            sv = np.array([-dv[1], dv[0], 0]) * 0.6
            g.add([q - dv, q + sv, q + dv, q - sv], [[0, 1, 2, 3]], "M_SakuraLight" if r.random() < 0.5 else "M_Sakura")
        return g

    def litter(r):
        g = Geo()
        for _ in range(5):
            q = np.array([*r.uniform(-0.35, 0.35, 2), 0.03])
            a = r.uniform(0, 2 * np.pi)
            dv = np.array([math.cos(a), math.sin(a), 0]) * 0.09
            sv = np.array([-dv[1], dv[0], 0]) * 0.2
            g.add([q - dv, q + sv, q + dv, q - sv], [[0, 1, 2, 3]], "M_Straw" if r.random() < 0.6 else "M_BambooLeaf")
        return g

    sak = veg["sakura"] if (veg := PLAN.get("veg")) else []
    n = 1 if FAST else 3
    C.scatter_cover(col, "Grass", 900 * n, anywhere, clear,
                    lambda r: C.grass_tuft_geo(r, ("M_Grass", "M_GrassLight"), h=0.32), rng)
    C.scatter_cover(col, "Flowers", 130 * n, anywhere, clear,
                    lambda r: C.flower_geo(r, ("M_FlowerPink", "M_FlowerWhite", "M_FlowerYellow", "M_FlowerBlue")
                                           [int(r.integers(4))]), rng)
    C.scatter_cover(col, "Fallen_Petals", 160 * n, lambda: sak[int(rng.integers(len(sak)))] +
                    rng.uniform(-3.2, 3.2, 2), lambda p: clear(p, 0.0), petals, rng)
    C.scatter_cover(col, "Leaf_Litter", 110 * n, near(BAMBOO, 8.0), lambda p: clear(p, 0.3), litter, rng)
    # water's edge: reeds round the pond and along the lower stream, lily pads on the pond
    def edge(p):
        if float(rim_q(*p)) > 0.9 or stair_zone(p, 0.3):
            return None
        sd = float(stream_dist(np.array([p[0]]), np.array([p[1]]))[0])
        pd_ = math.hypot(*(p - POND))
        if not (1.35 < sd < 1.9 or POND_R - 0.1 < pd_ < POND_R + 0.7):
            return None
        if math.hypot(*(p - TEA_HOUSE)) < 3.2 or p[1] < -19.5 and abs(p[0] + 2.7) < 1.2:
            return None
        if np.any(np.hypot(sol[:, 0] - p[0], sol[:, 1] - p[1]) < rad * 0.8):
            return None
        return gz(p)

    def edge_sample():
        if rng.random() < 0.5:
            a = rng.uniform(0, 2 * np.pi)
            return POND + rng.uniform(POND_R - 0.1, POND_R + 0.7) * np.array([math.cos(a), math.sin(a)])
        q = spts[int(rng.integers(len(spts)))]
        a = rng.uniform(0, 2 * np.pi)
        return q + 1.6 * np.array([math.cos(a), math.sin(a)])

    def lily(r):
        g = Geo()
        for k in range(int(r.integers(2, 5))):
            q = r.uniform(-0.5, 0.5, 2)
            rr = r.uniform(0.16, 0.3)
            a0 = r.uniform(0, 2 * np.pi)
            ring = [[q[0] + rr * math.cos(a0 + t), q[1] + rr * math.sin(a0 + t), 0.01]
                    for t in np.linspace(0.35, 2 * np.pi, 12)]
            g.add([[q[0], q[1], 0.012]] + ring, [[0, i + 1, i + 2] for i in range(11)], "M_LilyPad")
            if k == 0 and r.random() < 0.5:
                for m in range(7):
                    b = 2 * np.pi * m / 7
                    d_ = np.array([math.cos(b), math.sin(b), 0])
                    sv = np.array([-d_[1], d_[0], 0]) * 0.03
                    c0 = np.array([q[0], q[1], 0.02])
                    g.add([c0 + sv, c0 + d_ * 0.09 + [0, 0, 0.07], c0 - sv], [[0, 1, 2]], "M_FlowerPink")
                ball(g, (q[0], q[1], 0.05), 0.025, "M_FlowerCentre", subdiv=0)
        return g

    def pond_surface(p):
        if math.hypot(*(p - POND)) > POND_R - 0.6 or math.hypot(*(p - POND_ISLAND)) < 1.7:
            return None
        if abs(p[0] + 2.7) < 0.9 and p[1] < -15.8 or math.hypot(*(p - TEA_HOUSE)) < 3.0:
            return None
        return POND_Z + 0.02

    spts = resample(STREAM[3:], 0.5)
    C.scatter_cover(col, "Reeds", 30 * n, edge_sample, edge, lambda r: C.reeds_geo(r), rng)
    C.scatter_cover(col, "Lily_Pads", 9 * n, near(POND, POND_R), pond_surface, lily, rng)
    C.scatter_cover(col, "Ferns", 40 * n, lambda: near(BAMBOO, 8.0)() if rng.random() < 0.5 else anywhere(),
                    lambda p: clear(p, 0.5) if (math.hypot(*(p - BAMBOO)) < 8.5 or float(rim_q(*p)) > 0.8) else None,
                    lambda r: C.fern_geo(r), rng)
    C.scatter_cover(col, "Tall_Grass", 50 * n, anywhere, lambda p: clear(p, 1.3),
                    lambda r: C.grass_tuft_geo(r, ("M_Grass", "M_GrassLight"), h=0.7, blades=20), rng)

    def outcrop(r):
        g = Geo()
        for k in range(int(r.integers(1, 4))):
            sz = r.uniform(0.6, 1.6)
            C.boulder(g, (r.uniform(-1, 1), r.uniform(-1, 1), sz * 0.15), (sz * 1.5, sz, sz * 0.8),
                      seed=int(r.integers(1 << 30)), yaw=r.uniform(0, 6.3), subdiv=2, moss=0.2, tilt=0.3)
        return g

    C.scatter_cover(col, "Mountain_Outcrops", 18 * n, lambda: rng.uniform(-40, 40, 2),
                    lambda p: gz(p) if 0.93 < float(rim_q(*p)) < 1.3 and not (abs(p[0] - GATE[0]) < 5 and p[1] < -25)
                    else None, outcrop, rng)

    def road_slab():  # stepping slabs set into the main road
        pl = PATHS[0]
        i = int(rng.integers(len(pl) - 1))
        a_, b_ = np.asarray(pl[i], float), np.asarray(pl[i + 1], float)
        return a_ + (b_ - a_) * rng.random() + rng.uniform(-0.45, 0.45, 2)

    def slab(r):
        g = Geo()
        sz = r.uniform(0.35, 0.55)
        C.boulder(g, (0, 0, -0.02), (sz, sz * r.uniform(0.7, 1.0), 0.07), seed=int(r.integers(1 << 30)),
                  subdiv=1, strata=0, sink=0.5, tilt=0.02, mats=("M_Stone", "M_StoneDark", "M_StoneLight"))
        return g

    C.scatter_cover(col, "Road_Slabs", 40 * n, road_slab,
                    lambda p: None if wet(p, 0.1) or stair_zone(p) else gz(p), slab, rng, tries_per=3)
    spts = resample(STREAM, 0.5)
    C.scatter_cover(col, "Stream_Pebbles", 90 * n, lambda: spts[int(rng.integers(len(spts)))] +
                    rng.uniform(-2.4, 2.4, 2), lambda p: clear(p, 0.3, 0.95), lambda r: C.pebbles_geo(r), rng)


PLAN = {}


def build_spawns(col):
    chest = remap(C.chest_geo(False), LACQUER)
    out = []
    for i, (label, xy) in enumerate(SPAWNS):
        z = SPAWN_Z.get(i, gz(xy))
        yaw = float(RNG.uniform(0, 2 * np.pi))
        em = C.marker(f"TreasureSpawn_{i + 1:02d}", col, (xy[0], xy[1], z), "treasure_spawn", yaw, index=i + 1,
                      label=label)
        g = Geo().merge(chest)
        if i in (0, 7):
            C.coin_pile(g, (0.75, -0.35, 0.0), 0.12, seed=i)
        ch = emit(f"TreasureChest_{i + 1:02d}", g, col)
        ch.parent = em
        ch["kind"] = "treasure_chest"
        out.append((label, (xy[0], xy[1], z)))
        SOLIDS.append((np.asarray(xy, float), 0.9))
    players = []
    for k in range(6):
        p = GATE + [1.5 * (k % 3 - 1), 3.2 + 1.6 * (k // 3)]
        C.marker(f"PlayerSpawn_{k + 1:02d}", col, (p[0], p[1], gz(p)), "player_spawn", math.pi / 2, size=0.9,
                 index=k + 1)
        players.append(p)
    return out, players


# ---------------------------------------------------------------------------
# washi map: ink drawing of the layout
# ---------------------------------------------------------------------------

def draw_valley_map(path, spawns, players, veg, bridge, props=None):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[map] Pillow not available, skipping the map")
        return
    S, E = 1400, 36.0
    oy = -1.0  # the valley is a little taller than wide; centre the drawing on it
    xs = np.linspace(-E, E, S)
    X, Y = np.meshgrid(xs, xs[::-1] + oy)
    H = ground(X, Y)
    Q = rim_q(X, Y)
    noise = fbm(X / 2.0, Y / 2.0, 4, seed=31)

    def px(p):
        return ((p[0] + E) / (2 * E) * (S - 1), (E - (p[1] - oy)) / (2 * E) * (S - 1))

    img = hexc("#f3ead6")[None, None, :] * (0.94 + 0.06 * noise[..., None])
    gy, gx = np.gradient(H)
    shade = np.clip(1 + 0.5 * (gx * -0.7 + gy * 0.7), 0.75, 1.1)
    mtn = ss(0.9, 1.05, Q)
    img = img * (1 - mtn[..., None]) + img * shade[..., None] * mtn[..., None] * 0.9
    terr = ss(4.2, 4.8, H) * (1 - mtn)
    img = img * (1 - 0.06 * terr[..., None])
    water = (stream_dist(X, Y) < 1.1) | (np.hypot(X - POND[0], Y - POND[1]) < POND_R) | \
            (np.hypot(X - SPRING[0], Y - SPRING[1]) < 2.2)
    island = np.hypot(X - POND_ISLAND[0], Y - POND_ISLAND[1]) < 1.0
    water &= ~island
    img = np.where(water[..., None], hexc("#9cc9d3")[None, None, :] * (0.95 + 0.05 * noise[..., None]), img)
    for c, hs in PADDIES:
        inside = (np.abs(X - c[0]) < hs[0]) & (np.abs(Y - c[1]) < hs[1])
        img = np.where(inside[..., None], hexc("#c5dccf")[None, None, :], img)
    edge = water.astype(np.float32)
    edge = (K.box_blur(edge, 1, 1) > 0.05) & (K.box_blur(edge, 1, 1) < 0.95)
    img = np.where(edge[..., None], hexc("#3d5f73")[None, None, :], img)
    # terrace edge and valley rim as ink contours
    for lvl, colr in ((2.6, "#6d5a45"), (8.0, "#4a3b2c")):
        q = (H > lvl)
        e = (np.diff(q.astype(int), axis=0, prepend=q[:1].astype(int)) != 0) | \
            (np.diff(q.astype(int), axis=1, prepend=q[:, :1].astype(int)) != 0)
        img = np.where(e[..., None], hexc(colr)[None, None, :], img)
    r = np.maximum(abs(X), abs(Y - oy)) / E
    burn = ss(0.86, 1.0, r + 0.05 * noise)
    img = img * (1 - 0.35 * burn[..., None])
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    fp = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    try:
        title_f, label_f, small_f = (ImageFont.truetype(fp, s_) for s_ in (54, 22, 19))
    except OSError:
        title_f = label_f = small_f = ImageFont.load_default()
    ink, red, pink = (45, 35, 28), (190, 45, 35), (226, 128, 158)

    rng = np.random.default_rng(5)
    for _ in range(170):  # mountain peaks round the rim
        p = rng.uniform(-E, E, 2) + [0, oy]
        qq = float(rim_q(*p))
        if not 1.0 < qq < 1.5 or (abs(p[0] - GATE[0]) < 4 and p[1] < -26):
            continue
        x, y = px(p)
        sz = rng.uniform(12, 22)
        d.line([(x - sz, y + sz * 0.5), (x, y - sz * 0.6), (x + sz, y + sz * 0.5)], fill=ink, width=3)
        d.line([(x, y - sz * 0.6), (x + sz * 0.35, y + sz * 0.5)], fill=(120, 100, 80), width=2)
    for pl in PATHS:
        for a, b in zip(pl[:-1], pl[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            n = max(int(np.linalg.norm(b - a) / 0.8), 1)
            for k in range(0, n, 2):
                d.line([px(a + (b - a) * k / n), px(a + (b - a) * min(k + 1, n) / n)], fill=(150, 95, 55), width=4)
    for p in veg["bamboo"][::3]:
        x, y = px(p)
        d.line([(x, y - 7), (x, y + 7)], fill=(95, 140, 60), width=3)
    for p in veg["sakura"]:
        x, y = px(p)
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=pink, outline=(150, 70, 95))
    for p in veg["maple"]:
        x, y = px(p)
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(214, 90, 50), outline=(130, 50, 30))
    # landmarks
    x, y = px(PAGODA)
    for k in range(5):
        w_ = 34 - 6 * k
        d.polygon([(x - w_, y + 26 - 11 * k), (x + w_, y + 26 - 11 * k), (x + w_ - 8, y + 18 - 11 * k),
                   (x - w_ + 8, y + 18 - 11 * k)], fill=(70, 80, 95), outline=ink)
    d.line([(x, y - 30), (x, y - 44)], fill=ink, width=3)

    def torii_icon(c, sz=12):
        x, y = px(c)
        d.line([(x - sz, y - sz * 0.8), (x + sz, y - sz * 0.8)], fill=red, width=5)
        d.line([(x - sz * 0.8, y - sz * 0.35), (x + sz * 0.8, y - sz * 0.35)], fill=red, width=3)
        for s_ in (-1, 1):
            d.line([(x + s_ * sz * 0.6, y - sz * 0.8), (x + s_ * sz * 0.6, y + sz * 0.6)], fill=red, width=4)

    for y_ in (5.2, STAIR_Y0 + 1.2, STAIR_Y1 + 0.6):
        torii_icon((STAIR_X, y_))
    a0, a1 = px(bridge[0] - bridge[1] * 3.5), px(bridge[0] + bridge[1] * 3.5)
    d.line([a0, a1], fill=red, width=12)
    for c, (w_, h_) in [(c, (52, 40)) for c, _ in HOUSES] + [(c, (32, 22)) for c, _ in STALLS] + \
            [(TEA_HOUSE, (40, 40)), (PADDY_HUT, (26, 26)), (BELL_TOWER, (34, 34))]:
        x, y = px(c)
        d.rectangle([x - w_ / 2, y - h_ / 2, x + w_ / 2, y + h_ / 2], fill=(214, 196, 160), outline=ink, width=3)
        d.line([(x - w_ / 2, y - h_ / 2), (x + w_ / 2, y + h_ / 2)], fill=ink, width=1)
        d.line([(x - w_ / 2, y + h_ / 2), (x + w_ / 2, y - h_ / 2)], fill=ink, width=1)
    x, y = px(ZEN)
    d.rectangle([x - 50, y - 36, x + 50, y + 36], fill=(236, 232, 222), outline=ink, width=3)
    for k in range(-3, 4):
        d.line([(x - 44, y + 9 * k), (x + 44, y + 9 * k)], fill=(170, 160, 150), width=1)
    torii_icon(SHRINE + [1.2, -0.1], 9)
    x, y = px(GATE)
    d.rectangle([x - 30, y - 8, x + 30, y + 8], fill=(90, 70, 55), outline=ink)
    for name, c, (ox, oy_) in (("Pagoda", PAGODA, (-46, -82)), ("Zen Bahçesi", ZEN, (-66, 40)),
                               ("Kaynak", SPRING, (-40, -50)), ("Çan Kulesi", BELL_TOWER, (-56, 24)),
                               ("Torii Merdiveni", (STAIR_X, 10.5), (26, -10)), ("Şelale", (-10.8, 11.6), (-96, -8)),
                               ("Kırmızı Köprü", bridge[0], (-150, 6)), ("Koi Gölü", POND, (-48, 26)),
                               ("Çay Evi", TEA_HOUSE, (-110, -30)), ("Köy", HOUSES[0][0] + [3, 0], (-20, -64)),
                               ("Pazar", STALLS[1][0], (-30, 16)), ("Pirinç Tarlaları", (15.4, -19.5), (-90, 74)),
                               ("Bambu Ormanı", BAMBOO, (-74, 40)), ("Tilki Tapınağı", SHRINE, (-30, -46)),
                               ("Güney Kapısı", GATE, (40, -16))):
        x, y = px(c)
        d.text((x + ox, y + oy_), name, font=label_f, fill=ink)
    for i, p in enumerate(spawns):
        x, y = px(p)
        for s_ in (1, -1):
            d.line([(x - 12, y - 12 * s_), (x + 12, y + 12 * s_)], fill=red, width=6)
        d.text((x + 13, y - 2), str(i + 1), font=small_f, fill=red)
    for p in players:
        x, y = px(p)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(40, 70, 130), outline=ink)
    for _, _, p, _ in props or []:
        x, y = px(p)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(214, 160, 40), outline=ink)
    title = "Kiraz Vadisi"
    tw = d.textlength(title, font=title_f)
    d.rectangle([S / 2 - tw / 2 - 26, 30, S / 2 + tw / 2 + 26, 108], fill=(243, 234, 214), outline=red, width=4)
    d.text((S / 2 - tw / 2, 38), title, font=title_f, fill=ink)
    lx, ly = 60, S - 200
    rows = [("x", f"Hazine noktası (1–{len(spawns)})"), ("p", "Oyuncu başlangıcı"), ("l", "Patika")]
    if props:
        rows.append(("o", f"Hazine olabilecek eşya ({len(props)})"))
    d.rectangle([lx - 16, ly - 16, lx + 380, ly + 34 * len(rows) + 4], fill=(243, 234, 214), outline=ink, width=3)
    for k, (kind, text) in enumerate(rows):
        y = ly + 34 * k
        if kind == "x":
            for s_ in (1, -1):
                d.line([(lx, y + 12 - 9 * s_), (lx + 18, y + 12 + 9 * s_)], fill=red, width=5)
        elif kind == "p":
            d.ellipse([lx + 3, y + 5, lx + 16, y + 18], fill=(40, 70, 130), outline=ink)
        elif kind == "l":
            d.line([(lx, y + 12), (lx + 22, y + 12)], fill=(150, 95, 55), width=4)
        else:
            d.ellipse([lx + 4, y + 6, lx + 15, y + 17], fill=(214, 160, 40), outline=ink)
        d.text((lx + 34, y), text, font=small_f, fill=ink)
    cx_, cy_, R = S - 130, S - 150, 56
    d.ellipse([cx_ - R, cy_ - R, cx_ + R, cy_ + R], outline=ink, width=3)
    for k, lab in enumerate(("K", "D", "G", "B")):
        a = -math.pi / 2 + k * math.pi / 2
        tip = (cx_ + R * 0.95 * math.cos(a), cy_ + R * 0.95 * math.sin(a))
        l_ = (cx_ + 11 * math.cos(a + math.pi / 2), cy_ + 11 * math.sin(a + math.pi / 2))
        r_ = (cx_ + 11 * math.cos(a - math.pi / 2), cy_ + 11 * math.sin(a - math.pi / 2))
        d.polygon([tip, l_, (cx_, cy_), r_], fill=red if k == 0 else ink)
        d.text((cx_ + (R + 18) * math.cos(a) - 8, cy_ + (R + 18) * math.sin(a) - 12), lab, font=small_f, fill=ink)
    sb = 10 / (2 * E) * S
    d.line([(S - 330, S - 50), (S - 330 + sb, S - 50)], fill=ink, width=5)
    d.text((S - 330, S - 82), "10 m", font=small_f, fill=ink)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)
    print("[map]", path)


# ---------------------------------------------------------------------------
# preview lighting, export, main
# ---------------------------------------------------------------------------

def setup_world(rig, lamps):
    scene = bpy.context.scene
    world = bpy.data.worlds.new("SpringSky")
    scene.world = world
    if not world.node_tree:
        world.use_nodes = True
    nt = world.node_tree
    bg = next(n for n in nt.nodes if n.type == 'BACKGROUND')
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp_ = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp_.inputs["Fac"])
    nt.links.new(ramp_.outputs["Color"], bg.inputs["Color"])
    els = ramp_.color_ramp.elements
    els[0].position, els[1].position = 0.02, 0.35
    els[0].color = (*C.srgb_to_linear(hexc("#fbe7e4")), 1)
    els[1].color = (*C.srgb_to_linear(hexc("#86b9ea")), 1)
    bg.inputs["Strength"].default_value = 1.0
    sd = bpy.data.lights.new("Sun", 'SUN')
    sd.energy = 4.0
    sd.angle = math.radians(3)
    sd.color = tuple(C.srgb_to_linear(hexc("#fff1e0")))
    sun = bpy.data.objects.new("Sun", sd)
    sun.rotation_euler = (math.radians(42), 0, math.radians(35))
    rig.objects.link(sun)
    for k, p in enumerate(lamps):
        ld = bpy.data.lights.new(f"Lamp_{k}", 'POINT')
        ld.energy = 40
        ld.color = tuple(C.srgb_to_linear(hexc("#ffc27a")))
        ob = bpy.data.objects.new(f"Lamp_{k}", ld)
        ob.location = p
        rig.objects.link(ob)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = 'METRIC'
    make_materials()
    world_col = K.get_collection("BlossomValley")
    land_col = K.get_collection("Terrain", world_col)
    mark_col = K.get_collection("Landmarks", world_col)
    veg_col = K.get_collection("Vegetation", world_col)
    game_col = K.get_collection("Gameplay", world_col)
    prop_col = K.get_collection("TreasureProps", world_col)
    rig = K.get_collection("Preview_Rig")

    veg = plan_vegetation()
    PLAN["veg"] = veg
    print("[veg]", {k: len(v) for k, v in veg.items()})
    build_terrain(land_col, 110 if FAST else 230, veg)
    _, spts, swz, fall = build_water(land_col)
    build_waterfall_spot(mark_col, spts, swz, fall)
    build_pagoda(mark_col, [])
    build_stairs(mark_col)
    _, bmid, bax = build_bridge(mark_col)
    build_tea_house(mark_col)
    build_pond_details(mark_col)
    for k, (c, f) in enumerate(HOUSES):
        build_house(mark_col, c, f, k)
    for k, (c, f) in enumerate(STALLS):
        build_stall(mark_col, c, f, k)
    build_bell_tower(mark_col)
    build_zen_garden(mark_col)
    build_spring(mark_col)
    build_shrine(mark_col)
    build_paddy_hut(mark_col)
    build_gate(mark_col)
    build_bamboo_clearing(mark_col)
    build_vegetation(veg_col, veg)
    spawns, players = build_spawns(game_col)
    for label, p in spawns:
        print(f"[spawn] {label:28s} ({p[0]:6.1f}, {p[1]:6.1f}, {p[2]:5.2f})")
    props, meshes = build_props(prop_col, players)
    build_ground_cover(veg_col)

    lamps = [np.array([c[0] + sx * 2.5, y, gz((c[0] + sx * 2.5, y)) + 1.1])
             for c in (np.array([STAIR_X, 0]),) for sx in (-1, 1) for y in (6.2,)]
    setup_world(rig, lamps)
    hero = C.camera(rig, "Cam_Hero", (24, -50, 30), (0, 0, 1.5), lens=30)
    top = C.camera(rig, "Cam_Top", (0, -1, 90), (0, -1, 0), ortho=74)
    top.rotation_euler = (0, 0, 0)
    terrace = C.camera(rig, "Cam_Terrace", (-4, 2, 11), (6, 18, 6.5), lens=26)
    pond = C.camera(rig, "Cam_Pond", (6, -30, 7), (-5, -10, 0.5), lens=28)
    village = C.camera(rig, "Cam_Village", (21, -21, 13), (12, -4, 1.0), lens=28)

    bridge = (bmid, bax)
    draw_valley_map(os.path.join(RENDER_DIR, "valley_map.png"), [p for _, p in spawns], players, veg, bridge)
    draw_valley_map(os.path.join(RENDER_DIR, "valley_props_map.png"), [p for _, p in spawns], players, veg, bridge,
                    props=props)
    if not NO_RENDER:
        s = 12 if FAST else 64
        res = (960, 540) if FAST else (1920, 1080)
        C.render(hero, os.path.join(RENDER_DIR, "valley_hero.png"), res, s)
        C.render(terrace, os.path.join(RENDER_DIR, "valley_terrace.png"), res, s)
        C.render(pond, os.path.join(RENDER_DIR, "valley_pond.png"), res, s)
        C.render(village, os.path.join(RENDER_DIR, "valley_village.png"), res, s)
        C.render(top, os.path.join(RENDER_DIR, "valley_top.png"), (800, 800) if FAST else (1600, 1600), s)
        C.render_prop_catalog(rig, meshes, os.path.join(RENDER_DIR, "props_catalog.png"), s)
    bpy.context.scene.camera = hero

    C.setup_viewport()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "blossom_valley.blend"), compress=True,
                                relative_remap=True)
    objs = [o for o in world_col.all_objects if o.type in ('MESH', 'EMPTY')]
    C.export_glb(objs, os.path.join(EXPORT_DIR, "blossom_valley.glb"))


if __name__ == "__main__":
    main()
