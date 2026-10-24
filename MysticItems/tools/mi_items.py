"""The mystic items. Each class builds its geometry for lod 'hp' or 'lp'.

Build code must stay deterministic and lod-independent in everything except
tessellation density, so the two meshes describe the same surface.
"""
import math
import random

import bpy  # must precede bmesh when bpy is used as a pip module
import bmesh
from mathutils import Matrix, Vector

import mi_materials as M
from mi_core import (Canvas, TAU, apply_modifiers, catmull, circle_profile, deform, draw_runes,
                     fillet_polyline, handwriting, join, lathe, legacy_texture, mesh_object,
                     mod_bevel, mod_boolean, mod_displace, mod_remesh, mod_smooth, mod_subsurf,
                     offset_profile, rect_profile, resample, rune_width, runes_on_arc, set_material,
                     stadium_path, strands_along, sweep, uv_sphere)


def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def projector(name, loc, rot, half_w, half_h):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = 'PLAIN_AXES'
    e.location = loc
    e.rotation_euler = rot
    e.scale = (half_w, half_h, 1.0)
    bpy.context.scene.collection.objects.link(e)
    return e


def wobbly(cv, pts, w, rng, amp=0.0015, sub=6, closed=False):
    """Hand-cut line: subdivide and jitter so straight guides don't look ruled."""
    if closed:
        pts = pts + [pts[0]]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        for k in range(sub):
            t = k / sub
            out.append((a[0] + (b[0] - a[0]) * t + rng.uniform(-amp, amp),
                        a[1] + (b[1] - a[1]) * t + rng.uniform(-amp, amp)))
    out.append(pts[-1])
    cv.polyline(out, w)


def rounded_rect_pts(x0, y0, x1, y1, r, n=6):
    pts = []
    for cx, cy, a0 in ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def line_width(tokens, h, spacing=0.34):
    return sum((rune_width(t) + spacing) * h for t in tokens) - spacing * h


class Item:
    key = ''
    title = ''
    res = 2048
    extrusion = 0.002
    max_ray = 0.008
    ao_distance = 0.02
    sharp = 70.0          # edges sharper than this get split normals (and always a UV seam)
    island_angle = None   # smart-project angle that decides UV cuts; defaults to `sharp`
    view = {}
    emit_strength = 1.0

    def __init__(self):
        self.lod = 'hp'

    def S(self, hp, lp):
        return hp if self.lod == 'hp' else lp

    def setup(self):
        pass

    def build(self, lod):
        raise NotImplementedError

    def glass_material(self):
        return M.glass(f"M_{self.key}_Glass")


# ===========================================================================
# Rune stone
# ===========================================================================

class RuneStone(Item):
    """Hand-sized greywacke slab, inscribed in Elder Futhark: s-e-k : th-e-n : h-i-d-e.

    Biconical drilled hole with a leather thong, red ochre left in the grooves,
    lichen on the upward faces, chipped rim.
    """
    key = 'RuneStone'
    title = 'Rün Taşı'
    extrusion = 0.003
    max_ray = 0.012
    ao_distance = 0.03
    sharp = 180.0
    island_angle = 80.0
    view = dict(yaw=-28, pitch=18)

    W, T, H = 0.13, 0.044, 0.212
    HOLE = Vector((0.010, 0.0, 0.183))

    def setup(self):
        rng = random.Random(11)
        cv = Canvas(1024, 1536, seed=5)
        A = cv.aspect
        h, sw = 0.138, 0.012
        rows = [['s', 'e', 'k'], ['th', 'e', 'n'], ['h', 'i', 'd', 'e']]
        for row, base in zip(rows, (0.69, 0.43, 0.17)):
            w = line_width(row, h)
            draw_runes(cv, row, ((A - w) / 2 + rng.uniform(-0.01, 0.01), base), h, sw, rng)
        for y in (0.638, 0.377):
            wobbly(cv, [(0.075, y + rng.uniform(-0.004, 0.004)), (A - 0.075, y + rng.uniform(-0.004, 0.004))],
                   0.0075, rng, amp=0.002)
        wobbly(cv, rounded_rect_pts(0.045, 0.07, A - 0.045, 0.935, 0.07), 0.009, rng, amp=0.0018, sub=3,
               closed=True)
        front = cv.to_image('MSK_RuneStone_Front')

        cb = Canvas(1024, 1536, seed=9)
        c = (cb.aspect / 2, 0.55)
        cb.ring(c, 0.24, 0.011)
        cb.ring(c, 0.205, 0.006)
        rr = random.Random(4)
        draw_runes(cb, ['z'], (c[0] - 0.165, 0.39), 0.32, 0.016, rr)
        for k in range(12):
            a = TAU * k / 12 + 0.13
            cb.dot((c[0] + math.cos(a) * 0.29, c[1] + math.sin(a) * 0.29), 0.012)
        back = cb.to_image('MSK_RuneStone_Back')

        Wp, Hp = 0.102, 0.153
        pf = projector('PRJ_RuneStone_Front', (0, 0, 0.093), (math.pi / 2, 0, 0), Wp / 2, Hp / 2)
        pb = projector('PRJ_RuneStone_Back', (0, 0, 0.093), (math.pi / 2, 0, math.pi), Wp / 2, Hp / 2)
        self.m_stone = M.stone('M_RuneStone_Stone', masks=[(front, pf, (0, -1, 0)), (back, pb, (0, 1, 0))])
        self.m_thong = M.leather('M_RuneStone_Thong', base=(0.11, 0.055, 0.026), dark=(0.045, 0.022, 0.011),
                                 worn=(0.24, 0.14, 0.07), grain_scale=520, crease=0.4)

    def _stone(self):
        S = self.S
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(self.W, self.T, self.H + 0.01), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(0, 0, (self.H + 0.01) / 2 - 0.005), verts=bm.verts)
        ob = mesh_object('stone', bm)
        mod_bevel(ob, 0.016, 4, angle=30)
        mod_subsurf(ob, 2)
        ob = apply_modifiers(ob)
        W2, H = self.W / 2, self.H

        def shape(v):
            zn = max(0.0, v.z / H)
            x = v.x * (1 - 0.17 * zn ** 1.4) + 0.005 * math.sin(zn * 2.6 + 0.4)
            y = v.y * (1 - 0.26 * zn) * (1 + 0.16 * math.cos(v.x / W2 * 1.3))
            top = smoothstep(0.6, 1.0, zn)
            z = v.z - top * 0.03 * ((v.x / W2) - 0.3) ** 2
            return Vector((x, y, z))

        deform(ob, shape)

        rng = random.Random(23)
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bm.normal_update()
        rim = [(v.co.copy(), v.normal.copy()) for v in bm.verts
               if 0.25 < abs(v.normal.y) < 0.75 and v.co.z > 0.015]
        rim.sort(key=lambda cn: (round(cn[0].x, 4), round(cn[0].y, 4), round(cn[0].z, 4)))
        for _ in range(9):
            vco, n = rng.choice(rim)
            n = n + Vector((rng.uniform(-0.35, 0.35), rng.uniform(-0.2, 0.2), rng.uniform(-0.35, 0.35)))
            n.normalize()
            if abs(n.y) > 0.62:  # keep breaks on the rim instead of shaving a whole face flat
                n.y = math.copysign(0.62, n.y)
                n.normalize()
            depth = rng.uniform(0.0015, 0.0038)
            co = vco - n * depth
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=co,
                                   plane_no=n, clear_outer=True)
            bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, 0.0),
                               plane_no=(0, 0, -1), clear_outer=True)
        bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
        bm.to_mesh(ob.data)
        bm.free()

        # biconical hole: drilled from both faces until the cones met
        cut = lathe('hole_cut', [(0, -0.04), (0.0075, -0.04), (0.0068, -0.024), (0.0036, -0.001),
                                 (0.0036, 0.001), (0.0068, 0.024), (0.0075, 0.04), (0, 0.04)], 48)
        cut.rotation_euler = (math.pi / 2, 0, 0)
        cut.location = self.HOLE
        mod_boolean(ob, cut)
        ob = apply_modifiers(ob)
        bpy.data.objects.remove(cut)

        mod_remesh(ob, S(0.0007, 0.0018))
        if self.lod == 'hp':
            mod_smooth(ob, 0.5, 2)
        mod_displace(ob, legacy_texture('CLOUDS', noise_scale=0.05, noise_depth=2, noise_basis='ORIGINAL_PERLIN'), 0.006)
        mod_displace(ob, legacy_texture('CLOUDS', noise_scale=0.014, noise_depth=2, noise_basis='VORONOI_F2_F1',
                                        noise_type='HARD_NOISE'), -0.0005)
        mod_displace(ob, legacy_texture('CLOUDS', noise_scale=0.022, noise_depth=1, noise_basis='VORONOI_F1'),
                     0.0008)
        if self.lod == 'hp':
            mod_displace(ob, legacy_texture('CLOUDS', noise_scale=0.005, noise_depth=3, noise_basis='ORIGINAL_PERLIN'),
                         0.0007)
            mod_displace(ob, legacy_texture('CLOUDS', noise_scale=0.0018, noise_depth=1, noise_basis='VORONOI_F1'),
                         -0.00025)
        ob = apply_modifiers(ob)
        ob['lp_tris'] = 5000
        set_material(ob, self.m_stone)
        return ob

    def _thong(self):
        S = self.S
        c = self.HOLE
        cp = [
            (c.x, -0.004, c.z), (c.x, -0.019, c.z - 0.001), (c.x + 0.002, -0.025, c.z + 0.008),
            (c.x + 0.008, -0.020, c.z + 0.022), (c.x + 0.016, -0.006, c.z + 0.029),
            (c.x + 0.024, 0.0, c.z + 0.028),
            (c.x + 0.040, 0.003, c.z + 0.022), (0.060, 0.000, 0.168), (0.066, -0.004, 0.120),
            (0.072, -0.010, 0.070), (0.080, -0.026, 0.026), (0.090, -0.050, 0.0045),
            (0.112, -0.071, 0.0028), (0.140, -0.066, 0.0028), (0.150, -0.040, 0.0028),
            (0.134, -0.020, 0.0045), (0.100, -0.006, 0.020), (0.084, 0.010, 0.070),
            (0.074, 0.014, 0.125), (0.064, 0.010, 0.170), (c.x + 0.042, 0.006, c.z + 0.025),
            (c.x + 0.026, 0.005, c.z + 0.030), (c.x + 0.015, 0.008, c.z + 0.029),
            (c.x + 0.007, 0.020, c.z + 0.021), (c.x + 0.002, 0.025, c.z + 0.008),
            (c.x, 0.019, c.z - 0.001), (c.x, 0.004, c.z),
        ]
        path = resample(catmull(cp, 8, closed=True), S(0.0007, 0.006))
        prof = rect_profile(0.0048, 0.0018, 0.0007, S(3, 1))
        thong = sweep('thong', path[:-1], prof, closed=True, up=(0, 0, 1))
        # binding wrapped round both strands where they meet above the stone
        a = Vector((c.x + 0.019, 0.003, c.z + 0.0295))
        b = Vector((c.x + 0.034, 0.004, c.z + 0.027))
        ax = (b - a).normalized()
        n1 = ax.cross(Vector((0, 1, 0))).normalized()
        n2 = ax.cross(n1)
        turns, steps = 4.5, S(160, 40)
        helix = []
        for k in range(steps + 1):
            t = k / steps
            ang = TAU * turns * t
            r = 0.0042 + 0.0004 * math.sin(t * math.pi)
            helix.append(a.lerp(b, t) + n1 * math.cos(ang) * r + n2 * math.sin(ang) * r * 0.8)
        wrap = sweep('wrap', helix, rect_profile(0.0026, 0.0011, 0.0004, S(3, 1)))
        for ob in (thong, wrap):
            set_material(ob, self.m_thong)
        return [thong, wrap]

    def build(self, lod):
        self.lod = lod
        return {'body': [self._stone()] + self._thong(), 'glass': []}



# ===========================================================================
# Amulet
# ===========================================================================

def dome(name, loc, r, h, seg, normal=(0, 0, 1)):
    """Rivet / granule head: a squashed dome sitting on `loc` facing `normal`."""
    prof = [(0, -h * 0.4), (r, -h * 0.4)] + [(r * math.cos(t), h * math.sin(t))
                                             for t in [math.pi / 2 * k / seg for k in range(seg + 1)]]
    prof[-1] = (0, h)
    ob = lathe(name, prof, max(8, seg * 4))
    n = Vector(normal).normalized()
    ob.rotation_euler = Vector((0, 0, 1)).rotation_difference(n).to_euler()
    ob.location = loc
    return apply_transform(ob)


def apply_transform(ob):
    # build the matrix from loc/rot/scale: matrix_world is stale until the depsgraph updates
    m = Matrix.LocRotScale(ob.location, ob.rotation_euler.to_quaternion(), ob.scale)
    ob.data.transform(m)
    ob.location = (0, 0, 0)
    ob.rotation_euler = (0, 0, 0)
    ob.scale = (1, 1, 1)
    return ob


class Amulet(Item):
    """Cast bronze bracteate-style pendant on a fine trace chain, lying where it was dropped.

    Garnet cabochon in a rolled bezel, a ring of granulation, twisted-wire border and
    the charm words alu / laþu engraved around the field.
    """
    key = 'Amulet'
    title = 'Muska'
    res = 1024
    extrusion = 0.0004
    max_ray = 0.0016
    ao_distance = 0.004
    view = dict(yaw=-20, pitch=52)
    emit_strength = 0.05
    R = 0.022
    BAIL = Vector((0, 0.022 + 0.0024, 0.0021))

    def setup(self):
        rng = random.Random(5)
        cv = Canvas(1024, 1024, seed=2)
        c = (0.5, 0.5)
        span = 2 * 0.026  # projector covers +-26 mm
        k = 1.0 / span
        words = (['a', 'l', 'u', '|', 'l', 'a', 'th', 'u', '|'] * 3)
        runes_on_arc(cv, words, c, 0.0128 * k, 0.0034 * k, 0.0004 * k, rng, math.pi / 2 + 0.05)
        cv.ring(c, 0.0124 * k, 0.00022 * k)
        cv.ring(c, 0.0171 * k, 0.00022 * k)
        for i in range(72):
            a = TAU * i / 72
            r0, r1 = 0.0176 * k, 0.0186 * k
            cv.stroke((c[0] + math.cos(a) * r0, c[1] + math.sin(a) * r0),
                      (c[0] + math.cos(a + 0.03) * r1, c[1] + math.sin(a + 0.03) * r1), 0.0003 * k)
        mask = cv.to_image('MSK_Amulet_Field')
        pr = projector('PRJ_Amulet', (0, 0, 0), (0, 0, 0), 0.026, 0.026)
        self.m_metal = M.bronze('M_Amulet_Bronze', clean=(0.62, 0.38, 0.14), masks=[(mask, pr, (0, 0, 1))],
                                mask_depth=0.00018, patina_amount=0.7)
        self.m_gem = M.gem('M_Amulet_Garnet')

    def _pendant(self):
        S = self.S
        R = self.R
        prof = fillet_polyline([(0, 0, 0), (R - 0.0012, 0, 0.0006), (R + 0.0002, 0.0012, 0.0008),
                                (R + 0.0001, 0.0033, 0.0007), (R - 0.0018, 0.0039, 0.0006),
                                (R - 0.0031, 0.0026, 0.0005), (0.0105, 0.0023, 0.0), (0, 0.0023, 0)], S(6, 2))
        parts = [lathe('plate', prof, S(160, 40))]
        bez = fillet_polyline([(0.0086, 0.0018, 0), (0.0104, 0.0018, 0.0002), (0.0105, 0.0044, 0.0005),
                               (0.0096, 0.0056, 0.0004), (0.0087, 0.0051, 0.0002), (0.0086, 0.0018, 0)], S(4, 1))
        parts.append(lathe('bezel', bez[:-1], S(96, 32), cyclic=True))
        gem_prof = [(0, 0.0021), (0.0089, 0.0021), (0.009, 0.0031)]
        for k in range(1, S(12, 4) + 1):
            t = k / S(12, 4) * math.pi / 2
            gem_prof.append((0.009 * math.cos(t), 0.0031 + 0.0040 * math.sin(t) ** 0.9))
        gem_prof[-1] = (0, gem_prof[-1][1])
        gem = lathe('gem', gem_prof, S(64, 20))
        deform(gem, lambda v: Vector((v.x, v.y * 0.96, v.z)))
        set_material(gem, self.m_gem)
        for i in range(12):
            a = TAU * i / 12 + 0.12
            parts.append(dome(f'gran{i}', (math.cos(a) * 0.0118, math.sin(a) * 0.0118, 0.0021), 0.00072, 0.00068,
                              S(5, 2)))
        # twisted wire border: two plies wound round the outer edge
        Rr, zr = R + 0.0006, 0.0021
        for ply in range(2):
            pts = []
            n = S(720, 160)
            for i in range(n):
                th = TAU * i / n
                ph = th * 95 + math.pi * ply
                rr = Rr + 0.00042 * math.cos(ph)
                pts.append((rr * math.cos(th), rr * math.sin(th), zr + 0.00042 * math.sin(ph)))
            parts.append(sweep(f'rope{ply}', pts, circle_profile(0.00042, S(8, 4)), closed=True))
        # bail: a ribbed tube rolled from strip, lying in the plane of the pendant
        prof = [(0.0012, -0.003), (0.0019, -0.003)]
        n = S(36, 6)
        for k in range(n + 1):
            x = -0.003 + 0.006 * k / n
            prof.append((0.0019 + 0.00022 * max(0.0, math.cos(x / 0.006 * 3 * TAU)) ** 2, x))
        prof += [(0.0019, 0.003), (0.0012, 0.003)]
        bail = lathe('bail', prof, S(32, 10), cyclic=True)
        bail.rotation_euler = (0, math.pi / 2, 0)
        bail.location = self.BAIL
        parts.append(apply_transform(bail))
        return parts, gem

    def _chain(self):
        S = self.S
        Rj, rj = 0.0029, 0.0005
        tilt = math.asin((self.BAIL.z - rj) / (2 * Rj))
        jc = self.BAIL + Vector((0, Rj * math.cos(tilt), -Rj * math.sin(tilt)))
        jr = [jc + Vector((Rj * math.cos(a), Rj * math.sin(a) * math.cos(tilt), -Rj * math.sin(a) * math.sin(tilt)))
              for a in [TAU * i / S(48, 12) for i in range(S(48, 12))]]
        parts = [sweep('jumpring', jr, circle_profile(rj, S(10, 4)), closed=True)]
        start = jc + Vector((0, Rj * math.cos(tilt), -Rj * math.sin(tilt)))
        cp = [(start.x, start.y), (0.003, 0.040), (0.012, 0.050), (0.026, 0.054), (0.040, 0.047),
              (0.048, 0.034), (0.046, 0.018), (0.038, 0.004), (0.033, -0.012), (0.034, -0.028),
              (0.043, -0.040), (0.058, -0.044), (0.070, -0.036), (0.074, -0.022)]
        path = resample(catmull([(x, y, 0) for x, y in cp], 10), 0.0054)
        rng = random.Random(8)
        A, B, r = 0.0033, 0.0019, 0.00052
        back = 0.0054 - (A - r - rj)  # first link must hook through the jump ring
        link_path = stadium_path(2 * A, 2 * B, S(40, 10))
        for i in range(1, len(path) - 1):
            t = (path[i + 1] - path[i - 1]).normalized()
            p = path[i] - t * back
            side = Vector((0, 0, 1)).cross(t).normalized()
            if i % 2:  # propped up by its neighbours, falling to one side
                lean = math.radians(rng.uniform(20, 40)) * (1 if rng.random() < 0.5 else -1)
                wv = Vector((0, 0, 1)) * math.cos(lean) + side * math.sin(lean)
            else:
                lean = math.radians(rng.uniform(-7, 7))
                wv = side * math.cos(lean) + Vector((0, 0, 1)) * math.sin(lean)
            wv = (wv - t * wv.dot(t)).normalized()
            z = r + B * abs(wv.z) + 0.00002
            pts = [Vector((p.x, p.y, z)) + t * q[0] + wv * q[1] for q in link_path]
            parts.append(sweep(f'link{i}', pts, circle_profile(r, S(10, 4)), closed=True))
        return parts

    def build(self, lod):
        self.lod = lod
        parts, gem = self._pendant()
        parts += self._chain()
        for p in parts:
            set_material(p, self.m_metal)
        return {'body': parts + [gem], 'glass': []}


# ===========================================================================
# Potion
# ===========================================================================

def _profile_r(prof, z, side='outer'):
    """Linear lookup of r at height z on an (r, z) polyline (first hit)."""
    best = None
    for (r0, z0), (r1, z1) in zip(prof[:-1], prof[1:]):
        if (z0 - z) * (z1 - z) <= 0 and z0 != z1:
            t = (z - z0) / (z1 - z0)
            r = r0 + (r1 - r0) * t
            if best is None or r > best:
                best = r
    return best


class Potion(Item):
    """Mould-free blown flask, cork sealed under red wax, twine collar and a handwritten tag."""
    key = 'Potion'
    title = 'İksir Şişesi'
    extrusion = 0.0012
    max_ray = 0.005
    ao_distance = 0.012
    view = dict(yaw=-30, pitch=16)
    emit_strength = 0.1
    FILL = 0.054
    TAG_Z = 0.051
    TAG_H = 0.0375
    HOLE_Z = 0.051 + 0.0375 / 2 - 0.0042

    OUTER = [(0.0, 0.0062), (0.012, 0.0040), (0.025, 0.0009), (0.0305, 0.0), (0.0355, 0.0022),
             (0.0410, 0.0110), (0.0442, 0.0260), (0.0440, 0.0400), (0.0400, 0.0540), (0.0320, 0.0660),
             (0.0215, 0.0760), (0.0140, 0.0855), (0.0118, 0.0950), (0.0112, 0.1100), (0.0114, 0.1220),
             (0.0128, 0.1248), (0.0146, 0.1262), (0.0149, 0.1290), (0.0136, 0.1310), (0.0098, 0.1310)]
    INNER = [(0.0090, 0.1296), (0.0087, 0.1220), (0.0086, 0.1100), (0.0094, 0.0950), (0.0115, 0.0860),
             (0.0195, 0.0755), (0.0300, 0.0650), (0.0378, 0.0535), (0.0420, 0.0400), (0.0421, 0.0260),
             (0.0390, 0.0125), (0.0335, 0.0050), (0.0250, 0.0038), (0.0120, 0.0068), (0.0, 0.0090)]

    def setup(self):
        self.m_liquid = M.liquid('M_Potion_Liquid', z0=0.006, z1=self.FILL)
        self.m_cork = M.cork('M_Potion_Cork')
        sc = Canvas(512, 512, seed=6)
        sc.ring((0.5, 0.5), 0.36, 0.035)
        draw_runes(sc, ['o'], (0.5 - 0.14, 0.24), 0.52, 0.04, random.Random(2))
        for k in range(16):
            a = TAU * k / 16
            sc.dot((0.5 + math.cos(a) * 0.43, 0.5 + math.sin(a) * 0.43), 0.025)
        seal = sc.to_image('MSK_Potion_Seal')
        pr = projector('PRJ_Potion_Seal', (0.0012, 0.0004, 0.14), (0, 0, 0.4), 0.0072, 0.0072)
        self.m_wax = M.wax('M_Potion_Wax', masks=[(seal, pr, (0, 0, 1))])
        self.m_twine = M.twine('M_Potion_Twine')
        rng = random.Random(3)
        cv = Canvas(768, 1152, seed=4)
        handwriting(cv, 0.08, 0.66, cv.aspect - 0.16, 0.085, 4, rng, pen=0.0032)
        cv.ring((cv.aspect / 2, 0.2), 0.07, 0.006)
        cv.stroke((cv.aspect / 2 - 0.07, 0.2), (cv.aspect / 2 + 0.07, 0.2), 0.005)
        cv.stroke((cv.aspect / 2, 0.13), (cv.aspect / 2, 0.27), 0.005)
        cv.ring((cv.aspect / 2, 0.888), 0.05, 0.007)  # reinforcement ring around the hole
        ink = cv.to_image('MSK_Potion_Tag')
        self.tag_proj = projector('PRJ_Potion_Tag', (0, -0.0457, 0.051), (math.pi / 2, 0, 0), 0.0125, 0.0188)
        self.m_tag = M.parchment('M_Potion_Tag', masks=[(ink, self.tag_proj, None)])

    def glass_material(self):
        return M.glass('M_Potion_Glass', tint=(0.80, 0.90, 0.84), rough=0.03)

    def _hand_blown(self, ob):
        def fn(v):
            lean = 0.0016 * smoothstep(0.07, 0.13, v.z)
            ang = math.atan2(v.y, v.x)
            wob = 1 + 0.010 * math.sin(ang * 2 + v.z * 40) + 0.006 * math.sin(ang * 3 + 1.3)
            return Vector((v.x * wob + lean, v.y * wob + lean * 0.4, v.z))
        return deform(ob, fn)

    def _glass(self):
        S = self.S
        o = [(p.x, p.y) for p in catmull(self.OUTER, S(6, 2))]
        i = [(p.x, p.y) for p in catmull(self.INNER, S(6, 2))]
        ob = lathe('glass', o + i, S(96, 32))
        return self._hand_blown(ob)

    def _liquid(self):
        S = self.S
        inner = [(p.x, p.y) for p in catmull(self.INNER, S(6, 2))][::-1]
        prof = [(0, inner[0][1] + 0.0004)]
        for r, z in inner[1:]:
            if z >= self.FILL - 0.0012:
                break
            prof.append((max(0.0, r - 0.0004), z + 0.0004 if r < 0.03 and z < 0.01 else z))
        rw = _profile_r(self.INNER, self.FILL) - 0.0004
        for k in range(S(8, 3) + 1):  # meniscus climbing the wall
            t = k / S(8, 3)
            prof.append((rw - 0.0025 * t, self.FILL + 0.0011 * (1 - t) ** 2))
        prof.append((0, self.FILL))
        ob = lathe('liquid', prof, S(96, 32))
        self._hand_blown(ob)
        set_material(ob, self.m_liquid)
        return ob

    def _cork(self):
        S = self.S
        prof = fillet_polyline([(0, 0.1150, 0), (0.0080, 0.1150, 0.0008), (0.0087, 0.1240, 0),
                                (0.0093, 0.1325, 0), (0.0096, 0.1372, 0.0012), (0.0060, 0.1392, 0.0),
                                (0, 0.1396, 0)], S(5, 2))
        ob = lathe('cork', prof, S(64, 24))
        self._hand_blown(ob)
        set_material(ob, self.m_cork)
        return ob

    def _wax(self):
        S = self.S
        parts = [lathe('cap', fillet_polyline([(0, 0.1365, 0), (0.0101, 0.1365, 0), (0.0108, 0.1349, 0.0005),
                                               (0.0107, 0.1386, 0.0012), (0.0088, 0.1409, 0.0010),
                                               (0, 0.1413, 0)], 4), 48)]
        # centre line of a run of wax over cork, lip roll and down the neck
        surf = [(0.0106, 0.1352), (0.0099, 0.1335), (0.0099, 0.1318), (0.0105, 0.1316), (0.0125, 0.1316),
                (0.0142, 0.1313), (0.0153, 0.1296), (0.0154, 0.1276), (0.0150, 0.1259), (0.0134, 0.1250),
                (0.0123, 0.1231), (0.0119, 0.1200), (0.0117, 0.1100)]
        drips = [(0.3, 0.1235, 0.0013), (1.4, 0.1268, 0.0011), (2.5, 0.1125, 0.0015), (3.7, 0.1278, 0.0010),
                 (4.6, 0.1205, 0.0012), (5.5, 0.1300, 0.0009)]
        for ang, zend, w in drips:
            pts = []
            for (r0, z0), (r1, z1) in zip(surf[:-1], surf[1:]):
                pts.append(Vector((r0 * math.cos(ang), r0 * math.sin(ang), z0)))
                if z1 <= zend:
                    tt = (zend - z0) / (z1 - z0) if z1 != z0 else 1.0
                    rr = r0 + (r1 - r0) * tt
                    pts.append(Vector((rr * math.cos(ang), rr * math.sin(ang), zend)))
                    break
            pts = resample(catmull(pts, 6), 0.0003)
            bulb = lambda t: 0.8 + 0.5 * smoothstep(0.7, 0.95, t) - 0.75 * smoothstep(0.95, 1.0, t)
            parts.append(sweep(f'drip{ang:.1f}', pts, circle_profile(w, 10, 1.0, 0.7), scale_fn=bulb))
        wax = join(parts, 'wax')
        mod_remesh(wax, S(0.00016, 0.0006))
        mod_smooth(wax, 0.6, S(6, 2))
        mod_displace(wax, legacy_texture('CLOUDS', noise_scale=0.004, noise_depth=2), 0.00025)
        wax = apply_modifiers(wax)
        self._hand_blown(wax)
        wax['lp_tris'] = 1500
        set_material(wax, self.m_wax)
        return wax

    def _twine(self):
        S = self.S
        parts = []
        helix = []
        turns, z0, z1 = 5.2, 0.1035, 0.1135
        steps = S(900, 160)
        for k in range(steps + 1):
            t = k / steps
            a = -math.pi / 2 + TAU * turns * t
            z = z0 + (z1 - z0) * t + 0.0002 * math.sin(a * 3)
            r = 0.0112 + 0.0011
            helix.append(Vector((r * math.cos(a), r * math.sin(a), z)))
        parts += strands_along('collar', helix, 0.00052, 2, 1100, S(6, 3))
        knot_c = Vector((0.0005, -0.0128, 0.1085))
        knot = []
        for k in range(S(160, 40) + 1):
            t = k / S(160, 40)
            a = TAU * 2.4 * t
            knot.append(knot_c + Vector((0.0016 * math.cos(a), -0.0009 - 0.0006 * math.sin(a * 0.5),
                                         0.0016 * math.sin(a))))
        parts += strands_along('knot', knot, 0.0005, 2, 900, S(6, 3))
        hole = Vector((0.0, -0.0457, self.HOLE_Z))
        tail = [knot_c + Vector((0.001, -0.0012, -0.001)), Vector((0.0015, -0.0128, 0.098)),
                Vector((0.0015, -0.0146, 0.0868)), Vector((0.0015, -0.0198, 0.0805)),
                Vector((0.0012, -0.0232, 0.0768)), Vector((0.0008, -0.0272, 0.0735)),
                Vector((0.0004, -0.0380, 0.0690)), Vector((0.0, -0.0452, 0.0672)),
                hole + Vector((0.0, -0.0014, 0.0004))]
        parts += strands_along('tail', resample(catmull(tail, 8), 0.0005), 0.00052, 2, 900, S(6, 3))
        loop = [hole + Vector((0.0, -0.0014, 0.0004)), hole + Vector((0.0, 0.0, -0.0003)),
                hole + Vector((0.0002, 0.0014, 0.0003)), hole + Vector((0.0004, 0.0015, 0.0036)),
                hole + Vector((0.0006, -0.0003, 0.0055)), hole + Vector((0.0008, -0.0018, 0.0035)),
                hole + Vector((0.0006, -0.0019, 0.0012))]
        parts += strands_along('loop', resample(catmull(loop, 8), 0.0004), 0.0005, 2, 900, S(6, 3))
        short = [knot_c + Vector((-0.001, -0.0012, -0.001)), Vector((-0.002, -0.0150, 0.102)),
                 Vector((-0.004, -0.0160, 0.094)), Vector((-0.0045, -0.0175, 0.0885))]
        fray = lambda t: 1.0 + 0.6 * smoothstep(0.8, 1.0, t)
        parts += strands_along('short', resample(catmull(short, 8), 0.0005), 0.0005, 2, 900, S(6, 3),
                               scale_fn=fray)
        for p in parts:
            set_material(p, self.m_twine)
            self._hand_blown(p)
        return parts

    def _tag(self):
        S = self.S
        w, h, th = 0.025, 0.0375, 0.00035
        nx, ny = S(50, 6), S(75, 9)
        bm = bmesh.new()
        grid = [[bm.verts.new(((-w / 2 + w * i / nx), 0.0, (-h / 2 + h * j / ny))) for j in range(ny + 1)]
                for i in range(nx + 1)]
        for i in range(nx):
            for j in range(ny):
                bm.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]))
        ob = mesh_object('tag', bm)
        m = ob.modifiers.new('Solid', 'SOLIDIFY')
        m.thickness = th
        m.offset = 0
        hole = lathe('tag_hole', [(0, -0.01), (0.0015, -0.01), (0.0015, 0.01), (0, 0.01)], 24)
        hole.rotation_euler = (math.pi / 2, 0, 0)
        hole.location = (0, 0, h / 2 - 0.0042)
        mod_boolean(ob, hole)
        ob = apply_modifiers(ob)
        bpy.data.objects.remove(hole)
        mod_bevel(ob, 0.00012, S(2, 1), angle=40)
        ob = apply_modifiers(ob)
        Rb = 0.0457

        def wrap(v):
            # local y runs towards the bottle so the mapping keeps face winding (no mirror)
            ang = -math.pi / 2 + v.x / Rb
            r = Rb - v.y - 0.0004 * math.sin(v.z / h * math.pi)
            curl = 0.0012 * smoothstep(h * 0.3, h * 0.5, v.z)
            return Vector((r * math.cos(ang) * 1.0, r * math.sin(ang) - curl, 0.051 + v.z))

        deform(ob, wrap)
        self.tag_proj.location = (0, -Rb - 0.001, 0.051)
        set_material(ob, self.m_tag)
        return ob

    def build(self, lod):
        self.lod = lod
        body = [self._liquid(), self._cork(), self._wax(), self._tag()] + self._twine()
        return {'body': body, 'glass': [self._glass()]}



# ===========================================================================
# Grimoire
# ===========================================================================

def extrude_outline(name, outline, z0, z1):
    """Solid prism from a CCW 2D outline."""
    bm = bmesh.new()
    bot = [bm.verts.new((x, y, z0)) for x, y in outline]
    top = [bm.verts.new((x, y, z1)) for x, y in outline]
    n = len(outline)
    bm.faces.new(bot[::-1])
    bm.faces.new(top)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((bot[i], bot[j], top[j], top[i]))
    return mesh_object(name, bm, smooth=False)


def clip_line(a, b, rect):
    """Liang-Barsky clip of segment ab to rect (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 > t1:
        return None
    return (a[0] + dx * t0, a[1] + dy * t0), (a[0] + dx * t1, a[1] + dy * t1)


class Grimoire(Item):
    """Oak-board grimoire in blind-tooled calfskin: raised bands, brass corners, a central boss
    set with a moonstone, one surviving clasp strap and a faded silk bookmark."""
    key = 'Grimoire'
    title = 'Kadim Kitap'
    extrusion = 0.0016
    max_ray = 0.006
    ao_distance = 0.02
    sharp = 60.0
    view = dict(yaw=-34, pitch=32)
    emit_strength = 0.25
    X0, X1, Y, H, TB = -0.083, 0.0875, 0.1205, 0.058, 0.0045
    LIFT = 0.0008  # the brass corners keep the leather just off the table

    def _bow(self, x, y):
        u = (x - self.X0) / (self.X1 - self.X0)
        return 0.0011 * math.sin(math.pi * min(1, max(0, u))) * (1 - 0.35 * (y / self.Y) ** 2)

    def setup(self):
        rng = random.Random(21)
        cv = Canvas(1024, 1448, seed=12)
        A = cv.aspect
        for inset, w in ((0.028, 0.0042), (0.040, 0.0026)):
            wobbly(cv, rounded_rect_pts(inset, inset, A - inset, 1 - inset, 0.01, 3), w, rng, amp=0.0006,
                   sub=4, closed=True)
        panel = (0.1, 0.1, A - 0.1, 0.9)
        wobbly(cv, [(panel[0], panel[1]), (panel[2], panel[1]), (panel[2], panel[3]), (panel[0], panel[3])],
               0.0042, rng, amp=0.0006, sub=8, closed=True)
        step = 0.105
        for k in range(-12, 14):
            c0 = k * step
            for sgn in (1, -1):
                a = (panel[0] - 0.5, sgn * (panel[0] - 0.5) + c0 + 0.5)
                b = (panel[2] + 0.5, sgn * (panel[2] + 0.5) + c0 + 0.5)
                seg = clip_line(a, b, panel)
                if seg and math.dist(*seg) > 0.02:
                    cv.stroke(seg[0], seg[1], 0.0028, rough=0.15)
        for i in range(-12, 14):
            for j in range(-12, 14):
                x = 0.5 * A + (i - j) * step / 2
                y = 0.5 + (i + j) * step / 2
                if panel[0] + 0.02 < x < panel[2] - 0.02 and panel[1] + 0.02 < y < panel[3] - 0.02:
                    cv.ring((x, y), 0.0105, 0.0022)
                    cv.dot((x, y), 0.004)
        for cx, cy in ((0.07, 0.07), (A - 0.07, 0.07), (0.07, 0.93), (A - 0.07, 0.93)):
            for k in range(6):
                a = TAU * k / 6
                cv.dot((cx + 0.012 * math.cos(a), cy + 0.012 * math.sin(a)), 0.006)
        tool = cv.to_image('MSK_Grimoire_Tooling')
        cx = (self.X0 + self.X1) / 2
        pr = projector('PRJ_Grimoire_Cover', (cx, 0, 0), (0, 0, 0), (self.X1 - self.X0) / 2, self.Y)
        self.m_leather = M.leather('M_Grimoire_Leather', masks=[(tool, pr, (0, 0, 1))], mask_depth=0.00045)
        self.m_strap = M.leather('M_Grimoire_Strap', base=(0.05, 0.024, 0.013), dark=(0.022, 0.011, 0.006),
                                 worn=(0.16, 0.09, 0.045), grain_scale=450)
        self.m_pages = M.page_edges('M_Grimoire_Pages')
        self.m_ribbon = M.silk('M_Grimoire_Ribbon')
        bc = Canvas(1024, 1024, seed=13)
        c = (0.5, 0.5)
        k = 1 / 0.054
        for rr in (0.0132, 0.0205):
            bc.ring(c, rr * k, 0.00035 * k)
        r1, r2 = 0.0142 * k, 0.0196 * k
        for i in range(8):
            a = TAU * i / 8
            b = a + TAU / 16
            bc.polyline([(c[0] + math.cos(a) * r1, c[1] + math.sin(a) * r1),
                         (c[0] + math.cos(b) * r2, c[1] + math.sin(b) * r2),
                         (c[0] + math.cos(a + TAU / 8) * r1, c[1] + math.sin(a + TAU / 8) * r1)],
                        0.0004 * k, rough=0.1)
            bc.dot((c[0] + math.cos(b) * 0.0170 * k, c[1] + math.sin(b) * 0.0170 * k), 0.0009 * k)
        boss_mask = bc.to_image('MSK_Grimoire_Boss')
        bp = projector('PRJ_Grimoire_Boss', (cx, 0, 0), (0, 0, 0), 0.027, 0.027)
        self.m_brass = M.brass('M_Grimoire_Brass', masks=[(boss_mask, bp, (0, 0, 1))], mask_mode='emboss',
                               mask_depth=0.0005)
        self.m_stone = M.gem('M_Grimoire_Moonstone', color=(0.32, 0.34, 0.36), glow=(0.16, 0.24, 0.42))

    def _cover(self, front):
        S = self.S
        X0, X1, Y, TB, H = self.X0, self.X1, self.Y, self.TB, self.H
        z0, z1 = (H - TB, H) if front else (0.0, TB)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(X1 - X0, 2 * Y, z1 - z0), verts=bm.verts)
        bmesh.ops.translate(bm, vec=((X0 + X1) / 2, 0, (z0 + z1) / 2), verts=bm.verts)
        edges = [e for e in bm.edges if all(v.co.x > X1 - 1e-6 for v in e.verts)
                 and abs(e.verts[0].co.z - e.verts[1].co.z) > 1e-6]
        bmesh.ops.bevel(bm, geom=edges, offset=0.007, segments=S(10, 4), affect='EDGES', profile=0.5)
        ob = mesh_object('cover', bm)
        mod_bevel(ob, 0.0014, S(4, 1), angle=40)
        mod_subsurf(ob, S(3, 2), simple=True)
        ob = apply_modifiers(ob)

        def fn(v):
            d = self._bow(v.x, v.y)
            return Vector((v.x, v.y, v.z + (d if front else -0.25 * d)))

        deform(ob, fn)
        set_material(ob, self.m_leather)
        return ob

    def _spine(self):
        S = self.S
        H, X0 = self.H, self.X0
        a, b = 0.0115, H / 2
        n = S(28, 8)
        # solid half-ellipse; the rounded back of the text block sits inside it
        prof = [(0.004, -b + 0.0006)]
        for k in range(n + 1):
            ph = -math.pi / 2 + math.pi * k / n
            prof.append((-a * math.cos(ph), b * math.sin(ph)))
        prof[1] = (0.0, -b)
        prof[-1] = (0.0, b)
        prof.append((0.004, b - 0.0006))
        bands = [-0.084, -0.042, 0.0, 0.042, 0.084]
        L = 2 * self.Y

        def sc(t):
            y = -self.Y + L * t
            bump = sum(math.exp(-((y - yb) / 0.0032) ** 2) for yb in bands)
            ends = smoothstep(0.0, 0.006, self.Y - abs(y))
            return (1 + 0.16 * bump - 0.05 * (1 - ends), 1 + 0.012 * bump)

        path = [Vector((X0, -self.Y + L * k / S(320, 48), H / 2)) for k in range(S(320, 48) + 1)]
        ob = sweep('spine', path, prof, up=(0, 0, 1), scale_fn=sc)
        set_material(ob, self.m_leather)
        return ob

    def _block(self):
        S = self.S
        X0, H, TB = self.X0, self.H, self.TB
        xb0, xb1, yb = X0 - 0.004, 0.0835, 0.1165
        zb0, zb1 = TB + 0.0002, H - TB - 0.0002
        ob = rounded_box_mesh('pages', (xb1 - xb0, 2 * yb, zb1 - zb0), ((xb0 + xb1) / 2, 0, (zb0 + zb1) / 2))
        mod_bevel(ob, 0.0008, S(3, 1), angle=40)
        mod_subsurf(ob, S(5, 2), simple=True)
        ob = apply_modifiers(ob)

        def fn(v):
            zn = (v.z - zb0) / (zb1 - zb0)
            u = (v.x - xb0) / (xb1 - xb0)
            x = v.x - 0.0028 * math.sin(math.pi * zn) * u ** 3
            from mathutils import noise
            x += 0.00025 * noise.noise(Vector((v.y * 120, v.z * 900, 0.3))) * u ** 6
            y = v.y + 0.0002 * noise.noise(Vector((v.x * 200, v.z * 900, 1.7))) * math.copysign(1, v.y)
            return Vector((x, y, v.z))

        deform(ob, fn)
        set_material(ob, self.m_pages)
        return ob

    def _corner(self, sx, sy, front):
        """Folded brass corner. sx/sy are +-1 picking the corner; local frame has the corner at 0."""
        S = self.S
        Lc, o, rc = 0.031, 0.0008, 0.007
        pts = [(o, -Lc)]
        for k in range(S(10, 4) + 1):
            a = math.radians(90 * k / S(10, 4))
            pts.append((-rc + (rc + o) * math.cos(a), -rc + (rc + o) * math.sin(a)))
        pts.append((-Lc, o))
        n = S(36, 9)
        for k in range(1, n):
            s = k / n
            x = -Lc + (Lc + o) * s
            y = o + (-Lc - o) * s
            lobe = 0.0042 * math.sin(3 * math.pi * s) ** 2
            pts.append((x - lobe * 0.7071, y - lobe * 0.7071))
        X1, Y = self.X1, self.Y
        world = [(X1 + x if sx > 0 else self.X0 - x, sy * (Y + y)) for x, y in pts]
        if (sx > 0) != (sy > 0):
            world = world[::-1]
        z0, z1 = (self.H - self.TB - o, self.H + o + 0.0002) if front else (-o, self.TB + o)
        ob = extrude_outline('corner', world, z0, z1)
        mod_bevel(ob, 0.0004, S(3, 1), angle=40)
        ob = apply_modifiers(ob)
        if front:
            deform(ob, lambda v: Vector((v.x, v.y, v.z + self._bow(v.x, v.y))))
        parts = [ob]
        if front:
            for lx, ly in ((-0.008, -0.019), (-0.019, -0.008)):
                wx, wy = X1 + lx, sy * (Y + ly)
                parts.append(dome('rivet', (wx, wy, z1 + self._bow(wx, wy) - 0.0001), 0.0011, 0.0007, S(5, 2)))
        for p in parts:
            set_material(p, self.m_brass)
        return parts

    def _boss(self):
        S = self.S
        cx = (self.X0 + self.X1) / 2
        z = self.H + self._bow(cx, 0) - 0.0003
        prof = fillet_polyline([(0, 0, 0), (0.0255, 0, 0.0), (0.0262, 0.0009, 0.0004), (0.0252, 0.0019, 0.0004),
                                (0.0234, 0.0027, 0.0005), (0.0216, 0.0019, 0.0003), (0.0125, 0.0021, 0.0),
                                (0.0115, 0.0029, 0.0004), (0.0098, 0.0031, 0.0003), (0.0074, 0.0036, 0.0003),
                                (0.0068, 0.0028, 0), (0, 0.0028, 0)], S(5, 2))
        boss = lathe('boss', prof, S(128, 36))
        boss.location = (cx, 0, z)
        boss = apply_transform(boss)
        gem_prof = [(0, z + 0.0024), (0.0067, z + 0.0024)]
        for k in range(1, S(12, 4) + 1):
            t = k / S(12, 4) * math.pi / 2
            gem_prof.append((0.0067 * math.cos(t), z + 0.0030 + 0.0036 * math.sin(t)))
        gem_prof[-1] = (0, gem_prof[-1][1])
        gem = lathe('moonstone', gem_prof, S(64, 20))
        deform(gem, lambda v: Vector((cx + v.x, v.y, v.z)))
        set_material(gem, self.m_stone)
        parts = [boss]
        for i in range(4):
            a = TAU * i / 4 + TAU / 8
            parts.append(dome('nail', (cx + math.cos(a) * 0.0243, math.sin(a) * 0.0243, z + 0.0026), 0.0012,
                              0.0008, S(5, 2)))
        for p in parts:
            set_material(p, self.m_brass)
        return parts + [gem]

    def _clasp(self):
        S = self.S
        X1, H = self.X1, self.H
        top = H + self._bow(X1 - 0.01, 0)
        cp = [(X1 + 0.0011, 0, 0.0016), (X1 + 0.0011, 0, 0.02), (X1 + 0.0011, 0, H - 0.004),
              (X1 - 0.0003, 0, top + 0.0002), (X1 - 0.004, 0, top + 0.0012), (X1 - 0.025, 0, top + 0.0013)]
        path = resample(catmull(cp, 10), S(0.0006, 0.003))
        strap = sweep('strap', path, rect_profile(0.015, 0.0022, 0.0006, S(3, 1)), up=(1, 0, 0))
        set_material(strap, self.m_strap)
        zt = top + 0.0024
        outline = []
        n = S(16, 6)
        for k in range(n + 1):  # rounded tongue towards the spine
            a = math.pi / 2 + math.pi * k / n
            outline.append((X1 - 0.031 + 0.009 * math.cos(a) * 0.55, 0.009 * math.sin(a)))
        outline += [(X1 - 0.012, -0.009), (X1 - 0.010, -0.007), (X1 - 0.010, 0.007), (X1 - 0.012, 0.009)]
        plate = extrude_outline('clasp', outline, zt - 0.0002, zt + 0.0015)
        mod_bevel(plate, 0.0004, S(3, 1), angle=40)
        plate = apply_modifiers(plate)
        catch = rounded_box_mesh('catch', (0.007, 0.013, 0.0032), (X1 - 0.036, 0, top + 0.0013))
        mod_bevel(catch, 0.0009, S(3, 1), angle=40)
        catch = apply_modifiers(catch)
        parts = [plate, catch]
        for lx in (-0.015, -0.021):
            parts.append(dome('rivet', (X1 + lx, 0, zt + 0.0015), 0.0012, 0.0007, S(5, 2)))
        for p in parts:
            set_material(p, self.m_brass)
        return [strap] + parts

    def _ribbon(self):
        S = self.S
        cp = [(0.018, -0.108, 0.033), (0.019, -0.1172, 0.0332), (0.0205, -0.1215, 0.029),
              (0.022, -0.1250, 0.018), (0.025, -0.1300, 0.0065), (0.030, -0.1380, 0.0018),
              (0.038, -0.1490, 0.0011), (0.046, -0.1610, 0.0011)]
        path = resample(catmull(cp, 10), S(0.0005, 0.003))
        fray = lambda t: (1.0 - 0.15 * smoothstep(0.92, 1.0, t), 1.0)
        rib = sweep('ribbon', path, rect_profile(0.0062, 0.00045, 0.0002, S(2, 1)), up=(0, 0, 1), scale_fn=fray)
        set_material(rib, self.m_ribbon)
        return rib

    def build(self, lod):
        self.lod = lod
        parts = [self._cover(True), self._cover(False), self._spine(), self._block(), self._ribbon()]
        for sx in (1,):
            for sy in (1, -1):
                parts += self._corner(sx, sy, True)
                parts += self._corner(sx, sy, False)
        parts += self._boss() + self._clasp()
        for p in parts:
            p.location.z += self.LIFT
            apply_transform(p)
        return {'body': parts, 'glass': []}


def rounded_box_mesh(name, size, center):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    return mesh_object(name, bm)


# ===========================================================================
# Scrying orb
# ===========================================================================

class ScryingOrb(Item):
    """Rock-crystal sphere in a cast brass cradle on a turned walnut plinth with a felt pad.

    The full Elder Futhark row is chased around the brass foot, as on the Kylver stone.
    """
    key = 'ScryingOrb'
    title = 'Kehanet Küresi'
    extrusion = 0.0012
    max_ray = 0.005
    ao_distance = 0.016
    view = dict(yaw=-24, pitch=14)
    F = 0.0015
    RS = 0.055

    @property
    def ZS(self):
        return self.F + 0.1197

    def setup(self):
        rng = random.Random(31)
        cv = Canvas(4096, 256, seed=14)
        row = ['f', 'u', 'th', 'a', 'r', 'k', 'g', 'w', '|', 'h', 'n', 'i', 'j', 'ei', 'p', 'z', 's', '|',
               't', 'b', 'e', 'm', 'l', 'ng', 'd', 'o', '|']
        h = 0.62
        x = 0.25
        for ch in row:
            x = draw_runes(cv, [ch], (x, 0.2), h, 0.045, rng, spacing=0.5)
        z0, z1 = self.F + 0.0330, self.F + 0.0400

        def cyl(mb):
            xyz = mb.sep(mb.coord())
            u = mb.math('MULTIPLY_ADD', mb.math('ARCTAN2', xyz[1], xyz[0]), 1 / TAU, 0.5)
            u = mb.math('MULTIPLY', u, x / cv.aspect)
            v = mb.maprange(xyz[2], z0, z1, 0.0, 1.0, clamp=False)
            return mb.comb(u, v, 0.0)

        band = cv.to_image('MSK_ScryingOrb_Futhark')
        self.m_brass = M.brass('M_ScryingOrb_Brass', masks=[(band, cyl, None)], mask_depth=0.00022)
        self.m_wood = M.wood('M_ScryingOrb_Walnut', 'walnut', grain_axis='X')
        self.m_felt = M.felt('M_ScryingOrb_Felt')

    def glass_material(self):
        return M.glass('M_ScryingOrb_Crystal', tint=(0.93, 0.92, 0.90), rough=0.0)

    def _base(self):
        S, F = self.S, self.F
        felt = lathe('felt', fillet_polyline([(0, 0, 0), (0.0655, 0, 0.0004), (0.0655, F, 0.0004), (0, F, 0)],
                                             S(3, 1)), S(128, 40))
        set_material(felt, self.m_felt)
        wood = fillet_polyline([(0, F, 0), (0.068, F, 0.0015), (0.0702, F + 0.0035, 0.001),
                                (0.0702, F + 0.0072, 0.0018), (0.0662, F + 0.0102, 0.0015),
                                (0.0656, F + 0.0128, 0.0010), (0.0684, F + 0.0156, 0.0024),
                                (0.0662, F + 0.0188, 0.0012), (0.0585, F + 0.0200, 0.0008),
                                (0.0565, F + 0.0258, 0.0035), (0.0400, F + 0.0292, 0.0),
                                (0, F + 0.0292, 0)], S(6, 2))
        base = lathe('plinth', wood, S(160, 44))
        deform(base, lambda v: Vector((v.x * 1.006, v.y * 0.996, v.z)))
        set_material(base, self.m_wood)
        return [felt, base]

    def _stand(self):
        S, F = self.S, self.F
        outer = [(0, F + 0.0288, 0), (0.0420, F + 0.0288, 0.0004), (0.0432, F + 0.0308, 0.0009),
                 (0.0402, F + 0.0326, 0.0008), (0.0330, F + 0.0345, 0.004), (0.0230, F + 0.0378, 0.004),
                 (0.0160, F + 0.0412, 0.003), (0.0086, F + 0.0478, 0.002), (0.0136, F + 0.0530, 0.0035),
                 (0.0086, F + 0.0582, 0.002), (0.0112, F + 0.0602, 0.0015), (0.0220, F + 0.0660, 0.003),
                 (0.0312, F + 0.0712, 0.002), (0.0350, F + 0.0742, 0.0009), (0.0322, F + 0.0753, 0.0007),
                 (0.0200, F + 0.0706, 0.003), (0.0080, F + 0.0656, 0.002), (0, F + 0.0645, 0)]
        st = lathe('stand', fillet_polyline(outer, S(6, 2)), S(160, 44))
        set_material(st, self.m_brass)
        parts = [st]
        Ro = self.RS + 0.0008
        for k, th in enumerate((math.radians(-60), math.radians(60), math.radians(180))):
            pts2 = [(0.0352, F + 0.0700), (0.0356, F + 0.0730)]
            a0 = math.asin(0.0356 / Ro)
            n = S(60, 14)
            for i in range(1, n + 1):
                a = a0 + (math.radians(103) - a0) * i / n
                pts2.append((Ro * math.sin(a), self.ZS - Ro * math.cos(a)))
            # scroll curling away from the sphere
            px, pz = pts2[-1]
            ang = math.atan2(pz - pts2[-2][1], px - pts2[-2][0])
            m = S(70, 16)
            for i in range(1, m + 1):
                t = i / m
                rc = 0.0042 * (1 - 0.65 * t)
                ang -= (TAU * 1.25 / m)
                px += math.cos(ang) * rc * TAU * 1.25 / m
                pz += math.sin(ang) * rc * TAU * 1.25 / m
                pts2.append((px, pz))
            radial = Vector((math.cos(th), math.sin(th), 0))
            pts = [radial * r + Vector((0, 0, z)) for r, z in pts2]
            pts = resample(pts, S(0.0004, 0.0018))
            normals = []
            for i in range(len(pts)):
                d = pts[min(len(pts) - 1, i + 1)] - pts[max(0, i - 1)]
                tr, tz = d.dot(radial), d.z
                normals.append(radial * tz - Vector((0, 0, 1)) * tr)
            taper = lambda t: (1.0 - 0.45 * smoothstep(0.0, 1.0, t), 1.0 - 0.15 * t)
            prong = sweep(f'prong{k}', pts, rect_profile(0.0062, 0.0015, 0.0005, S(3, 1)),
                          normal_fn=lambda i, p, t, nn=normals: nn[i], scale_fn=taper)
            set_material(prong, self.m_brass)
            parts.append(prong)
            rv = dome(f'rivet{k}', radial * 0.0368 + Vector((0, 0, F + 0.0716)), 0.0013, 0.0008, S(5, 2),
                      normal=radial)
            set_material(rv, self.m_brass)
            parts.append(rv)
        return parts

    def _sphere(self):
        ob = uv_sphere('crystal', self.RS, self.S(128, 48), self.S(64, 24))
        ob.location = (0, 0, self.ZS)
        return apply_transform(ob)

    def build(self, lod):
        self.lod = lod
        return {'body': self._base() + self._stand(), 'glass': [self._sphere()]}


# ===========================================================================
# Hourglass
# ===========================================================================

class Hourglass(Item):
    """Three-post oak sandglass, hand-blown twin bulbs bound with waxed linen; most of the
    sand has already run through."""
    key = 'Hourglass'
    title = 'Kum Saati'
    extrusion = 0.0012
    max_ray = 0.005
    ao_distance = 0.02
    view = dict(yaw=-28, pitch=10)
    emit_strength = 0.08
    H = 0.19
    ZC = 0.095
    T = 0.0012
    OUTER = [(0.0125, -0.0825), (0.0138, -0.0815), (0.0126, -0.0800), (0.0160, -0.0750), (0.0258, -0.0620),
             (0.0303, -0.0480), (0.0306, -0.0400), (0.0278, -0.0280), (0.0182, -0.0140), (0.0080, -0.0050),
             (0.0034, -0.0013), (0.0029, 0.0)]

    def setup(self):
        self.m_wood = M.wood('M_Hourglass_OakPosts', 'oak', grain_axis='Z', ring_scale=70)
        self.m_plate = M.wood('M_Hourglass_OakPlates', 'oak', grain_axis='X', ring_scale=70,
                              pith=(0.0, 0.31, -0.42))
        self.m_sand = M.sand('M_Hourglass_Sand')
        self.m_bind = M.twine('M_Hourglass_Binding', color=(0.10, 0.065, 0.035), wrap_pitch=0.0008)

    def glass_material(self):
        return M.glass('M_Hourglass_Glass', tint=(0.90, 0.93, 0.91), rough=0.02)

    def _outer(self):
        half = [(p.x, p.y) for p in catmull(self.OUTER, self.S(6, 2))]
        top = [(r, -z) for r, z in half[::-1]][1:]
        return half + top

    def _inner_r(self, dz):
        return _profile_r(offset_profile(self._outer(), self.T, 0.0013), dz)

    def _glass(self):
        S, ZC = self.S, self.ZC
        outer = self._outer()
        inner = offset_profile(outer, self.T, 0.0013)
        inner[0] = (inner[0][0], -0.0825 + self.T)
        inner[-1] = (inner[-1][0], 0.0825 - self.T)
        prof = [(0, -0.0825)] + outer + [(0, 0.0825), (0, 0.0825 - self.T)] + inner[::-1] + [(0, -0.0825 + self.T)]
        ob = lathe('glass', [(r, z + ZC) for r, z in prof], S(96, 32))
        deform(ob, lambda v: Vector((v.x * (1 + 0.006 * math.sin(v.z * 90)), v.y, v.z)))
        return ob

    def _sand(self):
        S, ZC = self.S, self.ZC
        inner = offset_profile(self._outer(), self.T, 0.0013)
        rin = lambda dz: (_profile_r(inner, dz) or 0.002) - 0.0003
        repose = math.tan(math.radians(33))
        zb, ze = -0.0825 + self.T + 0.0002, -0.051
        prof = [(0, zb)]
        n = S(40, 10)
        for i in range(n + 1):
            z = zb + 0.0003 + (ze - zb - 0.0003) * i / n
            prof.append((rin(z), z))
        re = rin(ze)
        apex = ze + re * repose
        for i in range(1, n + 1):
            r = re * (1 - i / n)
            prof.append((r, ze + (re - r) * repose - 0.0008 * smoothstep(0.003, 0.0, r)))
        prof[-1] = (0, apex - 0.0008)
        bottom = lathe('sand_low', [(r, z + ZC) for r, z in prof], S(96, 28))

        zt0, zte = 0.0004, 0.0175
        prof = [(0, zt0)]
        for i in range(1, n + 1):
            z = zt0 + (zte - zt0) * i / n
            prof.append((rin(z), z))
        re = rin(zte)
        for i in range(1, n + 1):
            r = re * (1 - i / n)
            prof.append((r, zte - (re - r) * repose))
        prof[-1] = (0, max(zt0 + 0.0008, prof[-1][1]))
        top = lathe('sand_top', [(r, z + ZC) for r, z in prof], S(96, 28))
        stream = lathe('stream', [(0, apex - 0.0012 + ZC), (0.00042, apex - 0.0012 + ZC), (0.00036, 0.0006 + ZC),
                                  (0, 0.0008 + ZC)], S(12, 6))
        parts = [bottom, top, stream]
        for p in parts:
            set_material(p, self.m_sand)
        mod_displace(bottom, legacy_texture('CLOUDS', noise_scale=0.003, noise_depth=2), 0.0002)
        return [apply_modifiers(bottom), top, stream]

    def _binding(self):
        S, ZC = self.S, self.ZC
        outer = self._outer()
        gr = lambda dz: _profile_r(outer, dz)
        z0, z1 = -0.0068, 0.0068
        n = S(40, 10)
        prof = [(gr(z0) - 0.0004, z0 + 0.0002), (gr(z0) + 0.0006, z0)]
        for i in range(n + 1):
            z = z0 + 0.0004 + (z1 - z0 - 0.0008) * i / n
            prof.append((gr(z) + 0.0012 + 0.0002 * math.sin(i * 1.7), z))
        prof += [(gr(z1) + 0.0006, z1), (gr(z1) - 0.0004, z1 - 0.0002)]
        for i in range(n + 1):
            z = z1 - 0.0004 - (z1 - z0 - 0.0008) * i / n
            prof.append((gr(z) - 0.0004, z))
        ob = lathe('binding', [(r, z + ZC) for r, z in prof], S(96, 28), cyclic=True)
        set_material(ob, self.m_bind)
        return ob

    def _plates(self):
        S, H = self.S, self.H
        prof = fillet_polyline([(0, 0, 0), (0.0505, 0, 0.002), (0.0522, 0.0030, 0.0008), (0.0522, 0.0060, 0.0015),
                                (0.0492, 0.0080, 0.0012), (0.0472, 0.0098, 0.0008), (0.0442, 0.0122, 0.0015),
                                (0.0320, 0.0126, 0), (0, 0.0126, 0)], S(6, 2))
        low = lathe('plate_low', prof, S(160, 44))
        top = lathe('plate_top', [(r, H - z) for r, z in prof[::-1]], S(160, 44))
        for i, p in enumerate((low, top)):
            deform(p, lambda v, i=i: Vector((v.x * (1.008 if i else 0.994), v.y * (0.995 if i else 1.006), v.z)))
            set_material(p, self.m_plate)
        return [low, top]

    def _posts(self):
        S = self.S
        half = [(0.0035, -0.003), (0.0035, 0.0), (0.0058, 0.0008), (0.0062, 0.0035), (0.0048, 0.0062),
                (0.0040, 0.0085), (0.0043, 0.0120), (0.0058, 0.0300), (0.0050, 0.0450), (0.0037, 0.0600),
                (0.0041, 0.0660), (0.0056, 0.0695), (0.0043, 0.0730), (0.0039, 0.0770), (0.0050, 0.0805),
                (0.0060, 0.0825)]
        L = 0.165
        full = half + [(r, L - z) for r, z in half[::-1][1:]]
        smooth = [(p.x, p.y) for p in catmull(full, S(5, 1))]
        prof = [(0, -0.003)] + smooth + [(0, L + 0.003)]
        rng = random.Random(17)
        posts = []
        for k, a in enumerate((90, 210, 330)):
            sr = rng.uniform(0.97, 1.03)
            ob = lathe(f'post{k}', [(r * sr, z) for r, z in prof], S(64, 16))
            th = math.radians(a)
            ob.location = (math.cos(th) * 0.0405, math.sin(th) * 0.0405, 0.0126)
            ob = apply_transform(ob)
            set_material(ob, self.m_wood)
            posts.append(ob)
        return posts

    def build(self, lod):
        self.lod = lod
        body = self._plates() + self._posts() + self._sand() + [self._binding()]
        return {'body': body, 'glass': [self._glass()]}


ITEMS = [RuneStone, Amulet, Potion, Grimoire, ScryingOrb, Hourglass]
