"""Core helpers for the SeekThenHide mystic item generator.

Geometry is authored twice from the same deterministic code: once as a dense
"high" mesh (hp) that carries every bevel, chip and seam, and once as a light
game mesh (lp). The hp surface is then baked down onto the lp UVs.
"""
import math
import os

import bpy  # must precede bmesh when bpy is used as a pip module
import bmesh
import numpy as np
from mathutils import Vector

TAU = math.tau


# --------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.cycles.device = 'CPU'
    scn.unit_settings.system = 'METRIC'
    scn.unit_settings.length_unit = 'CENTIMETERS'
    world = bpy.data.worlds.new("World")
    scn.world = world
    import addon_utils
    for mod in ('io_scene_gltf2', 'io_scene_fbx'):
        addon_utils.enable(mod, default_set=True)
    return scn


def collection(name, parent=None):
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if col.name not in parent.children:
        parent.children.link(col)
    return col


def link(ob, col=None):
    (col or bpy.context.scene.collection).objects.link(ob)
    return ob


def mesh_object(name, bm, col=None, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    return link(ob, col)


def set_material(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    return ob


def select_only(*obs, active=None):
    for o in bpy.context.scene.objects:
        if o is not None:
            o.select_set(False)
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or (obs[-1] if obs else None)


# --------------------------------------------------------------------------
# modifiers
# --------------------------------------------------------------------------

def mod_bevel(ob, width, segments, angle=35.0, profile=0.5, clamp=True):
    m = ob.modifiers.new("Bevel", 'BEVEL')
    m.width = width
    m.segments = segments
    m.limit_method = 'ANGLE'
    m.angle_limit = math.radians(angle)
    m.profile = profile
    m.use_clamp_overlap = clamp
    return m


def mod_subsurf(ob, levels, simple=False):
    if levels <= 0:
        return None
    m = ob.modifiers.new("Subsurf", 'SUBSURF')
    if simple:
        m.subdivision_type = 'SIMPLE'
    m.levels = levels
    m.render_levels = levels
    m.quality = 3
    return m


_tex_cache = {}


def legacy_texture(kind, **props):
    key = (kind, tuple(sorted(props.items())))
    if key in _tex_cache and _tex_cache[key].name in bpy.data.textures:
        return _tex_cache[key]
    t = bpy.data.textures.new(f"TX_{kind}", kind)
    for k, v in props.items():
        setattr(t, k, v)
    _tex_cache[key] = t
    return t


def mod_displace(ob, tex, strength, mid=0.5, coords='LOCAL', direction='NORMAL', vgroup=None):
    m = ob.modifiers.new("Displace", 'DISPLACE')
    m.texture = tex
    m.strength = strength
    m.mid_level = mid
    m.texture_coords = coords
    m.direction = direction
    if vgroup:
        m.vertex_group = vgroup
    return m


def mod_remesh(ob, voxel):
    m = ob.modifiers.new("Remesh", 'REMESH')
    m.mode = 'VOXEL'
    m.voxel_size = voxel
    m.adaptivity = 0.0
    m.use_smooth_shade = True
    return m


def mod_smooth(ob, factor=0.5, iterations=2):
    m = ob.modifiers.new("Smooth", 'SMOOTH')
    m.factor = factor
    m.iterations = iterations
    return m


def mod_boolean(ob, cutter, op='DIFFERENCE'):
    m = ob.modifiers.new("Bool", 'BOOLEAN')
    m.operation = op
    m.object = cutter
    m.solver = 'EXACT'
    return m


def apply_modifiers(ob):
    """Bake the modifier stack (or curve geometry) into a plain mesh object."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    if ob.type == 'MESH':
        old = ob.data
        ob.modifiers.clear()
        ob.data = me
        bpy.data.meshes.remove(old)
        me.name = ob.name
        return ob
    new = bpy.data.objects.new(ob.name, me)
    new.matrix_world = ob.matrix_world
    for c in ob.users_collection:
        c.objects.link(new)
    bpy.data.objects.remove(ob)
    return new


def deform(ob, fn):
    """Move every vertex through fn(Vector) -> Vector (deterministic per position)."""
    me = ob.data
    for v in me.vertices:
        v.co = fn(v.co.copy())
    me.update()
    return ob


def join(objs, name, col=None):
    """Merge mesh objects (with their transforms and material slots) into one."""
    bpy.context.view_layer.update()
    bm = bmesh.new()
    mats = []
    for ob in objs:
        tmp = ob.data.copy()
        tmp.transform(ob.matrix_world)
        remap = []
        for m in tmp.materials:
            if m not in mats:
                mats.append(m)
            remap.append(mats.index(m))
        if remap:
            idx = np.zeros(len(tmp.polygons), dtype=np.int32)
            tmp.polygons.foreach_get('material_index', idx)
            tmp.polygons.foreach_set('material_index', np.array(remap, dtype=np.int32)[idx])
        part = bmesh.new()
        part.from_mesh(tmp)
        # every part is a closed solid; make sure nothing arrives inside-out
        bmesh.ops.recalc_face_normals(part, faces=part.faces)
        part.to_mesh(tmp)
        part.free()
        bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    for ob in objs:
        bpy.data.objects.remove(ob)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    return link(ob, col)


def tri_count(ob):
    return sum(len(p.vertices) - 2 for p in ob.data.polygons)


# --------------------------------------------------------------------------
# profiles & primitives
# --------------------------------------------------------------------------

def fillet_polyline(pts, seg):
    """pts: [(x, y, radius)]; returns a polyline with each corner rounded by `radius`."""
    out = []
    n = len(pts)
    for i, (x, y, rad) in enumerate(pts):
        if rad <= 0 or i == 0 or i == n - 1:
            out.append((x, y))
            continue
        p = Vector((x, y))
        a = Vector(pts[i - 1][:2])
        b = Vector(pts[i + 1][:2])
        da = (a - p).normalized()
        db = (b - p).normalized()
        cosang = max(-1.0, min(1.0, da.dot(db)))
        ang = math.acos(cosang)
        if ang < 1e-3 or abs(ang - math.pi) < 1e-3:
            out.append((x, y))
            continue
        t = rad / math.tan(ang / 2)
        t = min(t, (a - p).length * 0.49, (b - p).length * 0.49)
        p0 = p + da * t
        p1 = p + db * t
        # quadratic bezier through the corner keeps the arc tangent to both legs
        for k in range(seg + 1):
            s = k / seg
            q = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p + s * s * p1
            out.append((q.x, q.y))
    return out


def catmull(pts, samples_per_seg, closed=False):
    """Centripetal-ish Catmull-Rom through 2D/3D points."""
    P = [Vector(p) for p in pts]
    if closed:
        P = [P[-1]] + P + [P[0], P[1]]
    else:
        P = [P[0] + (P[0] - P[1])] + P + [P[-1] + (P[-1] - P[-2])]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(samples_per_seg):
            t = k / samples_per_seg
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(q)
    if not closed:
        out.append(P[-2])
    return out


def resample(pts, step):
    """Resample a polyline at a constant arc-length step."""
    pts = [Vector(p) for p in pts]
    d = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        d.append(d[-1] + (b - a).length)
    total = d[-1]
    n = max(2, int(round(total / step)) + 1)
    out = []
    j = 0
    for k in range(n):
        s = total * k / (n - 1)
        while j < len(d) - 2 and d[j + 1] < s:
            j += 1
        seg = d[j + 1] - d[j]
        t = 0 if seg == 0 else (s - d[j]) / seg
        out.append(pts[j].lerp(pts[j + 1], t))
    return out


def lathe(name, prof, seg, col=None, cyclic=False, angle0=0.0, smooth=True):
    """Surface of revolution around Z. prof: [(r, z)] walked so the solid is on the left."""
    bm = bmesh.new()
    rings = []
    for r, z in prof:
        if r < 1e-7:
            rings.append([bm.verts.new((0.0, 0.0, z))])
        else:
            rings.append([bm.verts.new((r * math.cos(angle0 + TAU * i / seg),
                                        r * math.sin(angle0 + TAU * i / seg), z))
                          for i in range(seg)])
    pairs = list(zip(rings[:-1], rings[1:]))
    if cyclic:
        pairs.append((rings[-1], rings[0]))
    for A, B in pairs:
        if len(A) == 1 and len(B) == 1:
            continue
        for i in range(seg):
            j = (i + 1) % seg
            if len(A) == 1:
                bm.faces.new((A[0], B[j], B[i]))
            elif len(B) == 1:
                bm.faces.new((A[i], A[j], B[0]))
            else:
                bm.faces.new((A[i], A[j], B[j], B[i]))
    # UV seams: one meridian, plus a ring wherever the profile turns a hard corner
    for A, B in pairs:
        if len(A) > 1 or len(B) > 1:
            _seam(bm, A[0], B[0])
    n = len(prof)
    for k in range(n):
        if len(rings[k]) == 1 or (not cyclic and k in (0, n - 1)):
            continue
        d0 = Vector(prof[k]) - Vector(prof[k - 1])
        d1 = Vector(prof[(k + 1) % n]) - Vector(prof[k])
        if (cyclic and k == 0) or (d0.length > 0 and d1.length > 0 and d0.angle(d1) > math.radians(40)):
            ring = rings[k]
            for i in range(seg):
                _seam(bm, ring[i], ring[(i + 1) % seg])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    return mesh_object(name, bm, col, smooth)


def _seam(bm, a, b):
    e = bm.edges.get((a, b))
    if e is not None:
        e.seam = True


def frames_along(path, up=None, closed=False):
    """Parallel-transport frames (T, N, B) along a polyline."""
    P = [Vector(p) for p in path]
    n = len(P)
    T = []
    for i in range(n):
        if closed:
            t = P[(i + 1) % n] - P[i - 1]
        elif i == 0:
            t = P[1] - P[0]
        elif i == n - 1:
            t = P[-1] - P[-2]
        else:
            t = P[i + 1] - P[i - 1]
        T.append(t.normalized())
    if up is None:
        a = Vector((0, 0, 1)) if abs(T[0].z) < 0.9 else Vector((1, 0, 0))
    else:
        a = Vector(up)
    N0 = (a - T[0] * a.dot(T[0])).normalized()
    N = [N0]
    for i in range(1, n):
        q = T[i - 1].rotation_difference(T[i])
        v = q @ N[-1]
        v = (v - T[i] * v.dot(T[i])).normalized()
        N.append(v)
    B = [t.cross(nv) for t, nv in zip(T, N)]
    return P, T, N, B


def sweep(name, path, profile, col=None, closed=False, up=None, normal_fn=None,
          scale_fn=None, twist=0.0, caps=True, smooth=True):
    """Sweep a closed 2D profile [(u, v)] along a path.

    normal_fn(i, p, t) can force the frame's N axis (e.g. a surface normal) so
    flat straps stay glued to whatever they wrap around.
    """
    P, T, N, B = frames_along(path, up, closed)
    n = len(P)
    lengths = [0.0]
    for a, b in zip(P[:-1], P[1:]):
        lengths.append(lengths[-1] + (b - a).length)
    total = lengths[-1] or 1.0
    bm = bmesh.new()
    rings = []
    for i in range(n):
        t = lengths[i] / total
        Ni, Bi = N[i], B[i]
        if normal_fn:
            want = Vector(normal_fn(i, P[i], t))
            Ni = (want - T[i] * want.dot(T[i])).normalized()
            Bi = T[i].cross(Ni)
        a = twist * t
        if a:
            Ni, Bi = Ni * math.cos(a) + Bi * math.sin(a), -Ni * math.sin(a) + Bi * math.cos(a)
        s = scale_fn(t) if scale_fn else 1.0
        su, sv = (s if isinstance(s, tuple) else (s, s))
        rings.append([bm.verts.new(P[i] + Bi * (u * su) + Ni * (v * sv)) for u, v in profile])
    m = len(profile)
    pairs = list(zip(rings[:-1], rings[1:]))
    if closed:
        # pick the vertex offset that removes any holonomy twist at the seam
        last, first = rings[-1], rings[0]
        best = min(range(m), key=lambda o: sum((last[k].co - first[(k + o) % m].co).length
                                                for k in range(m)))
        pairs.append((last, first[best:] + first[:best]))
    for A, Bv in pairs:
        for k in range(m):
            k1 = (k + 1) % m
            bm.faces.new((A[k], A[k1], Bv[k1], Bv[k]))
        _seam(bm, A[0], Bv[0])
    if caps and not closed:
        for ring in (rings[0], rings[-1]):
            c = bm.verts.new(sum((v.co for v in ring), Vector()) / m)
            for k in range(m):
                k1 = (k + 1) % m
                bm.faces.new((ring[k], ring[k1], c))
    if closed or caps:
        for ring in ((rings[0],) if closed else (rings[0], rings[-1])):
            for k in range(m):
                _seam(bm, ring[k], ring[(k + 1) % m])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return mesh_object(name, bm, col, smooth)


def offset_profile(prof, t, min_r=0.0):
    """Offset an (r, z) polyline by t to its left (into the solid). Returns a new list."""
    out = []
    n = len(prof)
    for i in range(n):
        a = Vector(prof[max(0, i - 1)])
        b = Vector(prof[min(n - 1, i + 1)])
        d = (b - a).normalized()
        left = Vector((-d.y, d.x))
        q = Vector(prof[i]) + left * t
        out.append((max(min_r, q.x), q.y))
    return out


def circle_profile(r, seg, sx=1.0, sy=1.0):
    return [(r * sx * math.cos(TAU * i / seg), r * sy * math.sin(TAU * i / seg)) for i in range(seg)]


def rect_profile(w, h, r, seg):
    """Rounded rectangle profile (u = width, v = height)."""
    pts = []
    hw, hh = w / 2, h / 2
    r = min(r, hw * 0.99, hh * 0.99)
    for cx, cy, a0 in ((hw - r, hh - r, 0), (-hw + r, hh - r, 90), (-hw + r, -hh + r, 180), (hw - r, -hh + r, 270)):
        for k in range(seg + 1):
            a = math.radians(a0 + 90 * k / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def rounded_box(name, size, bevel, seg, col=None, subdiv=0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    ob = mesh_object(name, bm, col)
    if bevel > 0:
        mod_bevel(ob, bevel, seg, angle=30)
    mod_subsurf(ob, subdiv)
    return ob


def uv_sphere(name, r, seg, rings, col=None):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=r)
    return mesh_object(name, bm, col)


def ico_sphere(name, r, subdiv, col=None):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=r)
    return mesh_object(name, bm, col)


def stadium_path(length, width, n):
    """Closed oval (two half circles + straights) in the XY plane, centred on origin."""
    r = width / 2
    straight = max(0.0, length - width)
    perim = 2 * straight + TAU * r
    pts = []
    for k in range(n):
        s = perim * k / n
        if s < straight:
            pts.append((-straight / 2 + s, -r, 0))
        elif s < straight + math.pi * r:
            a = -math.pi / 2 + (s - straight) / r
            pts.append((straight / 2 + r * math.cos(a), r * math.sin(a), 0))
        elif s < 2 * straight + math.pi * r:
            pts.append((straight / 2 - (s - straight - math.pi * r), r, 0))
        else:
            a = math.pi / 2 + (s - 2 * straight - math.pi * r) / r
            pts.append((-straight / 2 + r * math.cos(a), r * math.sin(a), 0))
    return pts


def strands_along(name, path, ply_r, strands, turns_per_m, seg, col=None, phase=0.0, scale_fn=None):
    """Twisted cord: `strands` plies wound around the centre line of `path`."""
    P, T, N, B = frames_along(path)
    lengths = [0.0]
    for a, b in zip(P[:-1], P[1:]):
        lengths.append(lengths[-1] + (b - a).length)
    objs = []
    for s in range(strands):
        pts = []
        for i, p in enumerate(P):
            a = phase + TAU * s / strands + TAU * turns_per_m * lengths[i]
            off = ply_r * 0.85
            if scale_fn:
                off *= scale_fn(lengths[i] / lengths[-1])
            pts.append(p + N[i] * math.cos(a) * off + B[i] * math.sin(a) * off)
        objs.append(sweep(f"{name}_ply{s}", pts, circle_profile(ply_r, seg), col,
                          scale_fn=scale_fn))
    return objs


# --------------------------------------------------------------------------
# mask images (numpy) — runes, sigils, handwriting
# --------------------------------------------------------------------------

def value_noise(h, w, cells, seed, octaves=4, persistence=0.5):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        c = int(cells * 2 ** o)
        grid = rng.random((c + 2, c + 2)).astype(np.float32)
        ys = np.linspace(0, c, h, endpoint=False)
        xs = np.linspace(0, c, w, endpoint=False)
        y0 = ys.astype(int)
        x0 = xs.astype(int)
        fy = ys - y0
        fx = xs - x0
        fy = fy * fy * (3 - 2 * fy)
        fx = fx * fx * (3 - 2 * fx)
        g00 = grid[y0][:, x0]
        g01 = grid[y0][:, x0 + 1]
        g10 = grid[y0 + 1][:, x0]
        g11 = grid[y0 + 1][:, x0 + 1]
        v = (g00 * (1 - fx) + g01 * fx) * (1 - fy)[:, None] + (g10 * (1 - fx) + g11 * fx) * fy[:, None]
        out += amp * v
        total += amp
        amp *= persistence
    return out / total


class Canvas:
    """Float canvas in normalised coords: x in [0, aspect], y in [0, 1] (y up)."""

    def __init__(self, w, h, seed=0):
        self.w, self.h = w, h
        self.aspect = w / h
        self.img = np.zeros((h, w), np.float32)
        ys = (np.arange(h) + 0.5) / h
        xs = (np.arange(w) + 0.5) / h
        self.X, self.Y = np.meshgrid(xs, ys)
        self.rough = value_noise(h, w, 24, seed, 3) - 0.5

    def stroke(self, a, b, width, depth=1.0, rough=0.25, profile=0.8):
        ax, ay = a
        bx, by = b
        pad = width * 1.5
        x0 = int(max(0, (min(ax, bx) - pad) * self.h))
        x1 = int(min(self.w, (max(ax, bx) + pad) * self.h + 1))
        y0 = int(max(0, (min(ay, by) - pad) * self.h))
        y1 = int(min(self.h, (max(ay, by) + pad) * self.h + 1))
        if x1 <= x0 or y1 <= y0:
            return
        X = self.X[y0:y1, x0:x1]
        Y = self.Y[y0:y1, x0:x1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy + 1e-12
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1)
        px = ax + t * dx - X
        py = ay + t * dy - Y
        d = np.sqrt(px * px + py * py)
        d = d * (1 + rough * 2 * self.rough[y0:y1, x0:x1])
        v = np.clip(1 - d / width, 0, 1) ** profile * depth
        np.maximum(self.img[y0:y1, x0:x1], v, out=self.img[y0:y1, x0:x1])

    def polyline(self, pts, width, depth=1.0, rough=0.25, profile=0.8):
        for a, b in zip(pts[:-1], pts[1:]):
            self.stroke(a, b, width, depth, rough, profile)

    def dot(self, c, r, depth=1.0, profile=0.8):
        d = np.sqrt((self.X - c[0]) ** 2 + (self.Y - c[1]) ** 2)
        v = np.clip(1 - d / r, 0, 1) ** profile * depth
        np.maximum(self.img, v, out=self.img)

    def ring(self, c, radius, width, depth=1.0, profile=0.8, rough=0.2):
        d = np.abs(np.sqrt((self.X - c[0]) ** 2 + (self.Y - c[1]) ** 2) - radius)
        d = d * (1 + rough * 2 * self.rough)
        v = np.clip(1 - d / width, 0, 1) ** profile * depth
        np.maximum(self.img, v, out=self.img)

    def to_image(self, name, colorspace='Non-Color'):
        # Round-trip through a PNG: packing a generated image can silently drop edited pixels.
        a = np.clip(self.img, 0, 1)
        rgba = np.stack([a, a, a, np.ones_like(a)], axis=-1)
        path = os.path.join(_mask_dir(), f"{name}.png")
        tmp = bpy.data.images.new(name + "_tmp", self.w, self.h, alpha=False, float_buffer=False)
        tmp.pixels.foreach_set(rgba.ravel())
        tmp.filepath_raw = path
        tmp.file_format = 'PNG'
        tmp.save()
        bpy.data.images.remove(tmp)
        img = bpy.data.images.load(path)
        img.name = name
        img.colorspace_settings.name = colorspace
        img.pack()
        return img


_MASK_DIR = None


def _mask_dir():
    global _MASK_DIR
    if _MASK_DIR is None:
        import tempfile
        _MASK_DIR = tempfile.mkdtemp(prefix='sth_masks_')
    return _MASK_DIR


# Elder Futhark glyphs as stroke lists in a 0.5 x 1.0 box (x right, y up).
RUNES = {
    'f': [[(0, 0), (0, 1)], [(0, 0.72), (0.42, 0.98)], [(0, 0.42), (0.42, 0.7)]],
    'u': [[(0, 0), (0, 1), (0.42, 0.68), (0.42, 0)]],
    'th': [[(0, 0), (0, 1)], [(0, 0.74), (0.36, 0.5), (0, 0.26)]],
    'a': [[(0, 0), (0, 1)], [(0, 0.98), (0.4, 0.74)], [(0, 0.7), (0.4, 0.46)]],
    'r': [[(0, 0), (0, 1)], [(0, 1), (0.38, 0.78), (0, 0.54), (0.42, 0)]],
    'k': [[(0.36, 0.86), (0.02, 0.6), (0.36, 0.34)]],
    'g': [[(0, 0), (0.5, 1)], [(0, 1), (0.5, 0)]],
    'w': [[(0, 0), (0, 1)], [(0, 1), (0.36, 0.8), (0, 0.6)]],
    'h': [[(0, 0), (0, 1)], [(0.46, 0), (0.46, 1)], [(0, 0.66), (0.46, 0.38)]],
    'n': [[(0.22, 0), (0.22, 1)], [(0.02, 0.62), (0.42, 0.4)]],
    'i': [[(0.1, 0), (0.1, 1)]],
    'j': [[(0.04, 0.72), (0.2, 0.86), (0.2, 0.56)], [(0.36, 0.44), (0.2, 0.3), (0.2, 0.14)]],
    'p': [[(0, 0), (0, 1)], [(0, 1), (0.32, 0.8), (0.32, 0.98)], [(0, 0), (0.32, 0.2), (0.32, 0.02)]],
    'z': [[(0.25, 0), (0.25, 1)], [(0.25, 0.58), (0, 1)], [(0.25, 0.58), (0.5, 1)]],
    's': [[(0.38, 1), (0.04, 0.66), (0.38, 0.34), (0.04, 0)]],
    't': [[(0.25, 0), (0.25, 1)], [(0, 0.72), (0.25, 1), (0.5, 0.72)]],
    'b': [[(0, 0), (0, 1)], [(0, 1), (0.36, 0.76), (0, 0.5), (0.36, 0.24), (0, 0)]],
    'e': [[(0, 0), (0, 1), (0.25, 0.68), (0.5, 1), (0.5, 0)]],
    'm': [[(0, 0), (0, 1)], [(0.5, 0), (0.5, 1)], [(0, 1), (0.5, 0.58)], [(0.5, 1), (0, 0.58)]],
    'l': [[(0, 0), (0, 1)], [(0, 1), (0.38, 0.74)]],
    'ng': [[(0.25, 0.74), (0.5, 0.5), (0.25, 0.26), (0, 0.5), (0.25, 0.74)]],
    'd': [[(0, 0), (0, 1), (0.6, 0), (0.6, 1), (0, 0)]],
    'o': [[(0, 0), (0.5, 0.56), (0.25, 0.86), (0, 0.56), (0.5, 0)]],
    'ei': [[(0.22, 0), (0.22, 1)], [(0.22, 1), (0.44, 0.84)], [(0.22, 0), (0.0, 0.16)]],
    '|': [],  # word separator, drawn as two dots
}


def rune_width(ch):
    if ch == '|':
        return 0.22
    xs = [p[0] for s in RUNES[ch] for p in s]
    return max(xs) if xs else 0.1


def draw_runes(cv, text, origin, height, stroke_w, rng, spacing=0.34, slant=0.0, depth=1.0):
    """Lay runes left to right with hand-carved jitter. Returns end x."""
    x, y = origin
    for ch in text:
        if ch == '|':
            cv.dot((x + 0.06 * height, y + 0.62 * height), stroke_w * 1.15, depth)
            cv.dot((x + 0.06 * height, y + 0.34 * height), stroke_w * 1.15, depth)
            x += 0.22 * height
            continue
        jit = lambda: (rng.uniform(-0.03, 0.03) * height, rng.uniform(-0.03, 0.03) * height)
        sc = height * rng.uniform(0.95, 1.04)
        dy = rng.uniform(-0.03, 0.03) * height
        for strokes in RUNES[ch]:
            pts = []
            for px, py in strokes:
                jx, jy = jit()
                pts.append((x + (px + slant * py) * sc + jx, y + dy + py * sc + jy))
            w = stroke_w * rng.uniform(0.85, 1.12)
            cv.polyline(pts, w, depth)
        x += (rune_width(ch) + spacing) * height
    return x


def runes_on_arc(cv, text, center, radius, height, stroke_w, rng, start_angle, depth=1.0):
    """Runes standing on a circle (feet towards the centre), running clockwise."""
    cx, cy = center
    ang = start_angle
    for ch in text:
        w = rune_width(ch) + 0.3
        step = w * height / radius
        a_mid = ang - step / 2
        up = Vector((math.cos(a_mid), math.sin(a_mid)))
        right = Vector((up.y, -up.x))
        base = Vector((cx, cy)) + up * radius - right * (rune_width(ch) * height / 2)
        if ch == '|':
            for py in (0.34, 0.66):
                p = base + right * 0.05 * height + up * py * height
                cv.dot((p.x, p.y), stroke_w * 1.1, depth)
        else:
            for strokes in RUNES[ch]:
                pts = []
                for px, py in strokes:
                    p = base + right * (px * height + rng.uniform(-0.02, 0.02) * height) \
                        + up * (py * height + rng.uniform(-0.02, 0.02) * height)
                    pts.append((p.x, p.y))
                cv.polyline(pts, stroke_w * rng.uniform(0.88, 1.1), depth)
        ang -= step


def handwriting(cv, x0, y0, line_w, line_h, lines, rng, pen=0.0025):
    """Cursive-like ink scribbles: words made of looped strokes on a baseline."""
    for li in range(lines):
        x = x0 + rng.uniform(0, 0.02)
        y = y0 - li * line_h
        end = x0 + line_w * (rng.uniform(0.55, 0.95) if li == lines - 1 else rng.uniform(0.9, 1.0))
        while x < end:
            wlen = rng.uniform(0.05, 0.16) * line_h * 6
            n = max(4, int(wlen / (line_h * 0.18)))
            pts = []
            ph = rng.uniform(0, TAU)
            for k in range(n * 6 + 1):
                t = k / (n * 6)
                xx = x + t * wlen
                loop = math.sin(t * n * math.pi + ph)
                tall = 1.0 + (rng.random() < 0.08) * 1.4
                yy = y + line_h * 0.32 * (0.5 + 0.5 * loop) * tall * (0.8 + 0.2 * math.sin(t * 7 + ph))
                xx += line_h * 0.11 * math.cos(t * n * math.pi + ph)
                pts.append((xx, yy))
            cv.polyline(pts, pen * rng.uniform(0.7, 1.2), 1.0, rough=0.1, profile=0.6)
            x += wlen + line_h * rng.uniform(0.25, 0.45)


# --------------------------------------------------------------------------
# material builder
# --------------------------------------------------------------------------

def _sock(sockets, name):
    if isinstance(name, int):
        return sockets[name]
    for s in sockets:
        if s.name == name and s.enabled:
            return s
    for s in sockets:
        if s.name == name:
            return s
    raise KeyError(name)


class MB:
    """Tiny shader-graph builder. Methods return output sockets."""

    def __init__(self, name):
        m = bpy.data.materials.new(name)
        if m.node_tree is None:
            m.use_nodes = True
        self.mat = m
        self.nt = m.node_tree
        self.nt.nodes.clear()
        self.out = self.nt.nodes.new('ShaderNodeOutputMaterial')
        self.bsdf = self.nt.nodes.new('ShaderNodeBsdfPrincipled')
        self.nt.links.new(self.bsdf.outputs[0], self.out.inputs[0])
        self._tc = None
        self._geo = None

    def node(self, t, **props):
        n = self.nt.nodes.new(t)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def put(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        else:
            if isinstance(v, (tuple, list)) and len(v) == 3 and sock.type == 'RGBA':
                v = (*v, 1.0)
            sock.default_value = v

    # inputs ---------------------------------------------------------------
    def coord(self, kind='Object', obj=None):
        if obj is not None:
            tc = self.node('ShaderNodeTexCoord')
            tc.object = obj
            return tc.outputs[kind]
        if self._tc is None:
            self._tc = self.node('ShaderNodeTexCoord')
        return self._tc.outputs[kind]

    def geo(self, out):
        if self._geo is None:
            self._geo = self.node('ShaderNodeNewGeometry')
        return self._geo.outputs[out]

    def attr(self, name, out='Fac'):
        n = self.node('ShaderNodeAttribute')
        n.attribute_name = name
        return n.outputs[out]

    def ao(self):
        """Self-occlusion baked into the hp vertex colours (1 = open, 0 = buried)."""
        return self.attr('AO', 'Fac')

    # math -----------------------------------------------------------------
    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        if c is not None:
            self.put(n.inputs[2], c)
        return n.outputs[0]

    def vmath(self, op, a, b=None, scale=None, out=0):
        n = self.node('ShaderNodeVectorMath', operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        if scale is not None:
            self.put(_sock(n.inputs, 'Scale'), scale)
        return n.outputs[out]

    def sep(self, v):
        n = self.node('ShaderNodeSeparateXYZ')
        self.put(n.inputs[0], v)
        return n.outputs

    def comb(self, x, y, z):
        n = self.node('ShaderNodeCombineXYZ')
        self.put(n.inputs[0], x)
        self.put(n.inputs[1], y)
        self.put(n.inputs[2], z)
        return n.outputs[0]

    def maprange(self, v, a, b, c=0.0, d=1.0, clamp=True, interp='LINEAR'):
        n = self.node('ShaderNodeMapRange', clamp=clamp, interpolation_type=interp)
        self.put(n.inputs['Value'], v)
        self.put(n.inputs['From Min'], a)
        self.put(n.inputs['From Max'], b)
        self.put(n.inputs['To Min'], c)
        self.put(n.inputs['To Max'], d)
        return n.outputs['Result']

    def smooth(self, v, a, b):
        return self.maprange(v, a, b, interp='SMOOTHSTEP')

    def mixc(self, fac, a, b, blend='MIX', clamp=True):
        n = self.node('ShaderNodeMix', data_type='RGBA', blend_type=blend, clamp_result=clamp)
        self.put(_sock(n.inputs, 'Factor'), fac)
        self.put(n.inputs[6], a)
        self.put(n.inputs[7], b)
        return n.outputs[2]

    def mixf(self, fac, a, b):
        n = self.node('ShaderNodeMix', data_type='FLOAT', clamp_factor=True)
        self.put(n.inputs[0], fac)
        self.put(n.inputs[2], a)
        self.put(n.inputs[3], b)
        return n.outputs[0]

    def ramp(self, fac, stops, interp='LINEAR'):
        n = self.node('ShaderNodeValToRGB')
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) > 1:
            cr.elements.remove(cr.elements[-1])
        cr.elements[0].position = stops[0][0]
        cr.elements[0].color = (*stops[0][1], 1.0) if len(stops[0][1]) == 3 else stops[0][1]
        for pos, colr in stops[1:]:
            e = cr.elements.new(pos)
            e.color = (*colr, 1.0) if len(colr) == 3 else colr
        self.put(n.inputs[0], fac)
        return n.outputs[0]

    def hsv(self, color, h=0.5, s=1.0, v=1.0):
        n = self.node('ShaderNodeHueSaturation')
        self.put(n.inputs['Hue'], h)
        self.put(n.inputs['Saturation'], s)
        self.put(n.inputs['Value'], v)
        self.put(n.inputs['Color'], color)
        return n.outputs[0]

    # textures -------------------------------------------------------------
    def mapping(self, vec, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        n = self.node('ShaderNodeMapping')
        self.put(n.inputs['Vector'], vec)
        n.inputs['Location'].default_value = loc
        n.inputs['Rotation'].default_value = rot
        n.inputs['Scale'].default_value = scale
        return n.outputs[0]

    def noise(self, vec=None, scale=5.0, detail=4.0, rough=0.5, distortion=0.0, out='Fac',
              lacunarity=2.0, w=None):
        n = self.node('ShaderNodeTexNoise')
        if w is not None:
            n.noise_dimensions = '4D'
            self.put(n.inputs['W'], w)
        self.put(n.inputs['Vector'], vec if vec is not None else self.coord())
        self.put(n.inputs['Scale'], scale)
        self.put(n.inputs['Detail'], detail)
        self.put(n.inputs['Roughness'], rough)
        self.put(n.inputs['Distortion'], distortion)
        self.put(n.inputs['Lacunarity'], lacunarity)
        return n.outputs[out]

    def voronoi(self, vec=None, scale=5.0, feature='F1', metric='EUCLIDEAN', out='Distance',
                randomness=1.0, detail=0.0):
        n = self.node('ShaderNodeTexVoronoi', feature=feature, distance=metric)
        self.put(n.inputs['Vector'], vec if vec is not None else self.coord())
        self.put(n.inputs['Scale'], scale)
        self.put(n.inputs['Randomness'], randomness)
        if 'Detail' in n.inputs:
            self.put(n.inputs['Detail'], detail)
        return n.outputs[out]

    def wave(self, vec=None, scale=5.0, distortion=0.0, detail=2.0, kind='BANDS', axis='X',
             profile='SIN', phase=0.0, detail_scale=1.0, out='Fac'):
        n = self.node('ShaderNodeTexWave', wave_type=kind, wave_profile=profile)
        if kind == 'BANDS':
            n.bands_direction = axis
        else:
            n.rings_direction = axis
        self.put(n.inputs['Vector'], vec if vec is not None else self.coord())
        self.put(n.inputs['Scale'], scale)
        self.put(n.inputs['Distortion'], distortion)
        self.put(n.inputs['Detail'], detail)
        self.put(n.inputs['Detail Scale'], detail_scale)
        self.put(n.inputs['Phase Offset'], phase)
        return n.outputs[out]

    def image(self, img, vec, extension='CLIP', interp='Cubic'):
        n = self.node('ShaderNodeTexImage', extension=extension, interpolation=interp)
        n.image = img
        self.put(n.inputs[0], vec)
        return n.outputs['Color']

    def projector(self, obj):
        """Object-space coords of `obj` remapped from [-1, 1] to [0, 1] for image lookup."""
        return self._madd(self.coord('Object', obj), 0.5, 0.5)

    def _madd(self, v, m, a):
        n = self.node('ShaderNodeVectorMath', operation='MULTIPLY_ADD')
        self.put(n.inputs[0], v)
        n.inputs[1].default_value = (m, m, m)
        n.inputs[2].default_value = (a, a, a)
        return n.outputs[0]

    def bump(self, height, strength=1.0, distance=0.001, normal=None, invert=False):
        n = self.node('ShaderNodeBump', invert=invert)
        self.put(n.inputs['Height'], height)
        self.put(n.inputs['Strength'], strength)
        self.put(n.inputs['Distance'], distance)
        if normal is not None:
            self.put(n.inputs['Normal'], normal)
        return n.outputs[0]

    # output ---------------------------------------------------------------
    def set(self, base=None, rough=None, metal=None, normal=None, emit=None, emit_strength=None,
            spec=None, coat=None, coat_rough=None, transmission=None, ior=None, alpha=None,
            sss=None, sheen=None):
        b = self.bsdf
        self.put(b.inputs['Base Color'], base)
        self.put(b.inputs['Roughness'], rough)
        self.put(b.inputs['Metallic'], metal)
        self.put(b.inputs['Normal'], normal)
        self.put(b.inputs['Emission Color'], emit)
        self.put(b.inputs['Emission Strength'], emit_strength)
        self.put(b.inputs['Specular IOR Level'], spec)
        self.put(b.inputs['Coat Weight'], coat)
        self.put(b.inputs['Coat Roughness'], coat_rough)
        self.put(b.inputs['Transmission Weight'], transmission)
        self.put(b.inputs['IOR'], ior)
        self.put(b.inputs['Alpha'], alpha)
        self.put(b.inputs['Subsurface Weight'], sss)
        self.put(b.inputs['Sheen Weight'], sheen)
        return self.mat


# --------------------------------------------------------------------------
# baking
# --------------------------------------------------------------------------

def bake_vertex_ao(ob, distance, samples=128):
    """Store self-occlusion of the hp mesh in a colour attribute named AO."""
    scn = bpy.context.scene
    me = ob.data
    if 'AO' not in me.color_attributes:
        me.color_attributes.new('AO', 'FLOAT_COLOR', 'POINT')
    me.color_attributes.active_color = me.color_attributes['AO']
    scn.world.light_settings.distance = distance
    scn.cycles.samples = samples
    select_only(ob)
    bpy.ops.object.bake(type='AO', target='VERTEX_COLORS', use_selected_to_active=False,
                        use_clear=True)


def _principled(mat):
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    return None


def _output(mat):
    for n in mat.node_tree.nodes:
        if n.type == 'OUTPUT_MATERIAL':
            return n
    return None


class _ChannelRouter:
    """Temporarily routes one principled input through an Emission shader."""

    def __init__(self, mats):
        self.mats = [m for m in mats if m and _principled(m)]
        self.extra = []

    def route(self, channel):
        self.restore()
        for m in self.mats:
            nt = m.node_tree
            p = _principled(m)
            out = _output(m)
            em = nt.nodes.new('ShaderNodeEmission')
            em.inputs['Strength'].default_value = 1.0
            self.extra.append((m, em))
            if channel == 'AO':
                at = nt.nodes.new('ShaderNodeAttribute')
                at.attribute_name = 'AO'
                self.extra.append((m, at))
                nt.links.new(at.outputs['Fac'], em.inputs['Color'])
            else:
                src = p.inputs[channel]
                if src.is_linked:
                    nt.links.new(src.links[0].from_socket, em.inputs['Color'])
                else:
                    dv = src.default_value
                    em.inputs['Color'].default_value = (tuple(dv) if hasattr(dv, '__len__')
                                                        else (dv, dv, dv, 1.0))
                if channel == 'Emission Color':
                    s = p.inputs['Emission Strength']
                    if not s.is_linked and s.default_value <= 0:
                        em.inputs['Color'].default_value = (0, 0, 0, 1)
                        for l in list(em.inputs['Color'].links):
                            nt.links.remove(l)
            nt.links.new(em.outputs[0], out.inputs['Surface'])

    def restore(self):
        for m, n in self.extra:
            m.node_tree.nodes.remove(n)
        self.extra = []
        for m in self.mats:
            m.node_tree.links.new(_principled(m).outputs[0], _output(m).inputs['Surface'])


def new_image(name, res, non_color, fill=(0, 0, 0, 1)):
    img = bpy.data.images.new(name, res, res, alpha=False, float_buffer=False)
    img.generated_color = fill
    img.colorspace_settings.name = 'Non-Color' if non_color else 'sRGB'
    return img


def bake_down(hp, lp, res, extrusion, max_ray, out_dir, prefix, samples=16):
    """Bake hp (procedural) -> lp (UV atlas). Returns dict of saved image paths."""
    scn = bpy.context.scene
    scn.cycles.samples = samples
    bk = scn.render.bake
    bk.use_selected_to_active = True
    bk.cage_extrusion = extrusion
    bk.max_ray_distance = max_ray
    bk.margin = max(4, res // 128)
    bk.margin_type = 'EXTEND'
    bk.target = 'IMAGE_TEXTURES'
    bk.normal_space = 'TANGENT'

    mat = bpy.data.materials.new(f"M_{prefix}_bake")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    img_node = nt.nodes.new('ShaderNodeTexImage')
    nt.nodes.active = img_node
    set_material(lp, mat)

    passes = [
        ('BaseColor', 'Base Color', False),
        ('Roughness', 'Roughness', True),
        ('Metallic', 'Metallic', True),
        ('AO', 'AO', True),
        ('Emissive', 'Emission Color', False),
        ('Normal', None, True),
    ]
    router = _ChannelRouter(list(hp.data.materials))
    images = {}
    os.makedirs(out_dir, exist_ok=True)
    for key, channel, non_color in passes:
        fill = (0.5, 0.5, 1.0, 1.0) if key == 'Normal' else (0, 0, 0, 1)
        img = new_image(f"T_{prefix}_{key}", res, non_color, fill)
        img_node.image = img
        select_only(hp, lp, active=lp)
        if channel is None:
            router.restore()
            scn.cycles.samples = max(4, samples // 2)
            bpy.ops.object.bake(type='NORMAL')
        else:
            router.route(channel)
            bpy.ops.object.bake(type='EMIT')
        images[key] = img
        print(f"    baked {key}", flush=True)
    router.restore()
    nt.nodes.remove(img_node)
    bpy.data.materials.remove(mat)
    return images


def img_array(img):
    a = np.empty(img.size[0] * img.size[1] * 4, np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(img.size[1], img.size[0], 4)


def save_png(name, arr, path, non_color, alpha=False):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=alpha, float_buffer=False)
    img.colorspace_settings.name = 'Non-Color' if non_color else 'sRGB'
    img.pixels.foreach_set(np.ascontiguousarray(arr, np.float32).ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    # plain save() writes the stored bytes; save_render() would push them through the view transform
    img.save()
    img.filepath = path
    img.source = 'FILE'
    img.reload()
    return img


def finish_textures(images, out_dir, prefix):
    """Write BaseColor, Normal (OpenGL), ORM, Unity MaskMap and (if any glow) Emissive PNGs.

    ORM     R = ambient occlusion, G = roughness, B = metallic   (Unreal, Godot, glTF)
    MaskMap R = metallic, G = occlusion, B = 0, A = smoothness   (Unity HDRP mask / URP metallic)
    """
    base = img_array(images['BaseColor'])
    nrm = img_array(images['Normal'])
    ao = img_array(images['AO'])[..., 0]
    rough = np.clip(img_array(images['Roughness'])[..., 0], 0.02, 1.0)
    metal = np.clip(img_array(images['Metallic'])[..., 0], 0.0, 1.0)
    emis = img_array(images['Emissive'])
    ao = 0.25 + 0.75 * np.clip(ao, 0, 1)  # keep baked AO subtle; engines add SSAO on top
    one = np.ones_like(ao)
    orm = np.stack([ao, rough, metal, one], axis=-1)
    mask = np.stack([metal, ao, np.zeros_like(ao), 1.0 - rough], axis=-1)
    paths = {k: os.path.join(out_dir, f"T_{prefix}_{k}.png") for k in ('BaseColor', 'Normal', 'ORM', 'MaskMap')}
    out = {
        'BaseColor': save_png(f"T_{prefix}_BaseColor", base, paths['BaseColor'], False),
        'Normal': save_png(f"T_{prefix}_Normal", nrm, paths['Normal'], True),
        'ORM': save_png(f"T_{prefix}_ORM", orm, paths['ORM'], True),
    }
    save_png(f"T_{prefix}_MaskMap", mask, paths['MaskMap'], True, alpha=True)
    if emis[..., :3].max() > 0.02:
        paths['Emissive'] = os.path.join(out_dir, f"T_{prefix}_Emissive.png")
        out['Emissive'] = save_png(f"T_{prefix}_Emissive", emis, paths['Emissive'], False)
    for img in images.values():
        bpy.data.images.remove(img)
    return out, paths


def game_material(prefix, tex, emit_strength=1.0):
    """Principled material wired the way glTF/FBX exporters understand."""
    mb = MB(f"M_{prefix}")
    nt = mb.nt
    bc = mb.node('ShaderNodeTexImage', image=tex['BaseColor'])
    orm = mb.node('ShaderNodeTexImage', image=tex['ORM'])
    nm = mb.node('ShaderNodeTexImage', image=tex['Normal'])
    sep = mb.node('ShaderNodeSeparateColor')
    nt.links.new(orm.outputs['Color'], sep.inputs[0])
    nmap = mb.node('ShaderNodeNormalMap')
    nt.links.new(nm.outputs['Color'], nmap.inputs['Color'])
    mb.set(base=bc.outputs['Color'], rough=sep.outputs[1], metal=sep.outputs[2],
           normal=nmap.outputs[0])
    if 'Emissive' in tex:
        em = mb.node('ShaderNodeTexImage', image=tex['Emissive'])
        mb.set(emit=em.outputs['Color'], emit_strength=emit_strength)
    # occlusion goes through the glTF settings group the exporter looks for
    grp = bpy.data.node_groups.get('glTF Material Output')
    if grp is None:
        grp = bpy.data.node_groups.new('glTF Material Output', 'ShaderNodeTree')
        grp.interface.new_socket('Occlusion', in_out='INPUT', socket_type='NodeSocketFloat')
    g = mb.node('ShaderNodeGroup')
    g.node_tree = grp
    nt.links.new(sep.outputs[0], g.inputs['Occlusion'])
    return mb.mat


# --------------------------------------------------------------------------
# lp prep
# --------------------------------------------------------------------------

def _split_long_islands(ob, max_frac=0.3):
    """Cut long, thin UV islands (cords, straps, chains) into shorter pieces.

    One long strip limits how far the packer can scale the whole atlas; chopping it lets
    the big surfaces get the texels. Works in edit mode on the active object.
    """
    from bpy_extras.bmesh_utils import bmesh_linked_uv_islands
    bm = bmesh.from_edit_mesh(ob.data)
    uvl = bm.loops.layers.uv.active
    islands = bmesh_linked_uv_islands(bm, uvl)
    uv_area = 0.0
    for f in bm.faces:
        pts = [l[uvl].uv for l in f.loops]
        for k in range(1, len(pts) - 1):
            uv_area += abs((pts[k] - pts[0]).cross(pts[k + 1] - pts[0])) / 2
    limit = max_frac * math.sqrt(uv_area)
    cut = 0
    for faces in islands:
        uv = np.array([l[uvl].uv[:] for f in faces for l in f.loops])
        c = uv.mean(axis=0)
        cov = np.cov((uv - c).T) if len(uv) > 2 else np.eye(2)
        w, v = np.linalg.eigh(cov)
        axis = v[:, -1]
        proj = (uv - c) @ axis
        length = proj.max() - proj.min()
        area = 0.0
        for f in faces:
            pts = [l[uvl].uv for l in f.loops]
            for k in range(1, len(pts) - 1):
                area += abs((pts[k] - pts[0]).cross(pts[k + 1] - pts[0])) / 2
        ext = uv.max(axis=0) - uv.min(axis=0)
        hollow = area / max(1e-12, ext[0] * ext[1]) < 0.3 and length > 0.4 * limit  # arcs and rings
        if length <= limit and not hollow:
            continue
        n = max(2 if hollow else 1, int(math.ceil(length / limit)))
        step = length / n
        bins = {}
        for f in faces:
            fc = np.array([l[uvl].uv[:] for l in f.loops]).mean(axis=0)
            bins[f] = int((float((fc - c) @ axis) - proj.min()) // step)
        fs = set(faces)
        for f in faces:
            for e in f.edges:
                for g in e.link_faces:
                    if g is not f and g in fs and bins[g] != bins[f]:
                        e.seam = True
                        cut += 1
    bmesh.update_edit_mesh(ob.data)
    return cut


def unwrap_part(ob, island_angle=70.0, margin=0.003):
    """UVs for one lp part, before joining.

    Lathes and sweeps arrive with seams already marked (meridian, hard profile corners,
    tube seams) and are simply relaxed with ABF. Anything else (boxes, remeshed or
    boolean'd solids) gets its cuts from smart project first. Parts may carry an
    `lp_tris` custom property to be decimated to that size beforehand.
    """
    if 'lp_tris' in ob and tri_count(ob) > ob['lp_tris']:
        m = ob.modifiers.new("Decimate", 'DECIMATE')
        m.ratio = ob['lp_tris'] / tri_count(ob)
        ob = apply_modifiers(ob)
    me = ob.data
    has_seams = any(e.use_seam for e in me.edges)
    if not me.uv_layers:
        me.uv_layers.new(name='UVMap')
    select_only(ob)
    with bpy.context.temp_override(active_object=ob, object=ob, selected_objects=[ob],
                                   selected_editable_objects=[ob]):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        if not has_seams:
            bpy.ops.uv.smart_project(angle_limit=math.radians(min(89.0, island_angle)), island_margin=margin,
                                     area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
            bpy.ops.uv.select_all(action='SELECT')
            bpy.ops.uv.seams_from_islands(mark_seams=True, mark_sharp=False)
        bpy.ops.uv.unwrap(method='ANGLE_BASED', fill_holes=True, correct_aspect=True, margin=margin)
        bpy.ops.object.mode_set(mode='OBJECT')
    return ob


def prepare_lowpoly(ob, sharp_angle=60.0, uv_margin=0.003):
    """Triangulate, mark hard edges, even out texel density and pack the joined lp."""
    # bake against the exact triangles the engine will draw, so tangent space matches
    m = ob.modifiers.new("Triangulate", 'TRIANGULATE')
    m.quad_method = 'BEAUTY'
    m.ngon_method = 'BEAUTY'
    apply_modifiers(ob)
    me = ob.data
    me.shade_smooth()
    if sharp_angle < 180:
        me.set_sharp_from_angle(angle=math.radians(sharp_angle))
    select_only(ob)
    with bpy.context.temp_override(active_object=ob, object=ob, selected_objects=[ob],
                                   selected_editable_objects=[ob]):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.uv.average_islands_scale()
        for _ in range(4):  # strips can unroll into arcs, so cut again until nothing is too long
            if not _split_long_islands(ob):
                break
            bpy.ops.uv.unwrap(method='ANGLE_BASED', fill_holes=True, correct_aspect=True, margin=uv_margin)
            bpy.ops.uv.select_all(action='SELECT')
            bpy.ops.uv.average_islands_scale()
        bpy.ops.uv.pack_islands(rotate=True, rotate_method='ANY', margin_method='FRACTION',
                                margin=uv_margin, shape_method='CONCAVE')
        bpy.ops.object.mode_set(mode='OBJECT')
    return ob


# --------------------------------------------------------------------------
# export & preview
# --------------------------------------------------------------------------

def export_item(objs, fbx_path, glb_path):
    os.makedirs(os.path.dirname(fbx_path), exist_ok=True)
    os.makedirs(os.path.dirname(glb_path), exist_ok=True)
    select_only(*objs, active=objs[0])
    bpy.ops.export_scene.fbx(filepath=fbx_path, use_selection=True, apply_unit_scale=True,
                             apply_scale_options='FBX_SCALE_UNITS', axis_forward='-Z', axis_up='Y',
                             mesh_smooth_type='FACE', use_tspace=True, path_mode='RELATIVE',
                             object_types={'MESH', 'EMPTY'}, bake_space_transform=False,
                             add_leaf_bones=False)
    bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB', use_selection=True,
                              export_tangents=True, export_yup=True, export_apply=True)


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def studio(col, center, radius, floor_z=0.0, floor_color=(0.16, 0.15, 0.14)):
    """Seamless backdrop + three soft lights scaled to the subject."""
    R = radius
    # backdrop: floor that curves up into a wall behind the subject
    bm = bmesh.new()
    ys = [-R * 14, R * 2] + [R * 2 + R * 3 * math.sin(k / 12 * math.pi / 2) for k in range(1, 13)]
    zs = [0, 0] + [R * 3 * (1 - math.cos(k / 12 * math.pi / 2)) for k in range(1, 13)]
    ys.append(R * 5)
    zs.append(R * 14)
    rows = []
    for y, z in zip(ys, zs):
        rows.append([bm.verts.new((x, y + center[1], z + floor_z)) for x in (-R * 16, R * 16)])
    for a, b in zip(rows[:-1], rows[1:]):
        bm.faces.new((a[0], a[1], b[1], b[0]))
    floor = mesh_object("Backdrop", bm, col)
    mb = MB("M_Backdrop")
    n = mb.noise(scale=3.0 / R, detail=3)
    mb.set(base=mb.mixc(mb.maprange(n, 0.3, 0.7, 0.0, 1.0), floor_color,
                        tuple(c * 1.15 for c in floor_color)), rough=0.72)
    set_material(floor, mb.mat)

    def area(name, loc, size, energy, color):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.shape = 'DISK'
        ld.size = size
        ld.energy = energy
        ld.color = color
        lo = bpy.data.objects.new(name, ld)
        lo.location = Vector(center) + Vector(loc)
        look_at(lo, center)
        return link(lo, col)

    k = R / 0.1
    area("Key", (-R * 4.0, -R * 3.5, R * 4.5), R * 4, 6.0 * k * k, (1.0, 0.93, 0.84))
    area("Fill", (R * 5.0, -R * 2.5, R * 2.0), R * 6, 1.6 * k * k, (0.85, 0.9, 1.0))
    area("Rim", (R * 1.5, R * 5.0, R * 4.0), R * 3, 5.0 * k * k, (1.0, 0.97, 0.92))
    w = bpy.context.scene.world
    if w.node_tree is None:
        w.use_nodes = True
    bg = w.node_tree.nodes.get('Background') or w.node_tree.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.012, 0.012, 0.014, 1)
    bg.inputs['Strength'].default_value = 1.0
    wo = w.node_tree.nodes.get('World Output') or w.node_tree.nodes.new('ShaderNodeOutputWorld')
    w.node_tree.links.new(bg.outputs[0], wo.inputs[0])
    return floor


def _preview_glass(mat):
    """Copy of a glass material with transparent shadows (render-only; exporters never see it)."""
    pm = mat.copy()
    pm.name = mat.name + "_preview"
    nt = pm.node_tree
    out = _output(pm)
    bsdf = _principled(pm)
    lp = nt.nodes.new('ShaderNodeLightPath')
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    tr.inputs['Color'].default_value = (0.92, 0.95, 0.93, 1)
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(lp.outputs['Is Shadow Ray'], mix.inputs[0])
    nt.links.new(bsdf.outputs[0], mix.inputs[1])
    nt.links.new(tr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    return pm


def render_preview(path, center, radius, res=1400, samples=160, yaw=-35.0, pitch=24.0, lens=85.0, aspect=1.0):
    scn = bpy.context.scene
    swapped = []
    for ob in scn.objects:
        if ob.type == 'MESH' and ob.data.materials and ob.data.materials[0] \
                and ob.data.materials[0].node_tree and _principled(ob.data.materials[0]) \
                and _principled(ob.data.materials[0]).inputs['Transmission Weight'].default_value > 0.5:
            orig = ob.data.materials[0]
            ob.data.materials[0] = _preview_glass(orig)
            swapped.append((ob, orig))
    cam_d = bpy.data.cameras.new("PreviewCam")
    cam_d.lens = lens
    cam_d.sensor_width = 36
    cam = bpy.data.objects.new("PreviewCam", cam_d)
    link(cam)
    fov = 2 * math.atan(18 / lens)
    dist = radius * 1.25 / math.sin(fov / 2)
    yaw_r, pitch_r = math.radians(yaw), math.radians(pitch)
    off = Vector((math.sin(yaw_r) * math.cos(pitch_r), -math.cos(yaw_r) * math.cos(pitch_r),
                  math.sin(pitch_r))) * dist
    cam.location = Vector(center) + off
    look_at(cam, center)
    cam_d.dof.use_dof = True
    cam_d.dof.focus_distance = dist
    cam_d.dof.aperture_fstop = 11.0
    scn.camera = cam
    scn.render.resolution_x = res
    scn.render.resolution_y = int(round(res / aspect))
    scn.render.film_transparent = False
    c = scn.cycles
    c.samples = samples
    c.use_denoising = True
    c.denoiser = 'OPENIMAGEDENOISE'
    c.max_bounces = 16
    c.transmission_bounces = 16
    c.glossy_bounces = 8
    c.transparent_max_bounces = 16
    c.caustics_reflective = False
    c.caustics_refractive = False
    scn.view_settings.view_transform = 'AgX'
    scn.view_settings.look = 'AgX - Medium High Contrast'
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_depth = '8'
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)
    for ob, orig in swapped:
        tmp = ob.data.materials[0]
        ob.data.materials[0] = orig
        bpy.data.materials.remove(tmp)
    bpy.data.objects.remove(cam)
    return None
