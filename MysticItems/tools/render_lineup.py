"""Render all finished items side by side at true scale (reads the .blend files).

    python render_lineup.py [--out ..] [--res 2000] [--samples 160]
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import mi_core as C  # noqa: E402
import mi_items  # noqa: E402

# left-to-right order and floor placement (x, y, yaw in degrees)
LAYOUT = [
    ('Grimoire', (-0.36, 0.06, -18)),
    ('ScryingOrb', (-0.13, 0.10, 0)),
    ('Hourglass', (0.05, 0.12, 0)),
    ('Potion', (0.19, 0.03, 25)),
    ('RuneStone', (0.33, 0.10, -12)),
    ('Amulet', (0.06, -0.10, 30)),
]


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.normpath(os.path.join(HERE, '..')))
    ap.add_argument('--res', type=int, default=2000)
    ap.add_argument('--samples', type=int, default=160)
    args = ap.parse_args(argv)

    C.reset_scene()
    col = C.collection('Lineup')
    keys = {cls.key for cls in mi_items.ITEMS}
    for key, (x, y, yaw) in LAYOUT:
        if key not in keys:
            continue
        path = os.path.join(args.out, 'Blend', f"{key}.blend")
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = [n for n in src.objects if n.startswith('SM_')]
        root = None
        for ob in dst.objects:
            col.objects.link(ob)
            if ob.parent is None:
                root = ob
        root.location = (x, y, 0)
        root.rotation_euler = (0, 0, math.radians(yaw))
    bpy.context.view_layer.update()
    pts = []
    for ob in col.objects:
        pts += [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), 0))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = (lo + hi) / 2
    radius = (hi - lo).length / 2
    C.studio(C.collection('Studio'), tuple(center), radius * 0.55)
    C.render_preview(os.path.join(args.out, 'Previews', 'Lineup.png'), tuple(center), radius * 0.62,
                     res=args.res, samples=args.samples, yaw=-8, pitch=20, lens=70, aspect=2.0)


if __name__ == '__main__':
    main()
