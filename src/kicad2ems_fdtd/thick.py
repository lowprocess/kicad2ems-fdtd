"""
Copper thickness in the FDTD model (--thickness on).

kicad2ems's board keeps copper as zero-thickness levels between the dielectrics. Here each copper layer gets its
stackup thickness t and everything above it moves up by t, so that every dielectric keeps its own thickness
between the copper, as the line model's cross-sections do (kicad2ems lines/reference.py, _cut_section). The bottom
copper layer grows down (outward); every other layer grows up: the top layer into the air or mask over the board,
an inner layer into the dielectric above it, which fills between its copper.

Each copper level maps onto that layer's two faces:

    dielectric boundaries      the layer's lower face (z_map): a dielectric above an inner layer starts there
    vias, edge walls           from the lower face of their lowest layer to the upper face of their highest
    ports across the stack     from the reference copper's face toward the trace to the trace's face toward it
    gap ports, lumped parts    across the slab, lower face to upper

With --sigma the copper is a solid metal body whose two faces are conducting sheets: each face is one side of a
conductor t / 2 thick (sigma / 4 and 2 t in openEMS's symmetric sheet model give exactly that impedance:
k / (sigma tanh(k t / 2)), k = (1 + j) / delta), so the faces carry the skin-effect loss where their fields are.

The plan is plain data (dict(t, up)), kept in the mesh dict as mesh['thick'], so that runs in other processes get
it with the mesh.
"""
EPS = 1e-7                                                          # mm: a z on a copper level


def plan(m):
    """dict(t={layer: mm}, up={layer: bool}) of board m's copper, or None when no layer has a thickness."""
    t = {lay: float(m.cu_t.get(lay, 0.0)) for lay in m.cu_z}
    if not any(v > 0 for v in t.values()):
        return None
    order = sorted(m.cu_z, key=lambda lay: m.cu_z[lay])
    return dict(t=t, up={lay: lay != order[0] for lay in order})


def z_map(m, th, z):
    """z (board model, mm) in the thick model: moved up by the layers below it that grow up."""
    return round(z + sum(th['t'][lay] for lay, zc in m.cu_z.items() if th['up'][lay] and zc < z - EPS), 9)


def faces(m, th, lay):
    """(lower, upper) z of copper layer `lay` in the thick model."""
    lo, t = z_map(m, th, m.cu_z[lay]), th['t'][lay]
    return (lo, round(lo + t, 9)) if th['up'][lay] else (round(lo - t, 9), lo)  # on the mesh's lines exactly


def layer_at(m, z):
    """The copper layer on level z, or None."""
    return next((lay for lay, zc in m.cu_z.items() if abs(zc - z) < EPS), None)


def toward(m, th, z, other):
    """z of a copper level mapped to its layer's face toward `other` (z_map off a copper level)."""
    lay = layer_at(m, z)
    if lay is None:
        return z_map(m, th, z)
    lo, up = faces(m, th, lay)
    return up if other > z else lo


def span(m, th, z0, z1):
    """An interval between levels (z0 <= z1) mapped to cover the copper at its ends."""
    lo0, hi1 = sorted((z0, z1))
    a, b = layer_at(m, lo0), layer_at(m, hi1)
    return (faces(m, th, a)[0] if a else z_map(m, th, lo0)), (faces(m, th, b)[1] if b else z_map(m, th, hi1))


def slab(m, th, z):
    """The copper slab (lower, upper) on level z (a flat part or gap port there), or (z_map, z_map) off copper."""
    lay = layer_at(m, z)
    return faces(m, th, lay) if lay else (z_map(m, th, z),) * 2


def all_faces(m, th):
    """{layer: (lower, upper)}."""
    return {lay: faces(m, th, lay) for lay in m.cu_z}


def top(m, th):
    """The board's top in the thick model: the highest copper's upper face, or the mapped thickness."""
    return max([z_map(m, th, m.thickness)] + [f[1] for f in all_faces(m, th).values()])


def bottom(m, th):
    return min([0.0] + [f[0] for f in all_faces(m, th).values()])
