"""
kicad2ems-openems: openEMS (3D FDTD) as a kicad2ems solver plugin and FDTD engine, found through the entry points
of this package's pyproject ('kicad2ems.solvers': openems, 'kicad2ems.engines': openems).

    plugin      OpenEMS, the solver 'openems' (alias fdtd): kicad2ems's FDTD model (kicad2ems.fdtd) run by openEMS
    engine      openEMS's CSXCAD / openEMS classes for kicad2ems.fdtd, and the check of that import
    venv        an openEMS installed in another Python environment (the site setting openems_venv)
    converge    --converge: the same model on several mesh levels, compared
"""
__version__ = '261027-r1'
