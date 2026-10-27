# kicad2ems-openems

[openEMS](https://openems.de) (3D FDTD) as a solver plugin of [kicad2ems](https://github.com/lowprocess/kicad2ems).
kicad2ems builds the FDTD model (the mesh, the boundaries, the ports and parts, the runs and their end criteria:
`kicad2ems.fdtd`, shared with gfdtd); this package runs it on openEMS's engine and adds what is openEMS's own:

- `plugin.py`: the solver `openems` (alias `fdtd`), registered through the `kicad2ems.solvers` entry point
- `engine.py`: openEMS as an FDTD engine (the `kicad2ems.engines` entry point), and the check of its import
  (`kicad2ems --check`)
- `venv.py`: an openEMS installed in another Python environment (the site setting `openems_venv`)
- `converge.py`: `--converge`, the same model on several mesh levels, compared

## Install

    python -m pip install 'kicad2ems[openems]'          # kicad2ems with this plugin
    python -m pip install 'kicad2ems-openems @ git+https://github.com/lowprocess/kicad2ems-openems'

openEMS and CSXCAD are not on PyPI: install them from your distribution or build them from source (see
kicad2ems's README). Without them the plugin still loads and `kicad2ems --check` says what is missing.

## Tests

    python -m pytest

They need kicad2ems's checkout for its example boards and fixtures (`KICAD2EMS_SRC`, default `../kicad2ems`),
and not openEMS: they cover the mesh levels of `--converge`, its verdict and a `--preview`, and `openems_venv`.

## Licence

GPL-3.0-or-later, as kicad2ems.
