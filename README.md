# kicad2ems-fdtd

The FDTD solvers of [kicad2ems](https://github.com/lowprocess/kicad2ems): [openEMS](https://openems.de) and
[gfdtd](https://github.com/lowprocess/gfdtd), on one shared FDTD model. This package turns kicad2ems's board into a
rectilinear mesh and an openEMS-style model (the boundaries, the ports and parts, the dielectrics, the field dumps,
the antenna's far-field box), runs one excitation per port with its end criteria, and reads the results back. Only
the engine differs between the two solvers. kicad2ems finds them through the `kicad2ems.solvers` entry points and
the package reaches kicad2ems only through `kicad2ems.api.solver`.

- `plugin.py`: `FDTD`, the solvers' common plugin class, and their shared settings (`MESH`, `RUN`: the GUI's
  FDTD section)
- `mesh.py`, `boundaries.py`, `dielectric.py`, `model.py`: the mesh, the six boundaries, dielectric loss, the
  CSX model (`build_fdtd`)
- `run.py`, `progress.py`: the runs, their end criteria (port watch) and the progress line
- `antenna.py`: antenna-mode runs; `readback.py`: the model as an engine gets it, for the parity checks
- `engines.py`: the engines by name (`'openems'`, `'gfdtd:cupy'`): their `ContinuousStructure` / `openEMS`
  classes. They ship here, with the FDTD layer that drives them
- `openems/`: the solver `openems` (alias `fdtd`): openEMS's engine and its import check (`kicad2ems --check`),
  an openEMS in another environment (`openems_venv`), `--converge`
- `gfdtd.py`: the solver `gfdtd` (alias `gpu-fdtd`) on gfdtd's engine: its backends, its speed estimate, its
  runs at once on the GPU

## Install

    python -m pip install 'kicad2ems[openems]'          # kicad2ems with this package and openEMS's side
    python -m pip install 'kicad2ems[gfdtd]'            # ... with gfdtd
    python -m pip install 'kicad2ems-fdtd[openems,gfdtd] @ git+https://github.com/lowprocess/kicad2ems-fdtd'

openEMS and CSXCAD are not on PyPI: install them from your distribution or build them from source (see
kicad2ems's README). Without them the `openems` solver still loads and `kicad2ems --check` says what is
missing; without gfdtd, `gfdtd` does the same.

## Tests

    python -m pytest

They need kicad2ems's checkout for its example boards and fixtures (`KICAD2EMS_SRC`, default `../kicad2ems`),
and not openEMS: CSXCAD and openEMS are replaced by strict stand-ins (`fake_openems`), and the runs by canned
results. They cover the mesh, the model, the dielectrics, the parts and walls on the grid, the runs' end
criteria and progress, antenna mode, the parity checks, the whole pipeline and `--preview`, `--converge`,
`openems_venv`, and gfdtd's connection (with gfdtd installed).

## Licence

GPL-3.0-or-later, as kicad2ems.
