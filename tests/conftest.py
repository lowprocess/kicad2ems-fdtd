"""The plugin's tests use kicad2ems's fixtures and example boards, from a kicad2ems checkout: KICAD2EMS_SRC,
default ../kicad2ems next to this repository."""
import os, sys

K2E = os.environ.get('KICAD2EMS_SRC') or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'kicad2ems')
if not os.path.exists(os.path.join(K2E, 'tests', 'k2e_fixtures.py')):
    raise RuntimeError(f"no kicad2ems checkout at {K2E}: set KICAD2EMS_SRC")
sys.path.insert(0, os.path.join(K2E, 'tests'))
from k2e_fixtures import *                                          # noqa: E402,F401,F403
