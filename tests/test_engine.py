"""openEMS's import check (engine.check, kicad2ems --check): this Python, the modules, and on a failure the real
error with a hint (from kicad2ems's tests)."""
import sys
from kicad2ems_openems.engine import check


def test_check_reports_python_and_modules():
    ok, lines = check()
    text = "\n".join(lines)
    assert sys.executable in text and 'numpy' in text
    if not ok:
        assert 'FAILED' in text and 'hints:' in text                # the real error and a hint are shown


def test_check_without_openems(no_csxcad):
    ok, lines = check()
    text = "\n".join(lines)
    assert not ok and 'FAILED  CSXCAD' in text and 'hints:' in text
