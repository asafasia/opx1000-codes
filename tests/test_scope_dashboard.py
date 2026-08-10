import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "scope" / "dashboard.py"
SPEC = importlib.util.spec_from_file_location("scope_dashboard", MODULE_PATH)
DASHBOARD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = DASHBOARD
SPEC.loader.exec_module(DASHBOARD)


def test_parse_ieee_block():
    parsed = DASHBOARD.parse_ieee_block(b"#15\x00\x7f\x80\xfe\xff\n")
    np.testing.assert_array_equal(parsed, [0, 127, 128, 254, 255])


@pytest.mark.parametrize("raw", [b"", b"hello", b"#0", b"#15\x00"])
def test_parse_ieee_block_rejects_invalid_or_incomplete_data(raw):
    with pytest.raises(ValueError):
        DASHBOARD.parse_ieee_block(raw)


def test_scope_resource_name():
    connection = DASHBOARD.ScopeConnection("192.0.2.10")
    assert connection.resource_name == "TCPIP::192.0.2.10::4000::SOCKET"
