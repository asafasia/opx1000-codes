from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "visa_usb_bridge.py"
SPEC = importlib.util.spec_from_file_location("visa_usb_bridge", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("*idn?", "*IDN?"),
        ("  syst:err?  ", "SYST:ERR?"),
        ("SYSTEM:VERSION?", "SYSTEM:VERSION?"),
    ],
)
def test_normalize_query_accepts_allowlisted_queries(command: str, expected: str) -> None:
    assert BRIDGE.normalize_query(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "*RST",
        "FREQ 5 GHZ",
        "READ?;FREQ 5 GHZ",
        "*IDN?\n*RST",
        "CAL?",
        "",
    ],
)
def test_normalize_query_rejects_writes_and_unapproved_queries(command: str) -> None:
    with pytest.raises(ValueError):
        BRIDGE.normalize_query(command)


def test_normalize_usb_resource_accepts_usb_instrument() -> None:
    resource = "USB0::0x1234::0x5678::SERIAL::INSTR"
    assert BRIDGE.normalize_usb_resource(f"  {resource}  ") == resource


@pytest.mark.parametrize(
    "resource",
    [
        "TCPIP0::192.168.88.249::INSTR",
        "ASRL1::INSTR",
        "GPIB0::1::INSTR",
        "",
        None,
    ],
)
def test_normalize_usb_resource_rejects_non_usb_targets(resource: object) -> None:
    with pytest.raises(ValueError):
        BRIDGE.normalize_usb_resource(resource)


def test_health_endpoint_requires_token_and_reports_read_only_mode() -> None:
    server = BRIDGE.VisaBridgeServer(("127.0.0.1", 0), "127.0.0.1", "test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/health"
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(url, timeout=2)
        assert error.value.code == 401

        request = Request(url, headers={"Authorization": "Bearer test-token"})
        with urlopen(request, timeout=2) as response:
            assert json.loads(response.read()) == {"status": "ok", "mode": "read-only"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
