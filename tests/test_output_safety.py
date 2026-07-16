import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from calibrations import BaseCalibration, CalibrationOptions
from calibrations.output_safety import (
    OutputsInhibitedError,
    clear_output_inhibit,
    engage_output_inhibit,
    output_inhibit_status,
)
from calibrations.runner import main


class SafetyCalibration(BaseCalibration[SimpleNamespace, object]):
    def create_qua_program(self):
        raise AssertionError("program construction must not occur while outputs are inhibited")


@pytest.fixture
def inhibit_file(tmp_path, monkeypatch):
    path = tmp_path / "outputs_inhibited.json"
    monkeypatch.setenv("OPX_OUTPUT_INHIBIT_FILE", str(path))
    return path


def test_engage_and_clear_output_inhibit(inhibit_file):
    engage_output_inhibit("fridge warm")
    assert output_inhibit_status()["reason"] == "fridge warm"
    assert json.loads(inhibit_file.read_text())["outputs_inhibited"] is True
    assert clear_output_inhibit() is True
    assert output_inhibit_status() is None


def test_real_calibration_is_blocked_before_program_construction(inhibit_file):
    engage_output_inhibit("fridge warm")
    calibration = SafetyCalibration(
        name="safety-test",
        parameters=SimpleNamespace(simulate=False, load_data_id=None),
        machine=object(),
        options=CalibrationOptions(
            save_raw_data=False,
            save_figures=False,
            plot_data=False,
            update_state=False,
            propose_profile_update=False,
        ),
    )

    with pytest.raises(OutputsInhibitedError, match="fridge warm"):
        calibration.run()


def test_simulation_is_not_blocked_by_output_inhibit(inhibit_file):
    engage_output_inhibit("fridge warm")
    calibration = SafetyCalibration(
        name="safety-test",
        parameters=SimpleNamespace(simulate=True, load_data_id=None),
        machine=object(),
    )
    with pytest.raises(AssertionError, match="program construction"):
        calibration.run()


def test_enable_requires_explicit_cold_confirmation(inhibit_file):
    engage_output_inhibit("fridge warm")
    with pytest.raises(SystemExit, match="confirm-fridge-cold"):
        main(["outputs", "enable"])
    assert Path(inhibit_file).is_file()


def test_enable_with_confirmation_clears_latch(inhibit_file):
    engage_output_inhibit("fridge warm")
    assert main(["outputs", "enable", "--confirm-fridge-cold"]) == 0
    assert not inhibit_file.exists()


def test_inhibit_closes_and_verifies_all_qms(inhibit_file):
    qmm = Mock()
    qmm.list_open_qms.return_value = []
    machine = Mock()
    machine.connect.return_value = qmm

    with patch("quam_config.create_machine", return_value=machine):
        assert main(["outputs", "inhibit"]) == 0

    assert inhibit_file.is_file()
    qmm.close_all_qms.assert_called_once_with()
    qmm.list_open_qms.assert_called_once_with()


def test_inhibit_fails_if_qop_still_reports_open_qms(inhibit_file):
    qmm = Mock()
    qmm.list_open_qms.return_value = ["qm-still-open"]
    machine = Mock()
    machine.connect.return_value = qmm

    with patch("quam_config.create_machine", return_value=machine):
        with pytest.raises(RuntimeError, match="qm-still-open"):
            main(["outputs", "inhibit"])

    assert inhibit_file.is_file()
