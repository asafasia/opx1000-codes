"""Offline coverage for the imported RF/LO sweep and saved summaries."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from profiles import Profile
from scripts import resonator_vs_qubit_frequency as sweep_script


def test_rf_lo_retuning_in_generated_configs_preserves_profile():
    documents = Profile("Or_profile", qubit="q3b").load()
    original = deepcopy(documents)
    sweep = sweep_script.SweepSettings()
    frequencies = sweep.frequencies()
    assert len(frequencies) == 100
    for frequency in frequencies[[0, 49, -1]]:
        point = sweep_script.point_settings(documents, "q3b", frequency, sweep)
        calibration = sweep_script.build_calibration(point, sweep_script.Parameters(qubits=["q3b"]))
        machine = calibration.machine
        config = machine.generate_config()
        xy = machine.qubits["q3b"].xy
        assert config["elements"][xy.name]["intermediate_frequency"] == 100_000_000
        output = config["controllers"]["con1"]["fems"][7]["analog_outputs"][7]
        assert output["upconverter_frequency"] == int(frequency) - 100_000_000
        assert machine.qubits["q3b"].resonator.RF_frequency == original["qubits"]["qubits"]["q3b"]["frequencies_hz"]["resonator"]
    assert documents == original


def test_invalid_last_point_fails_before_build_or_acquisition(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid sweep reached machine construction or acquisition")
    monkeypatch.setattr(sweep_script, "build_calibration", unexpected)
    monkeypatch.setattr(sweep_script, "run_resonator", unexpected)
    with pytest.raises(ValueError):
        sweep_script.main(qubit="q3b", sweep=sweep_script.SweepSettings(stop_hz=6_000_000_000))


def test_dry_run_never_connects_or_creates_output(monkeypatch, tmp_path):
    from quam_config import Quam
    def unexpected(*args, **kwargs):
        pytest.fail("Dry run attempted hardware access")
    monkeypatch.setattr(Quam, "connect", unexpected)
    output = tmp_path / "no-output"
    sweep_script.main(qubit="q3b", sweep=sweep_script.SweepSettings(points=2),
                      dry_run=True, output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_loop_checkpoints_completed_scans(monkeypatch, tmp_path, fail):
    seen = []
    def fake_scan(documents, parameters, directory):
        if fail and seen:
            raise RuntimeError("acquisition failed")
        seen.append(documents["qubits"]["qubits"]["q3b"]["frequencies_hz"]["qubit_f01"])
        assert parameters.states == ["g", "e"]
        assert parameters.qubit_operation == "saturation"
        directory.mkdir(parents=True)
        return directory
    monkeypatch.setattr(sweep_script, "run_resonator", fake_scan)
    monkeypatch.setattr(sweep_script, "summarize", lambda *args, **kwargs: None)
    kwargs = dict(qubit="q3b", sweep=sweep_script.SweepSettings(points=2), output_dir=tmp_path)
    if fail:
        with pytest.raises(RuntimeError, match="acquisition failed"):
            sweep_script.main(**kwargs)
    else:
        sweep_script.main(**kwargs)
    manifest = json.loads(next(tmp_path.glob("*/*/manifest.json")).read_text())
    assert manifest["status"] == ("failed" if fail else "complete")
    assert len(manifest["completed_scans"]) == (1 if fail else 2)
    assert seen == ([3_500_000_000] if fail else [3_500_000_000, 5_500_000_000])


def test_saved_shots_and_statistics_use_project_dataset(monkeypatch, tmp_path):
    documents = Profile("Or_profile", qubit="q3b").load()
    parameters = sweep_script.Parameters(qubits=["q3b"], num_shots=3)
    calibration = sweep_script.build_calibration(documents, parameters)
    calibration.create_qua_program()
    raw = xr.Dataset({
        key: (("qubit", "n_runs", "detuning"), np.ones((1, 3, 3)) * value
              + np.arange(3)[None, :, None])
        for key, value in {"Ig": 1, "Qg": 2, "Im": 3, "Qm": 4}.items()
    }, coords={"qubit": ["q3b"], "n_runs": [0, 1, 2], "detuning": [-1e6, 0, 1e6]})
    calibration.results["ds_raw"] = raw
    monkeypatch.setattr(calibration, "run", lambda: SimpleNamespace(interrupted=False))
    monkeypatch.setattr(sweep_script, "build_calibration", lambda *args: calibration)
    output = sweep_script.run_resonator(documents, parameters, tmp_path / "scan")
    assert (output / "raw_shots.npz").is_file()
    assert json.loads((output / "profile.json").read_text()) == documents
    with np.load(output / "results.npz") as saved:
        np.testing.assert_allclose(saved["separation"], np.sqrt(6))
        assert saved["frequency_hz"].shape == (3,)
    for field in ("update_state", "propose_profile_update", "apply_profile_update", "save_raw_data"):
        assert getattr(calibration.options, field) is False


def test_aggregation_selects_separation_and_ignores_nan(tmp_path):
    frequencies = np.array([3_500_000_000, 5_500_000_000])
    resonator = np.array([6_410_000_000, 6_411_000_000, 6_412_000_000])
    scans = []
    for index, separation in enumerate(([np.nan, 2, 1], [3, 7, 4])):
        scan = tmp_path / str(index)
        scan.mkdir()
        np.savez(scan / "results.npz", frequency_hz=resonator, separation=separation,
                 amplitude_volts_g=[1, 2, 3], amplitude_volts_saturated=[2, 3, 4],
                 contrast_volts=[9, 1, 2])
        scans.append(scan)
    best = sweep_script.summarize(tmp_path, frequencies, resonator, scans)
    assert best == {"qubit_frequency_hz": 5_500_000_000,
                    "resonator_frequency_hz": 6_411_000_000, "separation": 7.0}
    with np.load(tmp_path / "summary.npz") as saved:
        np.testing.assert_array_equal(saved["maximum_separation"], [2, 7])
    assert (tmp_path / "summary.png").stat().st_size > 1000
    assert (tmp_path / "resonator_grid_01.png").stat().st_size > 1000
