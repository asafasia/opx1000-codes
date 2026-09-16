import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from sweeps.ramsey_stability import (
    RamseyStabilityRun, StabilitySettings, main, persistent_ramsey_class,
    render_report, render_point_figure, save_dataset,
)
from sweeps.stability_analysis import overlapping_allan, point_diagnostics, summarize


class Clock:
    now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def make_runner(tmp_path, *, duration=10, interval=0, max_points=None, failure=None,
                rejected=False, point_seconds=2, render=False):
    clock = Clock()
    starts = []

    def factory(directory):
        # Each completed point must already be durable before starting the next.
        if starts:
            saved = json.loads((tmp_path / "summary.json").read_text())
            assert saved["attempts"] == len(starts)
        starts.append(clock.now)

        def run():
            clock.now += point_seconds
            if failure is not None and len(starts) == 2:
                raise failure
            calibration.results["fit_results"] = {"q3": {
                "decay": 20e-6, "decay_error": 1e-6,
                "freq_offset": 1000, "success": not rejected,
            }}

        calibration = SimpleNamespace(results={}, run=run)
        return calibration

    runner = RamseyStabilityRun(
        StabilitySettings("q3", duration_seconds=duration, interval_seconds=interval,
                          max_points=max_points), tmp_path, factory,
        time_fn=clock.time, sleep_fn=clock.sleep, utc_fn=lambda: "2026-09-07T00:00:00+00:00",
        render_reports=render,
    )
    return runner, clock, starts


def test_duration_incremental_saving_and_units(tmp_path):
    runner, clock, starts = make_runner(tmp_path, duration=5)
    result = runner.run()
    assert starts == [0, 2, 4]
    assert clock.now == 6  # In-flight point finishes across the deadline.
    assert result["mean_t2_us"] == pytest.approx(20)
    assert [r["elapsed_seconds"] for r in runner.rows] == [1, 3, 5]
    assert runner.rows[0]["t2_error_us"] == pytest.approx(1)
    assert len((tmp_path / "points.jsonl").read_text().splitlines()) == 3
    assert len((tmp_path / "points.csv").read_text().splitlines()) == 4
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "completed"


def test_fixed_cadence_does_not_oversleep_deadline(tmp_path):
    runner, clock, starts = make_runner(tmp_path, duration=10, interval=6)
    runner.run()
    assert starts == [0, 6]
    assert clock.now == 10


def test_overruns_skip_slots(tmp_path):
    runner, _, starts = make_runner(tmp_path, duration=25, interval=5, point_seconds=7)
    runner.run()
    assert starts == [0, 10, 20]


def test_max_points_avoids_final_sleep(tmp_path):
    runner, clock, starts = make_runner(tmp_path, duration=100, interval=20, max_points=2)
    runner.run()
    assert starts == [0, 20]
    assert clock.now == 22


@pytest.mark.parametrize("failure,status", [(KeyboardInterrupt(), "interrupted"),
                                             (RuntimeError("lost connection"), "failed")])
def test_partial_results_survive_failure(tmp_path, failure, status):
    runner, _, _ = make_runner(tmp_path, failure=failure)
    with pytest.raises(type(failure)):
        runner.run()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == status
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert (tmp_path / "points/000002/point.json").exists()
    assert "NaN" not in (tmp_path / "points.jsonl").read_text()


def test_existing_run_cannot_be_overwritten(tmp_path):
    runner, _, _ = make_runner(tmp_path, max_points=1)
    runner.run()
    original = (tmp_path / "points.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        runner.run()
    assert original == (tmp_path / "points.jsonl").read_bytes()


def test_repeated_rejections_stop(tmp_path):
    runner, _, starts = make_runner(tmp_path, duration=100, rejected=True)
    with pytest.raises(RuntimeError, match="15 consecutive rejected"):
        runner.run()
    assert len(starts) == 15
    assert summarize(runner.rows)["mean_t2_us"] is None


def row(value, point=0, accepted=True):
    return dict(t2_us=value, t2_error_us=.1, accepted=accepted,
                elapsed_seconds=float(point), jump_candidate=False)


def test_known_statistics_and_allan():
    rows = [row(value, i) for i, value in enumerate([1, 2, 3, 4])]
    summary = summarize(rows)
    assert summary["variance_t2_us2"] == pytest.approx(5/3)
    assert summary["drift_us_per_hour"] == pytest.approx(3600)
    assert summary["allan"]["deviation_us"] == pytest.approx([np.sqrt(.5)])
    allan = overlapping_allan(np.arange(16), np.arange(16))
    assert allan["tau_seconds"] == [1, 2, 4]
    assert allan["deviation_us"] == pytest.approx(np.array([1, 2, 4])/np.sqrt(2))
    assert allan["pair_count"] == [15, 13, 9]


def test_allan_rejects_gaps_and_irregular_spacing():
    assert "Irregular" in overlapping_allan(np.array([0, 1, 2, 4]), np.ones(4))["reason"]
    assert "missing" in overlapping_allan(np.arange(4), np.array([1, 1, np.nan, 1]))["reason"]
    assert overlapping_allan(np.arange(8), np.ones(8))["deviation_us"] == [0, 0]


def test_jump_requires_history_respects_uncertainty_and_failure_gaps():
    history = [row(20, i) for i in range(8)]
    candidate = point_diagnostics(history, row(25, 8), 20, 5)
    assert candidate["jump_candidate"]
    assert candidate["delta_t2_us"] == 5
    assert candidate["rolling_variance_us2"] == pytest.approx(np.var([20]*8+[25], ddof=1))
    noisy = row(25, 8)
    noisy["t2_error_us"] = 10
    assert not point_diagnostics(history, noisy, 20, 5)["jump_candidate"]
    assert not point_diagnostics(history[:3], row(25), 20, 5)["jump_candidate"]
    assert point_diagnostics(history+[row(20, accepted=False)], row(25), 20, 5)["delta_t2_us"] is None


@pytest.mark.parametrize("kwargs", [dict(duration_seconds=0), dict(duration_seconds=float("nan")),
    dict(interval_seconds=-1), dict(max_points=0), dict(rolling_window=4), dict(jump_sigma=float("inf"))])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        StabilitySettings("q3", **kwargs)


def test_npz_retains_dimensions_coordinates_and_units(tmp_path):
    dataset = xr.Dataset({"state": (("qubit", "idle_time"), [[.1, .2]])},
                         coords={"qubit": ["q3"], "idle_time": [16, 32]})
    dataset.idle_time.attrs["units"] = "ns"
    save_dataset(tmp_path, "raw", dataset)
    with np.load(tmp_path / "raw.npz", allow_pickle=False) as arrays:
        np.testing.assert_equal(arrays["state"], [[.1, .2]])
        assert arrays["qubit"].tolist() == ["q3"]
    metadata = json.loads((tmp_path / "raw.json").read_text())
    assert metadata["variables"]["state"]["dims"] == ["qubit", "idle_time"]
    assert metadata["variables"]["idle_time"]["attrs"]["units"] == "ns"


def test_raw_is_saved_before_analysis_failure(tmp_path, monkeypatch):
    from calibration_utils.ramsey import Parameters
    from calibrations.base import BaseCalibration, CalibrationOptions
    import calibrations.core.base as base
    monkeypatch.setattr(base, "assert_outputs_allowed", lambda: None)
    PersistentRamsey = persistent_ramsey_class()
    assert issubclass(PersistentRamsey, BaseCalibration)
    calibration = PersistentRamsey(
        parameters=Parameters(), machine=SimpleNamespace(), point_directory=tmp_path,
        options=CalibrationOptions(save_figures=False, plot_data=False, update_state=False,
                                   propose_profile_update=False, apply_profile_update=False,
                                   report_runtime_estimate=False),
    )
    monkeypatch.setattr(calibration, "create_qua_program", lambda: None)
    monkeypatch.setattr(calibration, "execute_qua_program", lambda: calibration.results.update(
        ds_raw=xr.Dataset({"I": ("idle_time", [.1, .2])}, coords={"idle_time": [16, 32]})))

    def fail():
        assert (tmp_path / "raw.npz").exists()
        raise ValueError("bad fit")

    monkeypatch.setattr(calibration, "analyse_data", fail)
    with pytest.raises(ValueError, match="bad fit"):
        calibration.run()
    assert (tmp_path / "raw.npz").exists()


def test_synthetic_ramsey_full_lifecycle_and_no_state_updates(tmp_path, monkeypatch):
    from calibration_utils.ramsey import Parameters
    from calibrations.base import CalibrationOptions
    import calibrations.core.base as base
    monkeypatch.setattr(base, "assert_outputs_allowed", lambda: None)
    delay = np.linspace(16, 20000, 101)
    signal = .5 + .35 * np.exp(-delay / 8000) * np.cos(2*np.pi*.0002*delay+.1)
    dataset = xr.Dataset({"state": (("qubit", "detuning_signs", "idle_time"),
                                      np.array([[signal, signal]]))},
                         coords={"qubit": ["q3"], "detuning_signs": [-1, 1], "idle_time": delay})
    calibration = persistent_ramsey_class()(
        parameters=Parameters(use_state_discrimination=True, frequency_detuning_in_mhz=.2),
        machine=SimpleNamespace(), point_directory=tmp_path,
        options=CalibrationOptions(save_figures=False, plot_data=False, update_state=False,
                                   propose_profile_update=False, apply_profile_update=False,
                                   report_runtime_estimate=False),
    )
    monkeypatch.setattr(calibration, "create_qua_program", lambda: None)
    monkeypatch.setattr(calibration, "execute_qua_program", lambda: calibration.results.update(ds_raw=dataset))
    monkeypatch.setattr(calibration, "update_state", lambda: pytest.fail("state update attempted"))
    monkeypatch.setattr(calibration, "propose_profile_update", lambda: pytest.fail("profile update attempted"))
    calibration.run()
    fit = calibration.results["fit_results"]["q3"]
    assert fit["success"]
    assert fit["decay"] == pytest.approx(8e-6, rel=.01)
    assert (tmp_path / "fit.npz").exists()
    assert json.loads((tmp_path / "fit_results.json").read_text())["q3"]["decay"] == fit["decay"]


def test_quality_gate_keeps_invalid_estimates_out_of_statistics(tmp_path):
    runner, _, _ = make_runner(tmp_path)
    tmp_path.joinpath("point").mkdir()
    fit = {"decay": 20e-6, "decay_error": 15e-6, "freq_offset": 1, "success": True}
    calibration = SimpleNamespace(results={"fit_results": {"q3": fit}},
                                  namespace={"measurement_started_monotonic": 2,
                                             "measurement_finished_monotonic": 4})
    runner._record(1, tmp_path / "point", 0, 10, "2026-09-07", calibration, None)
    assert runner.rows[0]["elapsed_seconds"] == 3  # Fitting time is excluded.
    assert not runner.rows[0]["accepted"]
    assert runner.rows[0]["t2_us"] == pytest.approx(20)  # Preserve rejected estimates.
    assert summarize(runner.rows)["mean_t2_us"] is None


def test_reports_render_without_gui_even_for_empty_or_failed_data(tmp_path):
    runner, _, _ = make_runner(tmp_path, duration=3, render=True)
    runner.run()
    assert (tmp_path / "stability.png").stat().st_size > 1000
    assert 'http-equiv="refresh"' in (tmp_path / "index.html").read_text(encoding="utf-8")
    runner.rows[0]["accepted"] = False
    runner._publish()


def test_dry_run_never_constructs_machine(monkeypatch, capsys):
    import quam_config

    def forbidden(**kwargs):
        pytest.fail("dry-run attempted machine construction")

    monkeypatch.setattr(quam_config, "create_machine", forbidden)
    assert main(["--qubit", "q3", "--duration-hours", "24", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["stability"]["duration_seconds"] == 86400
    assert result["stability"]["max_consecutive_failures"] == 15
    assert result["hardware_execution"] is False


def test_report_updates_each_experiment_and_figures_every_15(tmp_path, monkeypatch):
    import sweeps.ramsey_stability as stability

    published = []
    plotted = []

    def report(directory, rows, summary):
        published.append((len(rows), summary["status"]))
        (directory / "stability.png").write_bytes(f"chart at {len(rows)}".encode())

    monkeypatch.setattr(stability, "render_report", report)
    monkeypatch.setattr(stability, "render_point_figure",
                        lambda directory, row, results: plotted.append(row["point"]))
    runner, _, _ = make_runner(tmp_path, duration=100, max_points=31, render=True)
    runner.run()
    assert published == [(0, "running")] + [
        (point, "running") for point in range(1, 32)
    ] + [(31, "completed")]
    assert plotted == [15, 30]
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "stability_000015.png", "stability_000030.png"
    ]
    assert (tmp_path / "figures/stability_000015.png").read_bytes() == b"chart at 15"
    assert (tmp_path / "figures/stability_000030.png").read_bytes() == b"chart at 30"


def test_rejected_points_save_figures_even_before_15(tmp_path):
    runner, _, _ = make_runner(tmp_path, duration=100, max_points=5, rejected=True, render=True)
    runner.run()
    for point in range(1, 6):
        assert (tmp_path / f"points/{point:06d}/ramsey.png").stat().st_size > 1000
    assert not (tmp_path / "figures").exists()


@pytest.mark.parametrize("fit_kind", ["missing", "valid", "malformed"])
def test_ramsey_figure_saves_trace_with_optional_fit_without_gui(tmp_path, fit_kind):
    import matplotlib.pyplot as plt

    delay = np.linspace(16, 3000, 50)
    signal = .5 + .3 * np.exp(-delay / 2000) * np.cos(2*np.pi*.001*delay)
    dataset = xr.Dataset(
        {"state": (("qubit", "detuning_signs", "idle_time"), [[signal, signal]])},
        coords={"qubit": ["q3"], "detuning_signs": [-1, 1], "idle_time": delay},
    )
    results = {"ds_raw": dataset}
    if fit_kind == "valid":
        results["ds_fit"] = xr.Dataset(
            {"fit": (("qubit", "detuning_signs", "fit_vals"),
                     [[[.3, .001, 0, .5, .0005]] * 2])},
            coords={"qubit": ["q3"], "detuning_signs": [-1, 1],
                    "fit_vals": ["a", "f", "phi", "offset", "decay"]},
        )
    elif fit_kind == "malformed":
        results["ds_fit"] = xr.Dataset()
    existing = plt.get_fignums()
    render_point_figure(tmp_path, dict(point=1, qubit="q3", accepted=False,
                                     rejection_reason="relative_uncertainty_exceeds_limit"), results)
    assert (tmp_path / "ramsey.png").stat().st_size > 1000
    assert plt.get_fignums() == existing
