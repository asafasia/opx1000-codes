import ast
from pathlib import Path

import numpy as np
import xarray as xr
from calibration_utils.thermal_relaxation import (
    effective_temperature_from_population,
    fit_thermal_relaxation,
)

ROOT = Path(__file__).parent.parent


def test_t1_parameters_expose_ground_and_excited_initial_states():
    source = (ROOT / "calibration_utils" / "T1" / "parameters.py").read_text()
    assert 'initial_state: Literal["g", "e"] = "e"' in source


def test_t1_only_plays_x180_for_excited_initial_state():
    source = (ROOT / "calibrations" / "05_T1.py").read_text()
    tree = ast.parse(source)
    conditions = [node for node in ast.walk(tree) if isinstance(node, ast.If)]

    def is_excited_state_check(node):
        if not isinstance(node.test, ast.Compare):
            return False
        left = ast.unparse(node.test.left)
        comparators = [
            ast.literal_eval(comparator)
            for comparator in node.test.comparators
            if isinstance(comparator, ast.Constant)
        ]
        return left == "node.parameters.initial_state" and comparators == ["e"]

    assert any(
        is_excited_state_check(condition)
        and "qubit.xy.play('x180')" in ast.unparse(condition)
        for condition in conditions
    )


def test_t1_multiqubit_plot_saves_one_grid_figure():
    source = (ROOT / "calibrations" / "05_T1.py").read_text()
    plot_data_source = source[
        source.index("    def plot_data") : source.index("    def update_state")
    ]
    assert '"raw_fit": plot_raw_data_with_fit(' in plot_data_source
    assert "plot_per_qubit(" not in plot_data_source
    assert "CalibrationSaver().save_figures(" in plot_data_source


def test_comparison_script_runs_q1_with_mitigation_and_thermal_analysis():
    source = (ROOT / "calibrations" / "t1_initial_state_comparison.py").read_text()
    assert 'DEFAULT_QUBIT = "q1"' in source
    assert 'for initial_state in ("g", "e")' in source
    assert "save_raw_data=False" in source
    assert "save_analysis_result=False" in source
    assert "analyse_data=False" in source
    assert "plot_data=False" in source
    assert "parameters.use_state_discrimination = True" in source
    assert "parameters.use_readout_mitigation = True" in source
    assert "parameters.max_wait_time_in_ns = 150e3" in source
    assert "save_comparison_result(" in source
    assert "saver.save_xarray(" in source
    assert '{"t1_initial_state_comparison": figure}' in source
    assert '"use_readout_mitigation": True' in source
    assert 'dim=xr.IndexVariable("initial_state", ["g", "e"])' in source
    assert "fit_thermal_relaxation(" in source
    assert "effective_temperature_kelvin" in source


def test_joint_thermal_fit_recovers_t1_population_and_temperature():
    idle_time = np.linspace(0.0, 150_000.0, 101)
    expected_t1_ns = 30_000.0
    expected_equilibrium_population = 0.02
    ground_population = expected_equilibrium_population + (
        0.005 - expected_equilibrium_population
    ) * np.exp(-idle_time / expected_t1_ns)
    excited_population = expected_equilibrium_population + (
        0.98 - expected_equilibrium_population
    ) * np.exp(-idle_time / expected_t1_ns)

    def dataset(population):
        result = xr.Dataset(
            {"state": (("qubit", "idle_time"), population[np.newaxis, :])},
            coords={"qubit": ["q1"], "idle_time": idle_time},
        )
        result.state.attrs["readout_mitigated"] = True
        return result

    fit = fit_thermal_relaxation(
        dataset(ground_population),
        dataset(excited_population),
        qubit_name="q1",
        qubit_frequency_hz=4.265e9,
    )

    assert np.isclose(fit.t1_ns, expected_t1_ns, rtol=1e-5)
    assert np.isclose(
        fit.equilibrium_excited_population,
        expected_equilibrium_population,
        rtol=1e-5,
    )
    expected_temperature = effective_temperature_from_population(
        expected_equilibrium_population,
        4.265e9,
    )
    assert np.isclose(fit.effective_temperature_kelvin, expected_temperature)
