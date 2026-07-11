import ast
from pathlib import Path


ROOT = Path(__file__).parent.parent


def test_t1_parameters_expose_ground_and_excited_initial_states():
    source = (ROOT / "calibration_utils" / "T1" / "parameters.py").read_text()
    assert 'initial_state: Literal["g", "e"] = "e"' in source


def test_t1_only_plays_x180_for_excited_initial_state():
    source = (ROOT / "calibrations_v2" / "05_T1.py").read_text()
    tree = ast.parse(source)
    conditions = [node for node in ast.walk(tree) if isinstance(node, ast.If)]

    def is_excited_state_check(node):
        if not isinstance(node.test, ast.Compare):
            return False
        left = ast.unparse(node.test.left)
        comparators = [ast.literal_eval(comparator) for comparator in node.test.comparators if isinstance(comparator, ast.Constant)]
        return left == "node.parameters.initial_state" and comparators == ["e"]

    assert any(
        is_excited_state_check(condition)
        and "qubit.xy.play('x180')" in ast.unparse(condition)
        for condition in conditions
    )


def test_t1_multiqubit_plot_saves_one_grid_figure():
    source = (ROOT / "calibrations_v2" / "05_T1.py").read_text()
    plot_data_source = source[source.index("    def plot_data") : source.index("    def update_state")]
    assert '"raw_fit": plot_raw_data_with_fit(' in plot_data_source
    assert "plot_per_qubit(" not in plot_data_source
    assert "CalibrationSaver().save_figures(" in plot_data_source


def test_comparison_script_runs_both_preparations_and_uses_a_qubit_grid():
    source = (ROOT / "calibrations_v2" / "t1_initial_state_comparison.py").read_text()
    assert 'DEFAULT_QUBITS = ("q1", "q2", "q3", "q9", "q10")' in source
    assert 'for initial_state in ("g", "e")' in source
    assert "for qubit_name in qubit_names:" in source
    assert "save_raw_data=False" in source
    assert "save_analysis_result=False" in source
    assert "analyse_data=False" in source
    assert "plot_data=False" in source
    assert "parameters.use_state_discrimination = True" in source
    assert "parameters.max_wait_time_in_ns = 150e3" in source
    assert "save_comparison_result(" in source
    assert "saver.save_xarray(" in source
    assert '{"t1_initial_state_comparison": figure}' in source
    assert '"raw_data_saved": True' in source
    assert 'dim=xr.IndexVariable("initial_state", ["g", "e"])' in source
    assert "layout_qubits=compact_layout_qubits(" in source
    assert "selected_rows = {row for _col, row in selected_positions}" in source
    assert "if col not in selected_cols or row not in selected_rows:" in source
    assert "_layout_dataset(layout_qubits)" in source
    assert "axis.set_axis_off()" in source
    assert "QubitGrid" in source
    assert "initial g" in source
    assert "initial e" in source
