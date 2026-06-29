from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def test_gef_readout_frequency_stages_profile_shift_update():
    source = (
        REPOSITORY_ROOT
        / "calibrations_v2"
        / "14_gef_readout_frequency_optimization.py"
    ).read_text()

    assert "def profile_updates" in source
    assert "readout.gef_frequency_shift_hz" in source
    assert "resonator.GEF_frequency_shift =" in source


def test_profile_loader_applies_gef_readout_shift():
    source = (REPOSITORY_ROOT / "quam_config" / "populate_quam_lf_mw_fems.py").read_text()
    validation_source = (REPOSITORY_ROOT / "profiles" / "loader.py").read_text()

    assert '"gef_frequency_shift_hz"' in source
    assert "qubit.resonator.GEF_frequency_shift" in source
    assert "readout.gef_frequency_shift_hz must be numeric" in validation_source
