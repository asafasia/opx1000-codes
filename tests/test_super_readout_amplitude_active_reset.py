from pathlib import Path


SOURCE = Path("super_calibrations/readout_amplitude_active_reset.py").read_text()


def test_super_calibration_runs_thermal_then_active_at_each_amplitude():
    assert 'thermal = self._run_inner_iq_blobs("thermal")' in SOURCE
    assert 'self._install_thermal_discriminator(thermal.results["fit_results"])' in SOURCE
    assert 'active = self._run_inner_iq_blobs("active")' in SOURCE
    assert SOURCE.index('thermal = self._run_inner_iq_blobs("thermal")') < SOURCE.index(
        'self._install_thermal_discriminator(thermal.results["fit_results"])'
    )
    assert SOURCE.index(
        'self._install_thermal_discriminator(thermal.results["fit_results"])'
    ) < SOURCE.index('active = self._run_inner_iq_blobs("active")')


def test_active_reset_uses_thermal_threshold_and_restores_original_readout_state():
    assert 'operation.threshold = (' in SOURCE
    assert 'float(fit_result["ge_threshold"]) * operation.length / 2**12' in SOURCE
    assert "operation.amplitude = original.amplitude" in SOURCE
    assert "operation.threshold = original.threshold" in SOURCE
    assert "operation.integration_weights_angle = original.integration_weights_angle" in SOURCE


def test_dashboard_compares_thermal_and_active_fidelity():
    assert '"No active reset"' in SOURCE
    assert '"Active reset"' in SOURCE
    assert "Current amplitude (" in SOURCE
    assert "current amp:" in SOURCE
    assert "best no reset:" in SOURCE
    assert "best active reset:" in SOURCE
    assert 'fidelity_axis.set_ylabel("Assignment fidelity [%]")' in SOURCE


def test_dashboard_plots_iq_standard_deviation_around_points():
    assert '"iq_center_separation"' in SOURCE
    assert '"iq_center_separation_std"' in SOURCE
    assert "def _iq_separation_stats" in SOURCE
    assert "separation_axis.errorbar(" in SOURCE
    assert "uncertainty_axis.plot(" in SOURCE
    assert "yerr=1e3 * trace.iq_center_separation_std.values" in SOURCE
    assert 'separation_axis.set_ylabel("IQ sep. [mV]")' in SOURCE
    assert 'uncertainty_axis.set_ylabel("IQ std. [mV]")' in SOURCE


def test_profile_update_uses_active_reset_optimized_amplitude():
    assert 'best["active"]["readout_amplitude"]' in SOURCE
    assert "pulses.json.pulses.{qubit_name}.readout.amplitude" in SOURCE
    assert "metrics.json.qubits.{qubit_name}.readout.fidelity_percent.active" in SOURCE
    assert "self.profile_updater.stage(" in SOURCE
    assert "CalibrationOptions(update_state=False, propose_profile_update=True)" in SOURCE


def test_super_calibration_reuses_controller_connection_for_inner_runs():
    assert "class _SharedConnectionMachine" in SOURCE
    assert "qmm = self.machine.connect()" in SOURCE
    assert "self._shared_machine = _SharedConnectionMachine(self.machine, qmm)" in SOURCE
    assert "def connect(self):" in SOURCE
    assert "return self._qmm" in SOURCE
    assert "machine=self._shared_machine or self.machine" in SOURCE


def test_super_calibration_suppresses_inner_noise_and_uses_outer_progress():
    assert "contextlib.redirect_stdout(stdout)" in SOURCE
    assert "contextlib.redirect_stderr(stderr)" in SOURCE
    assert "logging.disable(logging.CRITICAL)" in SOURCE
    assert "logger=lambda _: None" in SOURCE
    assert "Super IQ amplitude [" in SOURCE
