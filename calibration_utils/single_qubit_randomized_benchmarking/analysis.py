import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_decay_exp


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    error_per_clifford: float
    error_per_gate: float
    fidelity: float
    fidelity_std: float
    error_per_gate_std: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
            s_fidelity = f"\tSingle qubit gate fidelity: {100 * fit_results[q]['fidelity']:.3f}"
            if not np.isnan(fit_results[q]["fidelity_std"]):
                s_fidelity += f" +/- {100 * fit_results[q]['fidelity_std']:.3f}"
            s_fidelity += " %\n"
        else:
            s_qubit += " FAIL!\n"
            s_fidelity = "\tSingle qubit gate fidelity: unavailable because the decay fit failed\n"
        log_callable(s_qubit + s_fidelity)
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    ds_fit = ds
    if node.parameters.use_state_discrimination:
        ds_fit["population"] = 1 - ds.state
        ds_fit["averaged_data"] = ds_fit.population.mean(dim="nb_of_sequences")
    else:
        ds_fit["averaged_data"] = 1 - ds.I.mean(dim="nb_of_sequences")
    # Fit the exponential decay. qualibration_libs can raise after its
    # per-qubit fitter returns None, so preserve the acquired data on failure.
    try:
        fit_data = fit_decay_exp(ds_fit["averaged_data"], "depths")
    except Exception as error:
        node.log(f"RB decay fit failed; preserving raw population data: {error}")
        template = ds_fit["averaged_data"].isel(depths=0, drop=True)
        fit_data = xr.concat(
            [xr.full_like(template, np.nan, dtype=float) for _ in range(3)],
            dim=xr.IndexVariable("fit_vals", ["a", "offset", "decay"]),
        )

    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])

    # Extract the relevant fitted parameters
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Extract the decay rate
    alpha = np.exp(fit.fit_data.sel(fit_vals="decay"))
    # average_gate_per_clifford = 45/24 = 1.875
    average_gate_per_clifford = (1 * 3 + 9 * 2 + 1 * 4 + 2 * 3 + 4 * 2 + 2 * 3) / 24
    # EPC from here: https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html#Step-5:-Fit-the-results
    fit["error_per_clifford"] = (1 - alpha) * (1 - 1 / 2)
    fit["error_per_gate"] = fit["error_per_clifford"] / average_gate_per_clifford
    fit["fidelity"] = 1 - fit["error_per_gate"]
    fidelity_std = _estimate_fidelity_std(fit, node, average_gate_per_clifford)
    fit["fidelity_std"] = fidelity_std
    fit["error_per_gate_std"] = fidelity_std
    # Assess whether the fit was successful or not
    nan_success = np.isnan(fit.error_per_gate)
    rb_success = (0 < fit.error_per_gate) & (fit.error_per_gate < 1)
    success_criteria = ~nan_success & rb_success
    fit = fit.assign({"success": success_criteria})

    # Save fitting results
    fit_results = {
        q: FitParameters(
            error_per_clifford=float(fit.sel(qubit=q)["error_per_clifford"]),
            error_per_gate=float(fit.sel(qubit=q)["error_per_gate"]),
            fidelity=float(fit.sel(qubit=q)["fidelity"]),
            fidelity_std=float(fit.sel(qubit=q)["fidelity_std"]),
            error_per_gate_std=float(fit.sel(qubit=q)["error_per_gate_std"]),
            success=bool(fit.sel(qubit=q).success),
        )
        for q in fit.qubit.values
    }
    node.outcomes = {q: "successful" if fit_results[q].success else "fail" for q in fit.qubit.values}

    return fit, fit_results


def _estimate_fidelity_std(
    fit: xr.Dataset,
    node: QualibrationNode,
    average_gate_per_clifford: float,
) -> xr.DataArray:
    """Estimate RB fidelity uncertainty by bootstrapping random sequences."""
    bootstrap_samples = int(getattr(node.parameters, "fidelity_bootstrap_samples", 100))
    if bootstrap_samples <= 1 or "nb_of_sequences" not in fit.dims:
        return xr.full_like(fit.error_per_gate, np.nan, dtype=float)

    sequence_count = int(fit.sizes["nb_of_sequences"])
    if sequence_count <= 1:
        return xr.full_like(fit.error_per_gate, np.nan, dtype=float)

    if hasattr(fit, "population"):
        sequence_data = fit.population
    elif hasattr(fit, "state"):
        sequence_data = 1 - fit.state
    elif hasattr(fit, "I"):
        sequence_data = 1 - fit.I
    else:
        return xr.full_like(fit.error_per_gate, np.nan, dtype=float)

    rng = np.random.default_rng(getattr(node.parameters, "fidelity_bootstrap_seed", None))
    bootstrap_fidelities = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, sequence_count, size=sequence_count)
        averaged = sequence_data.isel(nb_of_sequences=indices).mean(dim="nb_of_sequences")
        try:
            bootstrap_fit_data = fit_decay_exp(averaged, "depths")
        except Exception:
            continue
        bootstrap_alpha = np.exp(bootstrap_fit_data.sel(fit_vals="decay"))
        bootstrap_error_per_clifford = (1 - bootstrap_alpha) * (1 - 1 / 2)
        bootstrap_error_per_gate = bootstrap_error_per_clifford / average_gate_per_clifford
        bootstrap_fidelities.append(1 - bootstrap_error_per_gate)

    if len(bootstrap_fidelities) <= 1:
        return xr.full_like(fit.error_per_gate, np.nan, dtype=float)

    samples = xr.concat(
        bootstrap_fidelities,
        dim=xr.IndexVariable("bootstrap_sample", np.arange(len(bootstrap_fidelities))),
    )
    return samples.std(dim="bootstrap_sample", ddof=1)
