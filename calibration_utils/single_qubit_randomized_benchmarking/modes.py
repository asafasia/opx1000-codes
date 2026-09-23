"""Matched-reference IRB and population leakage fits; see docs/rb_modes.md."""
from dataclasses import dataclass, asdict
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from utils.experiment_readout import ground_population


@dataclass
class ModeFitParameters:
    success: bool = False
    mode: str = ""
    error_per_clifford: float = np.nan
    error_per_gate: float = np.nan
    fidelity: float = np.nan
    fidelity_std: float = np.nan
    error_per_gate_std: float = np.nan
    reference_decay: float = np.nan
    interleaved_decay: float = np.nan
    systematic_error_bound: float = np.nan
    error_bound_low: float = np.nan
    error_bound_high: float = np.nan
    leakage_per_clifford: float = np.nan
    seepage_per_clifford: float = np.nan
    leakage_per_clifford_std: float = np.nan
    seepage_per_clifford_std: float = np.nan
    leakage_decay: float = np.nan
    computational_population_asymptote: float = np.nan
    bootstrap_successes: int = 0
    message: str = ""


def decay_curve(depths, offset, amplitude, decay):
    return offset + amplitude * decay ** depths


def fit_decay(depths, values, *, bounded_population=True):
    """Bounded decay; flat/underidentified curves must not produce rates."""
    depths, values = np.asarray(depths, float), np.asarray(values, float)
    if len(np.unique(depths)) < 4 or not np.isfinite(values).all():
        raise ValueError("At least four finite distinct depths are required.")
    if np.ptp(values) < 1e-7:
        raise ValueError("Flat population: decay and asymptote are unidentifiable.")
    best = None
    offsets = (0.5, float(np.clip(values[-1], 0.01, 0.99))) if bounded_population else (values[-1], values.mean())
    bounds = ([0, -1, 0], [1, 1, 1]) if bounded_population else ([-np.inf, -np.inf, 0], [np.inf, np.inf, 1])
    for offset in offsets:
        amplitude = values[0] - offset
        if bounded_population:
            amplitude = np.clip(amplitude, -0.99, 0.99)
        initial = [offset, amplitude, np.exp(-2 / max(depths))]
        try:
            params, cov = curve_fit(decay_curve, depths, values, p0=initial,
                bounds=bounds, maxfev=10000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            score = np.sum((decay_curve(depths, *params) - values)**2)
            if best is None or score < best[0]:
                best = score, params, cov
        except (ValueError, RuntimeError):
            continue
    if best is None:
        raise ValueError("Exponential fit did not converge.")
    _, params, cov = best
    if not np.isfinite(cov).all() or np.linalg.cond(cov) > 1e14:
        raise ValueError("Decay/asymptote poorly identified; extend depth range or acquire more sequences.")
    return params


def interleaved_metrics(reference, interleaved):
    """Magesan et al., PRL 109, 080505 (2012), equations 4 and 5, d=2."""
    p, pc = float(reference), float(interleaved)
    if not 0 < p <= 1 or not 0 <= pc <= 1:
        raise ValueError("Nonphysical fitted RB decay.")
    error = (1 - pc / p) / 2
    bound = min((abs(p - pc / p) + 1 - p) / 2,
                1.5 * (1 - p) / p + 4 * np.sqrt(3 * (1 - p)) / p)
    return error, bound


def fit_mode_data(ds, node):
    mode = node.parameters.mode
    fit = ds.copy()
    if mode == "interleaved":
        if "rb_variant" not in ds.dims or set(ds.rb_variant.values) != {"reference", "interleaved"}:
            raise ValueError("Interleaved RB requires reference and interleaved data acquired together.")
        population = ground_population(ds) if node.parameters.use_state_discrimination else 1 - ds.I
        signals = population.sel(rb_variant=["reference", "interleaved"])
        fit["population"] = population
    else:
        if not all(f"population_{s}" in ds for s in "gef"):
            raise ValueError("Leakage RB requires separate population_g/e/f streams.")
        fit["population"] = ds.population_g
        fit["computational_population"] = ds.population_g + ds.population_e
        signals = fit.computational_population
    fit["averaged_data"] = signals.mean("nb_of_sequences")
    predicted = xr.full_like(fit.averaged_data, np.nan, dtype=float)
    results = {}
    rng = np.random.default_rng(getattr(node.parameters, "fidelity_bootstrap_seed", None))
    samples = int(getattr(node.parameters, "fidelity_bootstrap_samples", 100))
    x = ds.depths.values
    for q in ds.qubit.values:
        result = ModeFitParameters(mode=mode)
        data = signals.sel(qubit=q)
        # Keep each entire depth curve together when resampling sequences.
        values = data.transpose(*(["rb_variant"] if mode == "interleaved" else []), "nb_of_sequences", "depths").values
        def estimate(y):
            if mode == "interleaved":
                pars = [fit_decay(x, row, bounded_population=node.parameters.use_state_discrimination) for row in y]
                error, bound = interleaved_metrics(pars[0][2], pars[1][2])
                return np.array([error]), pars, bound
            pars = fit_decay(x, y)
            offset, _, decay = pars
            return np.array([(1 - offset) * (1 - decay), offset * (1 - decay)]), [pars], 0.
        try:
            metrics, pars, bound = estimate(values.mean(axis=-2))
            if mode == "interleaved":
                result.reference_decay, result.interleaved_decay = pars[0][2], pars[1][2]
                result.error_per_clifford = (1 - result.reference_decay) / 2
                result.error_per_gate = metrics[0]
                result.fidelity = 1 - metrics[0]
                result.systematic_error_bound = bound
                result.error_bound_low, result.error_bound_high = max(0., metrics[0] - bound), min(1., metrics[0] + bound)
                result.success = bool(0 <= metrics[0] <= 1)
                result.message = "Assumes weak, approximately gate-independent noise and negligible leakage."
                if not result.success:
                    result.message = "Negative/unphysical IRB estimate retained; check drift, statistics and model assumptions."
                predicted.loc[{"qubit": q}] = np.array([decay_curve(x, *p) for p in pars])
            else:
                result.leakage_per_clifford, result.seepage_per_clifford = metrics
                result.computational_population_asymptote = pars[0][0]
                result.leakage_decay = pars[0][2]
                result.success = True
                result.message = "Population-model rates per Clifford; assignment errors bias rates. No primitive gate fidelity inferred."
                predicted.loc[{"qubit": q}] = decay_curve(x, *pars[0])
            estimates = []
            count = values.shape[-2]
            if count > 1:
                for _ in range(samples):
                    indices = rng.integers(count, size=count)
                    try:
                        estimates.append(estimate(values[..., indices, :].mean(axis=-2))[0])
                    except (ValueError, RuntimeError):
                        pass
            result.bootstrap_successes = len(estimates)
            if len(estimates) >= max(2, samples // 2):
                std = np.std(estimates, axis=0, ddof=1)
                if mode == "interleaved":
                    result.fidelity_std = result.error_per_gate_std = std[0]
                else:
                    result.leakage_per_clifford_std, result.seepage_per_clifford_std = std
        except (ValueError, RuntimeError) as error:
            result.message = str(error)
        results[str(q)] = result
    fit["predicted_population"] = predicted
    fit["fit_residual"] = fit.averaged_data - predicted
    for key in asdict(next(iter(results.values()))):
        fit[key] = ("qubit", [getattr(results[str(q)], key) for q in ds.qubit.values])
    fit.attrs["rb_mode"] = mode
    return fit, results
