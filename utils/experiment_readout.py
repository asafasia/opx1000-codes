"""Shared acquisition and population streams for experiment readout modes."""

from qm.qua import Cast, assign, declare, declare_stream, save
from utils.readout_macro import active_reset_configured, measure_readout, readout_state_configured


def readout_states(parameters):
    states = list(getattr(parameters, "readout_states", ["g", "e"]))
    if states not in (["g", "e"], ["g", "e", "f"]):
        raise ValueError("readout_states must be ['g', 'e'] or ['g', 'e', 'f']")
    return states


def readout_operation(parameters):
    return "readout_GEF" if len(readout_states(parameters)) == 3 else "readout"


def reset_qubit(qubit, parameters):
    """Use the measurement basis for active reset, including the legacy alias."""
    if parameters.reset_type in {"active", "active_gef"} and not parameters.simulate:
        return active_reset_configured(
            qubit, num_states=len(readout_states(parameters)),
            pulse_name=readout_operation(parameters),
            max_attempts=getattr(parameters, "active_reset_max_attempts", 15),
        )
    return qubit.reset("thermal", parameters.simulate)


class PopulationStreams:
    """Apply the same QUA reduction to separate one-hot population streams.

    ``state`` remains P(e) for existing fits; population_g/e/f retain the
    full distribution. Integer state labels are never averaged.
    """

    def __init__(self, states, streams=None):
        self.states = tuple(states)
        self.streams = streams if streams is not None else {s: declare_stream() for s in states}

    def save_shot(self, state):
        indicator = declare(int)
        for index, label in enumerate(self.states):
            assign(indicator, Cast.to_int(state == index))
            save(indicator, self.streams[label])

    def _transform(self, method, *args):
        return PopulationStreams(self.states, {
            label: getattr(stream, method)(*args) for label, stream in self.streams.items()
        })

    def buffer(self, *args):
        return self._transform("buffer", *args)

    def average(self):
        return self._transform("average")

    def map(self, function):
        return self._transform("map", function)

    def _save(self, name, method):
        getattr(self.streams["e"], method)(name)
        suffix = name.removeprefix("state")
        for label, stream in self.streams.items():
            getattr(stream, method)(f"population_{label}{suffix}")

    def save(self, name):
        self._save(name, "save")

    def save_all(self, name):
        self._save(name, "save_all")


def convert_IQ_to_V(da, qubits=None, qubit_pairs=None, IQ_list=("I", "Q"), single_demod=False):
    """Scale IQ with the pulse actually selected for the acquisition."""
    import xarray as xr
    if qubits is None:
        from qualibration_libs.data import convert_IQ_to_V as original
        return original(da, qubit_pairs=qubit_pairs, IQ_list=IQ_list, single_demod=single_demod)
    qubits = list(qubits)
    lengths = xr.DataArray([
        q.resonator.operations[da.attrs.get("readout_operation", getattr(q.resonator, "selected_readout_operation", "readout"))].length
        for q in qubits
    ], coords=[("qubit", [q.name for q in qubits])])
    factor = 2 if single_demod else 1
    return da.assign({key: da[key] * factor * 2**12 / lengths for key in IQ_list})


def selected_readout_frequency(qubit):
    if getattr(qubit.resonator, "selected_readout_operation", "readout") == "readout_GEF":
        return qubit.resonator.readout_gef["frequency_hz"]
    return qubit.resonator.RF_frequency


def ground_population(dataset):
    """Return P(g), preserving legacy binary datasets without counting f as g."""
    if "population_g" in dataset:
        return dataset.population_g
    if "population_f" in dataset:
        return 1 - dataset.state - dataset.population_f
    if len(dataset.attrs.get("readout_states", ["g", "e"])) == 3:
        raise ValueError("GEF ground population requires population_g or population_f")
    return 1 - dataset.state
