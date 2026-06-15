
# Single QUA script generated at 2026-06-15 14:16:35.401095
# QUA library version: 1.3.0a1


from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    v2 = declare(fixed, )
    v3 = declare(fixed, )
    with for_(v1,0,(v1<100),(v1+1)):
        r1 = declare_output_stream()
        save(v1, r1)
        reset_if_phase('q2.resonator')
        wait(27000, 'q2.xy', 'q2.resonator')
        atr_r2 = declare_output_stream(adc_trace=True)
        measure('readout', 'q2.resonator', dual_demod.full("iw1", "iw2", v2), dual_demod.full("iw3", "iw1", v3), adc_stream=atr_r2)
        wait(2500, 'q2.resonator')
        align()
    with stream_processing():
        r1.save("n")
        atr_r2.input1().real().average().save("adcI1")
        atr_r2.input1().image().average().save("adcQ1")
        atr_r2.input1().real().save("adc_single_runI1")
        atr_r2.input1().image().save("adc_single_runQ1")

config = {
    "controllers": {
        "con1": {
            "fems": {
                "7": {
                    "type": "MW",
                    "analog_outputs": {
                        "1": {
                            "band": 3,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000,
                            "full_scale_power_dbm": 1,
                            "upconverter_frequency": 6800000000,
                        },
                        "2": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000,
                            "full_scale_power_dbm": -10,
                            "upconverter_frequency": 4100000000,
                        },
                    },
                    "analog_inputs": {
                        "1": {
                            "band": 3,
                            "downconverter_frequency": 6800000000,
                            "sampling_rate": 1000000000,
                            "shareable": False,
                            "lo_mode": "always_on",
                        },
                    },
                },
            },
        },
    },
    "elements": {
        "q2.xy": {
            "operations": {
                "x180": "q2.xy.x180.pulse",
                "EF_x180": "q2.xy.EF_x180.pulse",
                "x180_drag": "q2.xy.x180_drag.pulse",
                "x180_cosine": "q2.xy.x180_cosine.pulse",
                "saturation": "q2.xy.saturation.pulse",
                "y180": "q2.xy.y180.pulse",
                "x90": "q2.xy.x90.pulse",
                "-x90": "q2.xy.-x90.pulse",
                "y90": "q2.xy.y90.pulse",
                "-y90": "q2.xy.-y90.pulse",
            },
            "intermediate_frequency": -7900000.0,
            "MWInput": {
                "port": ('con1', 7, 2),
                "upconverter": 1,
            },
        },
        "q2.resonator": {
            "operations": {
                "readout": "q2.resonator.readout.pulse",
            },
            "intermediate_frequency": -20000000.0,
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 28,
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
        },
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "q2.xy.x180.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q2.xy.x180.wf.I",
                "Q": "q2.xy.x180.wf.Q",
            },
        },
        "q2.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q2.xy.EF_x180.wf.I",
                "Q": "q2.xy.EF_x180.wf.Q",
            },
        },
        "q2.xy.x180_drag.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q2.xy.x180_drag.wf.I",
                "Q": "q2.xy.x180_drag.wf.Q",
            },
        },
        "q2.xy.x180_cosine.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q2.xy.x180_cosine.wf.I",
                "Q": "q2.xy.x180_cosine.wf.Q",
            },
        },
        "q2.xy.saturation.pulse": {
            "operation": "control",
            "length": 50000,
            "waveforms": {
                "I": "q2.xy.saturation.wf.I",
                "Q": "q2.xy.saturation.wf.Q",
            },
        },
        "q2.xy.y180.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q2.xy.y180.wf.I",
                "Q": "q2.xy.y180.wf.Q",
            },
        },
        "q2.xy.x90.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q2.xy.x90.wf.I",
                "Q": "q2.xy.x90.wf.Q",
            },
        },
        "q2.xy.-x90.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q2.xy.-x90.wf.I",
                "Q": "q2.xy.-x90.wf.Q",
            },
        },
        "q2.xy.y90.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q2.xy.y90.wf.I",
                "Q": "q2.xy.y90.wf.Q",
            },
        },
        "q2.xy.-y90.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q2.xy.-y90.wf.I",
                "Q": "q2.xy.-y90.wf.Q",
            },
        },
        "q2.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 2000,
            "waveforms": {
                "I": "q2.resonator.readout.wf.I",
                "Q": "q2.resonator.readout.wf.Q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "iw1": "q2.resonator.readout.iw1",
                "iw2": "q2.resonator.readout.iw2",
                "iw3": "q2.resonator.readout.iw3",
            },
        },
    },
    "waveforms": {
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "const_wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.xy.x180.wf.I": {
            "type": "constant",
            "sample": 0.041823681167301265,
        },
        "q2.xy.x180.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.EF_x180.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.xy.EF_x180.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.x180_drag.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(1.5742741744950548e-06), np.float64(6.2861453557718335e-06), np.float64(1.94976841962637e-05), np.float64(5.4187210979377e-05), np.float64(0.00013944512952471135), np.float64(0.0003354778364293978), np.float64(0.0007568769320621657), np.float64(0.001603080115171227), np.float64(0.003188788523851882), np.float64(0.005958041063834781), np.float64(0.010457209316536318), np.float64(0.017241471577011457), np.float64(0.02670449271027051), np.float64(0.03885512193887259), np.float64(0.053108908291170695), np.float64(0.06819338430667099), np.float64(0.08225706542750265), np.float64(0.09320955842358894)] + [0.09922110301366054] * 2 + [np.float64(0.09320955842358894), np.float64(0.08225706542750265), np.float64(0.06819338430667099), np.float64(0.053108908291170695), np.float64(0.03885512193887259), np.float64(0.02670449271027051), np.float64(0.017241471577011457), np.float64(0.010457209316536318), np.float64(0.005958041063834781), np.float64(0.003188788523851882), np.float64(0.001603080115171227), np.float64(0.0007568769320621657), np.float64(0.0003354778364293978), np.float64(0.00013944512952471135), np.float64(5.4187210979377e-05), np.float64(1.94976841962637e-05), np.float64(6.2861453557718335e-06), np.float64(1.5742741744950548e-06), np.float64(0.0)],
        },
        "q2.xy.x180_drag.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 40,
        },
        "q2.xy.x180_cosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0006474868681043578), np.float64(0.002573177902642726), np.float64(0.005727198717339505), np.float64(0.010027861829824942), np.float64(0.015363782324520032), np.float64(0.021596762663442206), np.float64(0.02856537192984729), np.float64(0.03608912680417736), np.float64(0.04397316598723385), np.float64(0.05201329700547076), np.float64(0.060001284688802205), np.float64(0.06773024435212678), np.float64(0.07500000000000001), np.float64(0.08162226877976886), np.float64(0.08742553740855505), np.float64(0.09225950427718974), np.float64(0.09599897218294122), np.float64(0.0985470908713026), np.float64(0.0998378654067105), np.float64(0.09983786540671051), np.float64(0.0985470908713026), np.float64(0.09599897218294123), np.float64(0.09225950427718976), np.float64(0.08742553740855508), np.float64(0.08162226877976886), np.float64(0.07499999999999998), np.float64(0.06773024435212681), np.float64(0.060001284688802226), np.float64(0.05201329700547076), np.float64(0.043973165987233886), np.float64(0.0360891268041774), np.float64(0.028565371929847313), np.float64(0.021596762663442223), np.float64(0.015363782324520037), np.float64(0.010027861829824942), np.float64(0.0057271987173395), np.float64(0.0025731779026427204), np.float64(0.0006474868681043578), np.float64(0.0)],
        },
        "q2.xy.x180_cosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 40,
        },
        "q2.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.y180.wf.I": {
            "type": "constant",
            "sample": 2.5609618635047466e-18,
        },
        "q2.xy.y180.wf.Q": {
            "type": "constant",
            "sample": 0.041823681167301265,
        },
        "q2.xy.x90.wf.I": {
            "type": "constant",
            "sample": 0.020911840583650632,
        },
        "q2.xy.x90.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.-x90.wf.I": {
            "type": "constant",
            "sample": -0.020911840583650632,
        },
        "q2.xy.-x90.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.y90.wf.I": {
            "type": "constant",
            "sample": 1.2804809317523733e-18,
        },
        "q2.xy.y90.wf.Q": {
            "type": "constant",
            "sample": 0.020911840583650632,
        },
        "q2.xy.-y90.wf.I": {
            "type": "constant",
            "sample": -1.2804809317523733e-18,
        },
        "q2.xy.-y90.wf.Q": {
            "type": "constant",
            "sample": -0.020911840583650632,
        },
        "q2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.8912509381337456,
        },
        "q2.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [[1, 0]],
        },
    },
    "integration_weights": {
        "q2.resonator.readout.iw1": {
            "cosine": [(np.float64(0.32042550274971876), 3000)],
            "sine": [(np.float64(-0.9472737181974331), 3000)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(np.float64(0.9472737181974331), 3000)],
            "sine": [(np.float64(0.32042550274971876), 3000)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(np.float64(-0.9472737181974331), 3000)],
            "sine": [(np.float64(-0.32042550274971876), 3000)],
        },
    },
    "mixers": {},
    "oscillators": {},
}

loaded_config = {
    "controllers": {
        "con1": {
            "type": "opx1000",
            "fems": {
                "7": {
                    "type": "MW",
                    "analog_outputs": {
                        "1": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 1,
                            "band": 3,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 6800000000.0,
                                },
                            },
                        },
                        "2": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": -10,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4100000000.0,
                                },
                            },
                        },
                    },
                    "analog_inputs": {
                        "1": {
                            "band": 3,
                            "shareable": False,
                            "gain_db": 0,
                            "sampling_rate": 1000000000.0,
                            "downconverter_frequency": 6800000000.0,
                            "lo_mode": "always_on",
                        },
                    },
                },
            },
        },
    },
    "oscillators": {},
    "elements": {
        "q2.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {'readout': 'q2.resonator.readout.pulse'},
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 28,
            "intermediate_frequency": -20000000.0,
        },
        "q2.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {'x180_cosine': 'q2.xy.x180_cosine.pulse', 'y180': 'q2.xy.y180.pulse', 'EF_x180': 'q2.xy.EF_x180.pulse', '-y90': 'q2.xy.-y90.pulse', 'x180_drag': 'q2.xy.x180_drag.pulse', 'x180': 'q2.xy.x180.pulse', '-x90': 'q2.xy.-x90.pulse', 'saturation': 'q2.xy.saturation.pulse', 'x90': 'q2.xy.x90.pulse', 'y90': 'q2.xy.y90.pulse'},
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "MWInput": {
                "port": ('con1', 7, 2),
                "upconverter": 1,
            },
            "intermediate_frequency": -7900000.0,
        },
    },
    "pulses": {
        "q2.xy.x90.pulse": {
            "length": 40,
            "waveforms": {'I': 'q2.xy.x90.wf.I', 'Q': 'q2.xy.x90.wf.Q'},
            "integration_weights": {},
            "operation": "control",
        },
        "q2.xy.-y90.pulse": {
            "length": 40,
            "waveforms": {'I': 'q2.xy.-y90.wf.I', 'Q': 'q2.xy.-y90.wf.Q'},
            "integration_weights": {},
            "operation": "control",
        },
        "q2.xy.-x90.pulse": {
            "length": 40,
            "waveforms": {'I': 'q2.xy.-x90.wf.I', 'Q': 'q2.xy.-x90.wf.Q'},
            "integration_weights": {},
            "operation": "control",
        },
        "q2.resonator.readout.pulse": {
            "length": 2000,
            "waveforms": {'I': 'q2.resonator.readout.wf.I', 'Q': 'q2.resonator.readout.wf.Q'},
            "integration_weights": {'iw1': 'q2.resonator.readout.iw1', 'iw2': 'q2.resonator.readout.iw2', 'iw3': 'q2.resonator.readout.iw3'},
            "operation": "measurement",
        },
        "q2.xy.saturation.pulse": {
            "length": 50000,
            "waveforms": {'I': 'q2.xy.saturation.wf.I', 'Q': 'q2.xy.saturation.wf.Q'},
            "integration_weights": {},
            "operation": "control",
        },
        "q2.xy.x180.pulse": {
            "length": 40,
            "waveforms": {'I': 'q2.xy.x180.wf.I', 'Q': 'q2.xy.x180.wf.Q'},
            "integration_weights": {},
            "operation": "control",
        },
        "q2.xy.y180.pulse": {
            "length": 40,
            "waveforms": {'I': 'q2.xy.y180.wf.I', 'Q': 'q2.xy.y180.wf.Q'},
            "integration_weights": {},
            "operation": "control",
        },
        "q2.xy.x180_cosine.pulse": {
            "length": 40,
            "waveforms": {'I': 'q2.xy.x180_cosine.wf.I', 'Q': 'q2.xy.x180_cosine.wf.Q'},
            "integration_weights": {},
            "operation": "control",
        },
        "q2.xy.x180_drag.pulse": {
            "length": 40,
            "waveforms": {'I': 'q2.xy.x180_drag.wf.I', 'Q': 'q2.xy.x180_drag.wf.Q'},
            "integration_weights": {},
            "operation": "control",
        },
        "q2.xy.y90.pulse": {
            "length": 40,
            "waveforms": {'I': 'q2.xy.y90.wf.I', 'Q': 'q2.xy.y90.wf.Q'},
            "integration_weights": {},
            "operation": "control",
        },
        "const_pulse": {
            "length": 1000,
            "waveforms": {'I': 'const_wf', 'Q': 'zero_wf'},
            "integration_weights": {},
            "operation": "control",
        },
        "q2.xy.EF_x180.pulse": {
            "length": 40,
            "waveforms": {'I': 'q2.xy.EF_x180.wf.I', 'Q': 'q2.xy.EF_x180.wf.Q'},
            "integration_weights": {},
            "operation": "control",
        },
    },
    "waveforms": {
        "q2.xy.x180_drag.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 40,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.y180.wf.Q": {
            "type": "constant",
            "sample": 0.041823681167301265,
        },
        "q2.xy.-y90.wf.I": {
            "type": "constant",
            "sample": -1.2804809317523733e-18,
        },
        "q2.xy.x180_cosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006474868681043578, 0.002573177902642726, 0.005727198717339505, 0.010027861829824942, 0.015363782324520032, 0.021596762663442206, 0.02856537192984729, 0.03608912680417736, 0.04397316598723385, 0.05201329700547076, 0.060001284688802205, 0.06773024435212678, 0.07500000000000001, 0.08162226877976886, 0.08742553740855505, 0.09225950427718974, 0.09599897218294122, 0.0985470908713026, 0.0998378654067105, 0.09983786540671051, 0.0985470908713026, 0.09599897218294123, 0.09225950427718976, 0.08742553740855508, 0.08162226877976886, 0.07499999999999998, 0.06773024435212681, 0.060001284688802226, 0.05201329700547076, 0.043973165987233886, 0.0360891268041774, 0.028565371929847313, 0.021596762663442223, 0.015363782324520037, 0.010027861829824942, 0.0057271987173395, 0.0025731779026427204, 0.0006474868681043578, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180_cosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 40,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "const_wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.xy.EF_x180.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.y180.wf.I": {
            "type": "constant",
            "sample": 2.5609618635047466e-18,
        },
        "q2.xy.x90.wf.I": {
            "type": "constant",
            "sample": 0.020911840583650632,
        },
        "q2.xy.x90.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.x180.wf.I": {
            "type": "constant",
            "sample": 0.041823681167301265,
        },
        "q2.xy.-x90.wf.I": {
            "type": "constant",
            "sample": -0.020911840583650632,
        },
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.xy.x180_drag.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.5742741744950548e-06, 6.2861453557718335e-06, 1.94976841962637e-05, 5.4187210979377e-05, 0.00013944512952471135, 0.0003354778364293978, 0.0007568769320621657, 0.001603080115171227, 0.003188788523851882, 0.005958041063834781, 0.010457209316536318, 0.017241471577011457, 0.02670449271027051, 0.03885512193887259, 0.053108908291170695, 0.06819338430667099, 0.08225706542750265, 0.09320955842358894] + [0.09922110301366054] * 2 + [0.09320955842358894, 0.08225706542750265, 0.06819338430667099, 0.053108908291170695, 0.03885512193887259, 0.02670449271027051, 0.017241471577011457, 0.010457209316536318, 0.005958041063834781, 0.003188788523851882, 0.001603080115171227, 0.0007568769320621657, 0.0003354778364293978, 0.00013944512952471135, 5.4187210979377e-05, 1.94976841962637e-05, 6.2861453557718335e-06, 1.5742741744950548e-06, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.EF_x180.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.xy.-x90.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.y90.wf.Q": {
            "type": "constant",
            "sample": 0.020911840583650632,
        },
        "q2.xy.-y90.wf.Q": {
            "type": "constant",
            "sample": -0.020911840583650632,
        },
        "q2.xy.y90.wf.I": {
            "type": "constant",
            "sample": 1.2804809317523733e-18,
        },
        "q2.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.8912509381337456,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {
        "q2.resonator.readout.iw3": {
            "cosine": [(-0.9472737181974331, 3000)],
            "sine": [(-0.32042550274971876, 3000)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(0.9472737181974331, 3000)],
            "sine": [(0.32042550274971876, 3000)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(0.32042550274971876, 3000)],
            "sine": [(-0.9472737181974331, 3000)],
        },
    },
    "mixers": {},
}


