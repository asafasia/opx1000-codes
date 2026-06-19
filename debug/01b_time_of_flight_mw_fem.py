
# Single QUA script generated at 2026-06-19 20:03:57.593450
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
        reset_if_phase('q9.resonator')
        wait(27000, 'q9.xy', 'q9.resonator')
        atr_r2 = declare_output_stream(adc_trace=True)
        measure('readout', 'q9.resonator', dual_demod.full("iw1", "iw2", v2), dual_demod.full("iw3", "iw1", v3), adc_stream=atr_r2)
        wait(2500, 'q9.resonator')
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
                            "upconverter_frequency": 7500000000,
                        },
                        "6": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000,
                            "full_scale_power_dbm": -10,
                            "upconverter_frequency": 4400000000,
                        },
                    },
                    "analog_inputs": {
                        "1": {
                            "band": 3,
                            "downconverter_frequency": 7500000000,
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
        "q9.xy": {
            "operations": {
                "x180": "q9.xy.x180.pulse",
                "EF_x180": "q9.xy.EF_x180.pulse",
                "x180_drag": "q9.xy.x180_drag.pulse",
                "x180_cosine": "q9.xy.x180_cosine.pulse",
                "saturation": "q9.xy.saturation.pulse",
                "y180": "q9.xy.y180.pulse",
                "x90": "q9.xy.x90.pulse",
                "-x90": "q9.xy.-x90.pulse",
                "y90": "q9.xy.y90.pulse",
                "-y90": "q9.xy.-y90.pulse",
            },
            "intermediate_frequency": -50540000.0,
            "MWInput": {
                "port": ('con1', 7, 6),
                "upconverter": 1,
            },
        },
        "q9.resonator": {
            "operations": {
                "readout": "q9.resonator.readout.pulse",
            },
            "intermediate_frequency": -28500000.0,
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
        "q9.xy.x180.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q9.xy.x180.wf.I",
                "Q": "q9.xy.x180.wf.Q",
            },
        },
        "q9.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q9.xy.EF_x180.wf.I",
                "Q": "q9.xy.EF_x180.wf.Q",
            },
        },
        "q9.xy.x180_drag.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q9.xy.x180_drag.wf.I",
                "Q": "q9.xy.x180_drag.wf.Q",
            },
        },
        "q9.xy.x180_cosine.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q9.xy.x180_cosine.wf.I",
                "Q": "q9.xy.x180_cosine.wf.Q",
            },
        },
        "q9.xy.saturation.pulse": {
            "operation": "control",
            "length": 50000,
            "waveforms": {
                "I": "q9.xy.saturation.wf.I",
                "Q": "q9.xy.saturation.wf.Q",
            },
        },
        "q9.xy.y180.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q9.xy.y180.wf.I",
                "Q": "q9.xy.y180.wf.Q",
            },
        },
        "q9.xy.x90.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q9.xy.x90.wf.I",
                "Q": "q9.xy.x90.wf.Q",
            },
        },
        "q9.xy.-x90.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q9.xy.-x90.wf.I",
                "Q": "q9.xy.-x90.wf.Q",
            },
        },
        "q9.xy.y90.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q9.xy.y90.wf.I",
                "Q": "q9.xy.y90.wf.Q",
            },
        },
        "q9.xy.-y90.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "q9.xy.-y90.wf.I",
                "Q": "q9.xy.-y90.wf.Q",
            },
        },
        "q9.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 2000,
            "waveforms": {
                "I": "q9.resonator.readout.wf.I",
                "Q": "q9.resonator.readout.wf.Q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "iw1": "q9.resonator.readout.iw1",
                "iw2": "q9.resonator.readout.iw2",
                "iw3": "q9.resonator.readout.iw3",
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
        "q9.xy.x180.wf.I": {
            "type": "constant",
            "sample": 0.04041350198778254,
        },
        "q9.xy.x180.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.EF_x180.wf.I": {
            "type": "constant",
            "sample": 0.013636207468869618,
        },
        "q9.xy.EF_x180.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.x180_drag.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(2.5079648256533794e-06), np.float64(1.0014412798378551e-05), np.float64(3.10616199758289e-05), np.float64(8.632525473532948e-05), np.float64(0.00022214902945278173), np.float64(0.0005344473200299396), np.float64(0.0012057751779922354), np.float64(0.002553855361848493), np.float64(0.005080035983460646), np.float64(0.009491712218863569), np.float64(0.016659304691179308), np.float64(0.02746726393546835), np.float64(0.0425427346070498), np.float64(0.0618998143385166), np.float64(0.0846074185049941), np.float64(0.10863838837872725), np.float64(0.1310431372729689), np.float64(0.1484914748195147)] + [0.15806842311988445] * 2 + [np.float64(0.1484914748195147), np.float64(0.1310431372729689), np.float64(0.10863838837872725), np.float64(0.0846074185049941), np.float64(0.0618998143385166), np.float64(0.0425427346070498), np.float64(0.02746726393546835), np.float64(0.016659304691179308), np.float64(0.009491712218863569), np.float64(0.005080035983460646), np.float64(0.002553855361848493), np.float64(0.0012057751779922354), np.float64(0.0005344473200299396), np.float64(0.00022214902945278173), np.float64(8.632525473532948e-05), np.float64(3.10616199758289e-05), np.float64(1.0014412798378551e-05), np.float64(2.5079648256533794e-06), np.float64(0.0)],
        },
        "q9.xy.x180_drag.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(1.7800666540956407e-06), np.float64(5.537300326153148e-06), np.float64(1.613413518039914e-05), np.float64(4.401791565668063e-05), np.float64(0.00011240139721036766), np.float64(0.00026850915053710383), np.float64(0.0005996979007084093), np.float64(0.0012513334083936415), np.float64(0.0024371470186487472), np.float64(0.004425380760531229), np.float64(0.0074802997430121546), np.float64(0.01174640491001847), np.float64(0.017088130261238715), np.float64(0.02293773204286488), np.float64(0.02823971039423026), np.float64(0.03158108804830414), np.float64(0.03153960424274798), np.float64(0.027174303351692082), np.float64(0.018475511883683134), np.float64(0.006555693341950587), np.float64(-0.006555693341950587), np.float64(-0.018475511883683134), np.float64(-0.027174303351692082), np.float64(-0.03153960424274798), np.float64(-0.03158108804830414), np.float64(-0.02823971039423026), np.float64(-0.02293773204286488), np.float64(-0.017088130261238715), np.float64(-0.01174640491001847), np.float64(-0.0074802997430121546), np.float64(-0.004425380760531229), np.float64(-0.0024371470186487472), np.float64(-0.0012513334083936415), np.float64(-0.0005996979007084093), np.float64(-0.00026850915053710383), np.float64(-0.00011240139721036766), np.float64(-4.401791565668063e-05), np.float64(-1.613413518039914e-05), np.float64(-5.537300326153148e-06), np.float64(-1.7800666540956407e-06)],
        },
        "q9.xy.x180_cosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0006474868681043578), np.float64(0.002573177902642726), np.float64(0.005727198717339505), np.float64(0.010027861829824942), np.float64(0.015363782324520032), np.float64(0.021596762663442206), np.float64(0.02856537192984729), np.float64(0.03608912680417736), np.float64(0.04397316598723385), np.float64(0.05201329700547076), np.float64(0.060001284688802205), np.float64(0.06773024435212678), np.float64(0.07500000000000001), np.float64(0.08162226877976886), np.float64(0.08742553740855505), np.float64(0.09225950427718974), np.float64(0.09599897218294122), np.float64(0.0985470908713026), np.float64(0.0998378654067105), np.float64(0.09983786540671051), np.float64(0.0985470908713026), np.float64(0.09599897218294123), np.float64(0.09225950427718976), np.float64(0.08742553740855508), np.float64(0.08162226877976886), np.float64(0.07499999999999998), np.float64(0.06773024435212681), np.float64(0.060001284688802226), np.float64(0.05201329700547076), np.float64(0.043973165987233886), np.float64(0.0360891268041774), np.float64(0.028565371929847313), np.float64(0.021596762663442223), np.float64(0.015363782324520037), np.float64(0.010027861829824942), np.float64(0.0057271987173395), np.float64(0.0025731779026427204), np.float64(0.0006474868681043578), np.float64(0.0)],
        },
        "q9.xy.x180_cosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 40,
        },
        "q9.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q9.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.y180.wf.I": {
            "type": "constant",
            "sample": 2.4746132925836542e-18,
        },
        "q9.xy.y180.wf.Q": {
            "type": "constant",
            "sample": 0.04041350198778254,
        },
        "q9.xy.x90.wf.I": {
            "type": "constant",
            "sample": 0.02020675099389127,
        },
        "q9.xy.x90.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.-x90.wf.I": {
            "type": "constant",
            "sample": -0.02020675099389127,
        },
        "q9.xy.-x90.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.y90.wf.I": {
            "type": "constant",
            "sample": 1.2373066462918271e-18,
        },
        "q9.xy.y90.wf.Q": {
            "type": "constant",
            "sample": 0.02020675099389127,
        },
        "q9.xy.-y90.wf.I": {
            "type": "constant",
            "sample": -1.2373066462918271e-18,
        },
        "q9.xy.-y90.wf.Q": {
            "type": "constant",
            "sample": -0.02020675099389127,
        },
        "q9.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.8912509381337456,
        },
        "q9.resonator.readout.wf.Q": {
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
        "q9.resonator.readout.iw1": {
            "cosine": [(np.float64(-0.06240599924514471), 3000)],
            "sine": [(np.float64(-0.9980508460285052), 3000)],
        },
        "q9.resonator.readout.iw2": {
            "cosine": [(np.float64(0.9980508460285052), 3000)],
            "sine": [(np.float64(-0.06240599924514471), 3000)],
        },
        "q9.resonator.readout.iw3": {
            "cosine": [(np.float64(-0.9980508460285052), 3000)],
            "sine": [(np.float64(0.06240599924514471), 3000)],
        },
    },
    "mixers": {},
    "oscillators": {},
}

loaded_config = None


