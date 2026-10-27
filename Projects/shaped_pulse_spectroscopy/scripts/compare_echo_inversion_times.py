"""Compare four echo inversion times at Gaussian-convention beta +0.5.

Default is offline QUA/waveform validation. --execute measures exactly four
maps with the current profile and pulse; never updates calibration parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Projects.shaped_pulse_spectroscopy.scripts import optimize_echo_drag_kappa as opt
from shaped_pulse_spectroscopy.lorentzian import build_waveform, drag_quadrature_waveform
from utils.rabi_amplitude import pi_pulse_effective_length_ns

TIMES_NS = (4, 8, 16, 50)
CONVENTION = 'gaussian_positive_derivative_v1'


def parameters(settings, transition_ns):
    p = opt.parameters_for_pair(settings, .5, 0, validation=True)
    p.echo_transition_time_ns = float(transition_ns)
    p.use_three_state_discrimination = False
    p.timeout = 120
    return p


def preflight(settings, machine):
    from qm import QuantumMachinesManager, generate_qua_script
    QuantumMachinesManager.set_capabilities_offline()
    if list(machine.active_qubit_names) != [settings.qubit]:
        raise ValueError('Unexpected active qubits.')
    q = machine.qubits[settings.qubit]
    checks = []
    for transition in TIMES_NS:
        p = parameters(settings, transition)
        cal = opt.EchoLorentzian(parameters=p, machine=machine, qubit=settings.qubit,
            auto_connect=False, options=opt.beta_tools.calibration_options())
        program = cal.create_qua_program()
        config = machine.generate_config()
        if not generate_qua_script(program, config):
            raise ValueError('QUA generation failed.')
        i = np.asarray(build_waveform(p))
        qwave = np.asarray(drag_quadrature_waveform(i, drag_beta=.5,
            anharmonicity_mhz=abs(q.anharmonicity)/1e6, pulse_length_ns=p.lorentzian_length_in_ns))
        peak = float(np.max(abs(i+1j*qwave)))
        if peak >= 1 or cal.namespace['lorentzian_play_duration_cycles'] is not None:
            raise ValueError('Unsafe headroom or unexpected waveform stretching.')
        pulse = config['pulses'][config['elements'][q.xy.name]['operations'][p.operation]]
        for channel, expected in [('I', i), ('Q', qwave)]:
            actual = config['waveforms'][pulse['waveforms'][channel]]['samples']
            np.testing.assert_array_equal(actual, expected)
        checks.append(dict(transition_ns=transition, peak_complex_v=peak,
            peak_i_v=float(np.max(abs(i))), peak_q_v=float(np.max(abs(qwave))),
            waveform_samples=len(i), qua_generated=True,
            waveform_sha256=hashlib.sha256(np.asarray(i+1j*qwave, dtype=np.complex128).tobytes()).hexdigest()))
    reference = q.xy.operations['x180']
    effective = pi_pulse_effective_length_ns(reference)
    return dict(qubit=settings.qubit, profile=settings.profile, beta=.5,
        drag_beta_convention=CONVENTION, transitions=checks,
        rabi_reference=dict(type=type(reference).__name__, amplitude_v=reference.amplitude,
            length_ns=reference.length, effective_length_ns=effective,
            rabi_hz_per_volt=1/(2*reference.amplitude*effective*1e-9)),
        pulse=parameters(settings,16).model_dump(mode='json'))


def plot(output, records):
    if not records:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14,4.5), layout='constrained')
    for r in records:
        label=f"{r['transition_ns']} ns"
        if r['valid']:
            axes[0].errorbar(r['amplitudes'], (np.array(r['center_hz'])-r['weighted_center_hz'])/1000,
                yerr=np.array(r['center_error_hz'])/1000, fmt='.-',label=label)
        axes[1].plot(r['amplitudes'],np.array(r['fwhm_hz'])/1000,'.-',label=label)
    good=[r for r in records if r['valid']]
    axes[2].bar([str(r['transition_ns']) for r in good],[r['center_rms_hz']/1000 for r in good])
    axes[0].set(xlabel='Amplitude factor',ylabel='Centered dip frequency (kHz)',title='Residual center oscillations')
    axes[1].set(xlabel='Amplitude factor',ylabel='Dip FWHM (kHz)',title='Width')
    axes[2].set(xlabel='Inversion time (ns)',ylabel='Dip-center RMS (kHz)',title='Same score for all four maps')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend();axes[1].legend()
    fig.suptitle('q1 echo: beta = +0.5, positive-derivative convention, no chirp')
    fig.savefig(output/'comparison.png',dpi=170)
    plt.close(fig)


def run(settings, machine, output, checks, *, resume=False):
    opt.assert_outputs_allowed()
    snapshot=output/'input_profiles'
    if resume:
        saved=json.loads((output/'settings.json').read_text())
        if saved['settings']!=asdict(settings) or saved['preflight']!=checks:
            raise ValueError('Resume settings or generated waveforms differ from the saved run.')
        before=saved['profile_hashes_before']
        records=json.loads((output/'records.json').read_text())
        completed=[r['transition_ns'] for r in records]
        if completed!=list(TIMES_NS[:len(completed)]):
            raise ValueError('Saved transitions are not an unambiguous completed prefix.')
        if (output/'summary.json').exists():
            shutil.copy2(output/'summary.json',output/'summary_before_resume.json')
    else:
        output.mkdir(parents=True, exist_ok=False)
        before=opt.profile_hashes(settings)
        if not before:raise ValueError('Missing profile.')
        shutil.copytree(ROOT/'profiles'/settings.profile,snapshot/settings.profile)
        opt.beta_tools.write_json(output/'settings.json',dict(settings=asdict(settings),preflight=checks,
            profile_hashes_before=before,transitions_ns=TIMES_NS,score='RMS dip centers about covariance-weighted mean'))
        records=[]
    def snapshot_hashes():
        return {str(Path('profiles')/settings.profile/p.name):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((snapshot/settings.profile).glob('*.json'))}
    if snapshot_hashes()!=before:raise ValueError('Input profile snapshot changed.')
    saver=opt.CalibrationSaver(output_root=output/'raw',profiles_root=snapshot)
    complete=False
    try:
        for transition in TIMES_NS[len(records):]:
            if (snapshot_hashes() if resume else opt.profile_hashes(settings))!=before:
                raise RuntimeError('Profile changed during comparison; stopping before next acquisition.')
            name=f'echo_inversion_{transition:02d}ns_beta_plus0p5'
            print(f'ACQUIRE {name}: 25 amplitudes x 121 detunings x 400 shots',flush=True)
            p=parameters(settings,transition)
            cal=opt.EchoLorentzian(parameters=p,machine=machine,qubit=settings.qubit,
                auto_connect=False,saver=saver,options=opt.beta_tools.calibration_options(acquire=True),name=name)
            try:
                status=cal.run()
            except BaseException:
                if 'ds_raw' in cal.results:
                    np.savez_compressed(output/f'{name}_recovery.npz',
                        **{k:v.values for k,v in cal.results['ds_raw'].variables.items()})
                raise
            if status.interrupted:raise RuntimeError('Acquisition interrupted; partial data retained.')
            record,fitted=opt.score_dataset(cal.results['ds_raw'],settings,.5,0,validation=True)
            scoring=opt.scoring_settings(settings,validation=True)
            record.update(transition_ns=transition,drag_beta_convention=CONVENTION,
                amplitudes=scoring.amplitude_factors.tolist(),raw_directory=str(cal.namespace['calibration_run_directory']))
            records.append(record)
            opt.beta_tools.write_json(output/'records.json',records)
            opt.beta_tools.save_scan(fitted,record,scoring,output/f'{name}.npz')
            opt.beta_tools.save_2d_plot(fitted,record,scoring,output/f'{name}.png')
            plot(output,records)
            print(f"RESULT {transition} ns: RMS={record['center_rms_hz']} Hz, accepted={sum(record['accepted'])}/25",flush=True)
        complete=True
    finally:
        best=opt.beta_tools.select_best(records)
        summary=dict(complete=complete,measurements=len(records),
            all_valid=len(records)==4 and all(r['valid'] for r in records),
            best_measured_transition_ns=best['transition_ns'] if best else None,
            profile_files_unchanged=before==opt.profile_hashes(settings),profiles_updated=False,
            input_profile_snapshot_unchanged=before==snapshot_hashes(),resumed_from_snapshot=resume,
            results=[dict(transition_ns=r['transition_ns'],rms_hz=r['center_rms_hz'],
                median_width_hz=float(np.median(r['fwhm_hz'])),median_contrast=float(np.median(r['contrast'])),
                accepted=sum(r['accepted'])) for r in records])
        opt.beta_tools.write_json(output/'summary.json',summary)
        plot(output,records)
        print(f'OUTPUT {output}',flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Continue an existing output using its frozen profile; skip saved maps.')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    settings=opt.Settings(beta=.5)
    if args.resume:
        if args.output is None:parser.error('--resume requires --output')
        from profiles import Profile
        machine=opt.create_machine(profile=Profile(settings.profile,qubit=settings.qubit,
            root=args.output.resolve()/'input_profiles'))
    else:
        machine=opt.create_machine(profile_name=settings.profile,qubit=settings.qubit)
    checks=preflight(settings,machine)
    if not args.execute:
        print(json.dumps(checks,indent=2),flush=True)
        return
    output=args.output or ROOT/'data/echo_inversion_comparison'/datetime.now().strftime('%Y%m%d_%H%M%S')
    print(json.dumps(run(settings,machine,output.resolve(),checks,resume=args.resume),indent=2),flush=True)


if __name__=='__main__':main()
