"""QM session cleanup without suppressing an interrupted calibration."""

from contextlib import contextmanager

from qualang_tools.multi_user import qm_session as _qm_session


@contextmanager
def qm_session(qmm, config, timeout=100):
    """Close the QM on Ctrl+C, then propagate cancellation to the caller.

    The result fetcher handles the first Ctrl+C by recovering completed data.
    Preserve unhandled interrupts (including a second Ctrl+C during recovery)
    after qualang-tools closes the QM instead of accidentally resuming analysis.
    """
    interrupted = None
    with _qm_session(qmm, config, timeout=timeout) as qm:
        try:
            yield qm
        except KeyboardInterrupt as error:
            interrupted = error
            raise
    if interrupted is not None:
        raise interrupted
