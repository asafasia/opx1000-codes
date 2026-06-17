"""v2 placeholder for 01b_time_of_flight_mw_fem."""

from .pending import PendingCalibration


class TimeOfFlightMwFem(PendingCalibration):
    legacy_file = "01b_time_of_flight_mw_fem.py"

    def __init__(self, **kwargs):
        super().__init__(name="01b_time_of_flight_mw_fem", **kwargs)
