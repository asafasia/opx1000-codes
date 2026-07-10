"""Local, searchable registry for calibration runs and derived results.

Raw arrays and figures remain in ``data/calibrations``.  This package stores
only small, queryable records and paths to those files in SQLite.
"""

from .database import CalibrationResultsDatabase, DEFAULT_DATABASE_PATH

__all__ = ["CalibrationResultsDatabase", "DEFAULT_DATABASE_PATH"]
