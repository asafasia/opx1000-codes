@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%"

echo ============================================================
echo   PUTTING OPX OUTPUTS IN FRIDGE-HEATING SAFE MODE
echo ============================================================
echo.

"C:\Users\owner\miniconda3\envs\opx1000_env\python.exe" -m calibrations.runner outputs inhibit --reason "dilution refrigerator heating"

if errorlevel 1 goto failed

echo.
echo ============================================================
echo   SUCCESS: SOFTWARE OUTPUT INHIBIT IS ON
echo   SUCCESS: ALL OPEN QUANTUM MACHINES ARE CLOSED
echo ============================================================
echo.
echo Leave this inhibit on for the entire heating period.
echo This does not replace disabling external RF/DC instruments.
echo.
pause
exit /b 0

:failed
echo.
echo ============================================================
echo   SAFETY CHECK FAILED
echo ============================================================
echo.
echo The software inhibit is ON, so new repository calibrations are blocked.
echo Existing hardware output could not be verified safe.
echo DO NOT HEAT until QOP jobs and external RF/DC sources are checked manually.
echo.
pause
exit /b 1
