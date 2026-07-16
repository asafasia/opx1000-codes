@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%"

echo ============================================================
echo   REMOVE FRIDGE-HEATING SOFTWARE SAFE MODE
echo ============================================================
echo.
echo Only continue after confirming:
echo   - The dilution refrigerator is cold and stable.
echo   - RF/DC wiring and attenuation are restored and safe.
echo   - External instruments are in the intended state.
echo.

choice /C YN /N /M "Is the refrigerator cold and the hardware safe? [Y/N]: "
if errorlevel 2 goto cancelled

echo.
"C:\Users\owner\miniconda3\envs\opx1000_env\python.exe" -m calibrations.runner outputs enable --confirm-fridge-cold
if errorlevel 1 goto failed

"C:\Users\owner\miniconda3\envs\opx1000_env\python.exe" -m calibrations.runner outputs status
if errorlevel 1 goto failed

echo.
echo ============================================================
echo   SUCCESS: SOFTWARE SAFE MODE IS OFF
echo ============================================================
echo.
echo No pulse or calibration was started.
echo.
pause
exit /b 0

:cancelled
echo.
echo Cancelled. Software safe mode remains unchanged.
echo.
pause
exit /b 1

:failed
echo.
echo ============================================================
echo   SAFE MODE CHANGE FAILED
echo ============================================================
echo.
echo Check the error above. Do not assume outputs are enabled.
echo.
pause
exit /b 1
