@echo off
setlocal
cd /d "%~dp0"

set "OPX1000_PROJECT_ROOT=%~dp0..\opx1000-codes"
if exist "%~dp0..\Fridge Monitor\fridge_monitor.py" set "FRIDGE_MONITOR_ROOT=%~dp0..\Fridge Monitor"
set "QUANTUM_COHERENCE_LAB_LOG_ROOT=%~dp0logs"
set "LAB_APP=%~dp0dist\Quantum Coherence Lab.exe"

if exist "%LAB_APP%" goto launch
echo Quantum Coherence Lab has not been built yet.
echo Run: powershell -ExecutionPolicy Bypass -File "%~dp0build_desktop.ps1"
pause
exit /b 1

:launch
start "Quantum Coherence Lab" "%LAB_APP%"
exit /b 0
