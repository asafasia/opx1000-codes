@echo off
setlocal
set "LAB_LAUNCHER=%~dp0..\Quantum Coherence Lab\Quantum Coherence Lab.cmd"
if exist "%LAB_LAUNCHER%" goto launch

:missing
echo The separate Quantum Coherence Lab app was not found.
echo Expected: %LAB_LAUNCHER%
pause
exit /b 1

:launch
start "Quantum Coherence Lab" "%LAB_LAUNCHER%"
exit /b 0
