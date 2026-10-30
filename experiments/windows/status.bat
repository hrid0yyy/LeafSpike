@echo off
rem Shows queue progress and the last lines of the log. Works over SSH.
setlocal
set "HERE=%~dp0"
if exist "%HERE%local_env.bat" call "%HERE%local_env.bat"
if "%LEAFSPIKE_ROOT%"=="" set "LEAFSPIKE_ROOT=%USERPROFILE%\leafspike"
pushd "%HERE%.."
"%LEAFSPIKE_ROOT%\venv\Scripts\python.exe" run_all.py --status
popd
echo.
echo ----- last 25 log lines (%LEAFSPIKE_ROOT%\runs\runner.log) -----
powershell -NoProfile -Command "if (Test-Path '%LEAFSPIKE_ROOT%\runs\runner.log') { Get-Content '%LEAFSPIKE_ROOT%\runs\runner.log' -Tail 25 }"
echo.
schtasks /Query /TN LeafSpikeExperiments /FO LIST 2>nul | findstr /B /C:"Status" /C:"Last Run Time" /C:"Last Result"
