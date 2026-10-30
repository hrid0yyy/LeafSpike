@echo off
rem Runs (or continues) the whole experiment queue. Called by the "LeafSpikeExperiments" start-up task.
rem Safe to run by hand too: a second copy exits immediately, finished jobs are skipped.
setlocal
set "HERE=%~dp0"
if exist "%HERE%local_env.bat" call "%HERE%local_env.bat"
if "%LEAFSPIKE_ROOT%"=="" set "LEAFSPIKE_ROOT=%USERPROFILE%\leafspike"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
set "TORCH_HOME=%LEAFSPIKE_ROOT%\cache\torch"
set "HF_HOME=%LEAFSPIKE_ROOT%\cache\huggingface"
if not exist "%LEAFSPIKE_ROOT%\runs" mkdir "%LEAFSPIKE_ROOT%\runs"
pushd "%HERE%.."
echo ===== %DATE% %TIME% run_experiments.bat %* >> "%LEAFSPIKE_ROOT%\runs\console.log"
"%LEAFSPIKE_ROOT%\venv\Scripts\python.exe" run_all.py %* >> "%LEAFSPIKE_ROOT%\runs\console.log" 2>&1
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
