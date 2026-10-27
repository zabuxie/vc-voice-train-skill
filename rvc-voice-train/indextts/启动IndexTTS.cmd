@echo off
REM ===================================================================
REM  IndexTTS-2.5 adapter for dsh-plugin-tts  -  one-click launcher
REM  Keep this file ASCII-only and CRLF-terminated.
REM ===================================================================
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ENV_ROOT: 部署根目录。默认 = 本脚本上一级；若你把本脚本放在 indextts/ 子目录，
REM 自动退回两级。也可在运行前 set INDEX_TTS_ENV_ROOT 覆盖。
if defined INDEX_TTS_ENV_ROOT (
  set "ENV_ROOT=%INDEX_TTS_ENV_ROOT%"
) else (
  pushd "%~dp0.." & set "ENV_ROOT=%CD%" & popd
  if not exist "%ENV_ROOT%\index-tts\.venv\Scripts\python.exe" (
    pushd "%~dp0..\.." & set "ENV_ROOT=%CD%" & popd
  )
)
if not defined INDEX_TTS_PORT set "INDEX_TTS_PORT=7880"
set "PY=%ENV_ROOT%\index-tts\.venv\Scripts\python.exe"
set "LOG=%ENV_ROOT%\adapter.log"
set "PORT=%INDEX_TTS_PORT%"
set "TMPCHK=%TEMP%\_indextts_chk.txt"
set "ADAPTER=%~dp0adapter.py"

title IndexTTS-2.5 adapter (port %PORT%)

echo ===================================================================
echo   IndexTTS-2.5 adapter  for  dsh-plugin-tts
echo ===================================================================
echo.

if not exist "%PY%" (
  echo [ERROR] Python venv not found:
  echo         %PY%
  echo         Reinstall the IndexTTS environment first.
  goto :end
)

REM ---------- 1. already running? ----------
set "ALIVE=0"
powershell -NoProfile -Command "try{$c=(Invoke-WebRequest -Uri 'http://127.0.0.1:%PORT%/health' -TimeoutSec 3 -UseBasicParsing).StatusCode}catch{$c=0};Write-Host $c" > "%TMPCHK%" 2>nul
for /f "usebackq delims=" %%a in ("%TMPCHK%") do set "ALIVE=%%a"
del "%TMPCHK%" 2>nul
if "%ALIVE%"=="200" (
  echo [OK] Adapter already running on port %PORT%.
  echo      Health: http://127.0.0.1:%PORT%/health
  goto :end
)

REM ---------- 2. free GPU ----------
echo [1/4] Freeing GPU memory...
set "GPU_USED=0"
for /f "usebackq delims=" %%a in (`powershell -NoProfile -Command "try{(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -replace '[^0-9]',''}catch{'0'}"`) do set "GPU_USED=%%a"
if not defined GPU_USED set "GPU_USED=0"
echo      GPU memory in use: %GPU_USED% MiB
if not "%GPU_USED%"=="0" (
  echo      Stopping common GPU consumers ^(QQ / douyin / music / MC launcher^)...
  taskkill /F /IM QQ.exe >nul 2>&1
  taskkill /F /IM douyin.exe >nul 2>&1
  taskkill /F /IM cloudmusic.exe >nul 2>&1
  taskkill /F /IM "Plain Craft Launcher 2.exe" >nul 2>&1
  timeout /t 3 /nobreak >nul
)

REM ---------- 3. commit charge guard ----------
echo [2/4] Checking commit charge ^(WinError 1455 guard^)...
powershell -NoProfile -Command "$o=Get-CimInstance Win32_OperatingSystem;$f=[math]::Round($o.FreeVirtualMemory/1KB,0);Write-Host ('      commit free: '+$f+' MB');if($f -lt 6000){Write-Host '      [WARN] Low commit charge. Close heavy apps or model load may fail (1455).' -ForegroundColor Yellow}"

REM ---------- 4. launch ----------
echo [3/4] Starting adapter...
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "PYTHONIOENCODING=utf-8:replace"
REM Launch detached via VBScript (wscript): the python process is not a child of
REM this console, so it keeps running after the script exits.
wscript "%ENV_ROOT%\launch_hidden.vbs"

echo [4/4] Waiting for service on port %PORT% ...
set "READY="
for /l %%i in (1,1,30) do (
  if not defined READY (
    timeout /t 1 /nobreak >nul
    set "ST=0"
    powershell -NoProfile -Command "try{$c=(Invoke-WebRequest -Uri 'http://127.0.0.1:%PORT%/health' -TimeoutSec 2 -UseBasicParsing).StatusCode}catch{$c=0};Write-Host $c" > "%TMPCHK%" 2>nul
    for /f "usebackq delims=" %%b in ("%TMPCHK%") do set "ST=%%b"
    if "!ST!"=="200" set "READY=1"
  )
)
del "%TMPCHK%" 2>nul

echo.
if defined READY (
  echo   [OK] Service is up.
) else (
  echo   [WARN] Not reachable yet. Check the log:
  echo          %LOG%
)

echo.
echo ===================================================================
echo   In DSH:  Settings - Plugins - Speech
echo     Provider : Local voice ^(Index-TTS2^)
echo     Address  : http://127.0.0.1:%PORT%
echo     Voice    : ^<name^>    ^(click Refresh first^)
echo.
echo   Log: %LOG%
echo   Model loads on the FIRST synthesis request ^(~30-60 s^).
echo   Use a NEW sentence when testing - the plugin caches by text.
echo ===================================================================

:end
echo.
pause
