@echo off
REM Launches the PhantomKit GUI in Docker and opens it in your browser.
REM
REM One-time setup (Windows):
REM   1. Make sure Docker Desktop is installed and running.
REM   2. Double-click this file.
REM
REM Your user folder is mounted read/write into the container so the GUI's
REM file browser can reach your scan data. If your data lives on a
REM different drive (e.g. D:\Data), set PHANTOMKIT_EXTRA_MOUNT=D:\Data
REM before running this script to mount that too.
setlocal

if not defined PHANTOMKIT_IMAGE set PHANTOMKIT_IMAGE=arkiev/phantomkit:latest
if not defined PHANTOMKIT_PORT set PHANTOMKIT_PORT=7878
set NAME=phantomkit-gui
set URL=http://localhost:%PHANTOMKIT_PORT%

docker info >nul 2>&1
if errorlevel 1 (
    echo Docker doesn't seem to be running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo Checking for a newer image (%PHANTOMKIT_IMAGE%)...
docker pull %PHANTOMKIT_IMAGE%
if errorlevel 1 echo Warning: couldn't reach Docker Hub - using whatever's cached locally.

set MOUNTS=-v "%USERPROFILE%:/hostuser"
if defined PHANTOMKIT_EXTRA_MOUNT set MOUNTS=%MOUNTS% -v "%PHANTOMKIT_EXTRA_MOUNT%:/extra"

REM Always restart fresh rather than silently reusing whatever's already
REM running under this name - otherwise a container started from a stale
REM image before an update just keeps serving forever.
docker rm -f %NAME% >nul 2>&1
echo Starting PhantomKit GUI...
docker run -d --rm --name %NAME% -p %PHANTOMKIT_PORT%:7878 %MOUNTS% -e PHANTOMKIT_HOME=/hostuser %PHANTOMKIT_IMAGE% gui >nul

echo Waiting for the server to come up...
timeout /t 5 /nobreak >nul

start "" "%URL%"

echo.
echo PhantomKit GUI running at %URL%
echo To stop it later, run: docker stop %NAME%
pause
