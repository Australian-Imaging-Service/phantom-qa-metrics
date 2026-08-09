@echo off
REM Launches the PhantomKit GUI in Docker and opens it in your browser.
REM
REM One-time setup (Windows):
REM   1. Make sure Docker Desktop is installed and running.
REM   2. Double-click this file.
REM
REM Your entire user folder is mounted read/write into the container so the
REM GUI's built-in file browser can reach your scan data wherever it lives.
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

docker ps --format "{{.Names}}" | findstr /x "%NAME%" >nul 2>&1
if %errorlevel%==0 (
    echo PhantomKit GUI is already running -^> %URL%
) else (
    docker rm -f %NAME% >nul 2>&1
    echo Starting PhantomKit GUI ^(first run may take a moment to pull the image^)...
    docker run -d --rm --name %NAME% -p %PHANTOMKIT_PORT%:7878 -v "%USERPROFILE%:/hostuser" -e PHANTOMKIT_HOME=/hostuser %PHANTOMKIT_IMAGE% gui >nul
)

echo Waiting for the server to come up...
timeout /t 5 /nobreak >nul

start "" "%URL%"

echo.
echo PhantomKit GUI running at %URL%
echo To stop it later, run: docker stop %NAME%
pause
