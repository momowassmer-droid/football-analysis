@echo off
REM ============================================================
REM  Overnight auto-labeling of the 3 test clips with the strong
REM  (x-size) football model. Fully hands-off:
REM   - restarts the python job automatically if it ever crashes
REM     (the job itself resumes from the last finished frame),
REM   - stops by itself once data\autolabel\ALL_DONE appears.
REM  Leave this window open. Keep the laptop plugged in.
REM  In the morning: check data\autolabel\*\autolabel.log
REM ============================================================
cd /d "%~dp0"
set PYTHONPATH=.

:loop
if exist "data\autolabel\ALL_DONE" goto done
echo [%date% %time%] (re)starting auto-label batch...
python -m tools.autolabel --batch
echo [%date% %time%] python exited. Checking completion...
timeout /t 5 /nobreak >nul
goto loop

:done
echo.
echo ============================================================
echo  ALL DONE. Auto-labeling finished. You can close this window.
echo ============================================================
pause
