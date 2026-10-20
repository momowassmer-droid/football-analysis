@echo off
REM ============================================================
REM  Nacht-Lauf: baut das Produktions-Tracking neu, assembliert
REM  globale IDs und legt einen datierten Schnappschuss ab, den
REM  das Review-Tool als 🌙-Lauf zum A/B-Vergleich anbietet.
REM  Laptop eingesteckt lassen; Fenster darf zugehen wenn fertig.
REM ============================================================
cd /d "%~dp0"
set PYTHONPATH=.
python -m tools.nightly
echo.
echo Fertig. Schnappschuss liegt unter data\output\review_runs\.
echo Im Tester oben bei "Lauf A/B" den 🌙-Eintrag waehlen.
pause
