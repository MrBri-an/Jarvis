@echo off
rem Stops any running JARVIS (python processes running run.py) - nothing else.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" | Where-Object { $_.CommandLine -match 'run\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host ('Stopped JARVIS (pid ' + $_.ProcessId + ')') }"
echo.
echo Done. If nothing was listed, JARVIS wasn't running.
pause
