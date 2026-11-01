# Installs a Windows autostart shortcut so JARVIS launches in the tray at login.
# Run with:  powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
$pythonw = python -c "import sys; print(sys.executable.replace('python.exe','pythonw.exe'))"
$workdir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$ws = New-Object -ComObject WScript.Shell
$lnkPath = [Environment]::GetFolderPath('Startup') + '\JARVIS.lnk'
$lnk = $ws.CreateShortcut($lnkPath)
$lnk.TargetPath = $pythonw
$lnk.Arguments  = 'run.py --tray'
$lnk.WorkingDirectory = $workdir
$lnk.Description = 'JARVIS voice assistant (system tray)'
$lnk.Save()
Write-Host "Installed: $lnkPath"
Write-Host "JARVIS will start in the tray next time you log in."
Write-Host "To remove: delete that .lnk file."
