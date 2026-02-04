# Auto-run client/admin UI with a simple menu
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$clientPath = Join-Path $root "client"
$adminPath = Join-Path $root "admin"

Write-Host "Select an option:"
Write-Host "  1) Client"
Write-Host "  2) Admin"
Write-Host "  Q) Quit"

$choice = Read-Host "Enter selection"

switch ($choice.ToUpper()) {
    "1" {
        Start-Process -FilePath "powershell" -ArgumentList @(
            "-NoExit",
            "-Command",
            "conda activate p13_MiniProject; Set-Location `"$clientPath`"; python python_src/client_ui.py"
        )
    }
    "2" {
        Start-Process -FilePath "powershell" -ArgumentList @(
            "-NoExit",
            "-Command",
            "conda activate p13_MiniProject; Set-Location `"$adminPath`"; python src/admin_ui.py"
        )
    }
    "Q" {
        Write-Host "Exiting."
        return
    }
    default {
        Write-Host "Invalid selection."
    }
}
