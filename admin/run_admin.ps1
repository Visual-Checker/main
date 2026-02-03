#!/usr/bin/env pwsh
# Admin UI 실행 스크립트

Write-Host "=== Admin UI 시작 ===" -ForegroundColor Green
Set-Location $PSScriptRoot
python src\admin_ui.py
