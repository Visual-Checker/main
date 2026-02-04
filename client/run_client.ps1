#!/usr/bin/env pwsh
# Client UI 실행 스크립트

Write-Host "=== Client UI 시작 ===" -ForegroundColor Green
Set-Location $PSScriptRoot
python python_src\client_ui.py
