#!/usr/bin/env pwsh
# Local Server 실행 스크립트

Write-Host "=== Local Server 시작 ===" -ForegroundColor Green
Set-Location $PSScriptRoot
python src\run_local_server.py
