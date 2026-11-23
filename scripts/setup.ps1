# 一次性安装脚本（Windows / PowerShell）
param(
    [switch]$WithModel
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..

python -m pip install -e .
python -m pip install pytest ruff

if ($WithModel) {
    python -m pip install -e ".[model]"
}

Write-Host "完成。下一步："
Write-Host "  python -m pytest              # 跑测试"
Write-Host "  python -m finrag.cli check-corpus"
Write-Host "  python -m finrag.cli eval"
