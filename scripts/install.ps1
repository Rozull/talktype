<#
.SYNOPSIS
    Instala o atalho do talktype no menu Iniciar.

.DESCRIPTION
    Verifica os pré-requisitos e cria (ou atualiza) o atalho
    "%APPDATA%\Microsoft\Windows\Start Menu\Programs\talktype.lnk", que executa
    ".venv\Scripts\pythonw.exe -m talktype" na pasta do repositório.

    - Exige Windows 10 ou 11 (build 10240 ou mais recente).
    - Exige o uv no PATH e o ambiente .venv criado por "uv sync".
    - Não precisa de administrador e pode ser executado de novo sem efeitos colaterais.
    - Nunca altera a inicialização automática (chave Run); ative-a pela bandeja.
    Compatível com o Windows PowerShell 5.1.

.PARAMETER FakeOsBuild
    Só para testes: usa este número de build no lugar do build real do Windows.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install.ps1
#>
[CmdletBinding()]
param(
    [int]$FakeOsBuild = 0
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

function Fail([string]$Message) {
    Write-Host "ERRO: $Message"
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $scriptDir '..')).Path

# 1. Windows 10 ou 11.
$build = $FakeOsBuild
if ($build -le 0) { $build = [Environment]::OSVersion.Version.Build }
if ($build -lt 10240) {
    Fail "Windows 10 ou 11 é necessário (build $build encontrado)."
}

# 2. uv no PATH.
if (-not (Get-Command 'uv' -ErrorAction SilentlyContinue)) {
    Fail 'uv não encontrado no PATH. Instale-o em https://docs.astral.sh/uv/ e abra um novo terminal.'
}

# 3. O ambiente virtual criado por "uv sync".
$pythonw = Join-Path $root '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonw -PathType Leaf)) {
    Fail ('Ambiente .venv não encontrado em ' + $root + '. Execute `uv sync` primeiro.')
}

# 4. O atalho no menu Iniciar (sobrescrito a cada execução, nunca duplicado).
if (-not $env:APPDATA) { Fail 'A variável APPDATA não está definida.' }
$programs = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
New-Item -ItemType Directory -Force -Path $programs | Out-Null
$link = Join-Path $programs 'talktype.lnk'

$shell = New-Object -ComObject WScript.Shell
try {
    $shortcut = $shell.CreateShortcut($link)
    $shortcut.TargetPath = $pythonw
    $shortcut.Arguments = '-m talktype'
    $shortcut.WorkingDirectory = $root
    $shortcut.Description = 'talktype: ditado local por voz'
    $icon = Join-Path $root 'src\talktype\assets\talktype.ico'
    if (Test-Path -LiteralPath $icon) { $shortcut.IconLocation = $icon }
    $shortcut.Save()
} finally {
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
}

# 5. A chave Run (inicialização automática) nunca é tocada aqui.
Write-Host "Atalho criado: $link"
Write-Host 'Abra "talktype" no menu Iniciar. Na primeira execução o modelo de voz é baixado (0,7 a 1,6 GB).'
exit 0
