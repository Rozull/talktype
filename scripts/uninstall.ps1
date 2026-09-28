<#
.SYNOPSIS
    Desinstala o talktype instalado por setup.ps1.

.DESCRIPTION
    Encerra o talktype e remove o atalho do menu Iniciar, a inicialização automática, a
    entrada em "Aplicativos instalados" e a pasta do app.

    - A pasta só é apagada quando contém o marcador ".talktype-install" criado pelo setup.
      Rodar este script num clone de desenvolvimento nunca apaga o repositório.
    - A configuração e o histórico ("%APPDATA%\talktype") ficam, a menos que -Purge seja
      usado. O cache de modelos do Hugging Face e o uv também ficam (outros apps podem usar).
    Compatível com o Windows PowerShell 5.1.

.PARAMETER Purge
    Apaga também a configuração, o histórico e os logs.

.PARAMETER Quiet
    Não espera uma tecla no fim (a janela de "Aplicativos instalados" fecha sozinha).
#>
[CmdletBinding()]
param(
    [switch]$Purge,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$appDir = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..')).Path
$installed = Test-Path -LiteralPath (Join-Path $appDir '.talktype-install')

# 1. Encerra o talktype que roda desta pasta.
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*$appDir*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Milliseconds 500

# 2. O atalho, só se ele aponta para esta pasta.
$link = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\talktype.lnk'
if (Test-Path -LiteralPath $link) {
    $shell = New-Object -ComObject WScript.Shell
    try { $target = $shell.CreateShortcut($link).TargetPath } finally {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
    }
    if ($target -like "$appDir*") { Remove-Item -LiteralPath $link -Force; Write-Host 'Atalho removido.' }
}

# 3. A inicialização automática (valor "talktype" da chave Run), se aponta para esta pasta.
$run = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
$command = (Get-ItemProperty -Path $run -Name 'talktype' -ErrorAction SilentlyContinue)
if ($command -and ([string]$command.talktype) -like "*$appDir*") {
    Remove-ItemProperty -Path $run -Name 'talktype'
    Write-Host 'Inicialização automática removida.'
}

# 4. A entrada em "Aplicativos instalados".
$uninstallKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\talktype'
$entry = Get-ItemProperty -Path $uninstallKey -ErrorAction SilentlyContinue
if ($entry -and ([string]$entry.InstallLocation) -eq $appDir) { Remove-Item -Path $uninstallKey -Recurse -Force }

# 5. Os dados do usuário, só com -Purge.
$home_tt = if ($env:TALKTYPE_HOME) { $env:TALKTYPE_HOME } else { Join-Path $env:APPDATA 'talktype' }
if ($Purge -and (Test-Path -LiteralPath $home_tt)) {
    Remove-Item -LiteralPath $home_tt -Recurse -Force
    Write-Host "Configuração e histórico apagados ($home_tt)."
} elseif (Test-Path -LiteralPath $home_tt) {
    Write-Host "Configuração e histórico mantidos em $home_tt."
}

# 6. A pasta do app, só quando foi criada pelo setup.
if ($installed) {
    Set-Location -LiteralPath ([IO.Path]::GetTempPath())
    Remove-Item -LiteralPath $appDir -Recurse -Force
    Write-Host "talktype desinstalado ($appDir)."
} else {
    Write-Host "A pasta $appDir não foi criada pelo instalador e não foi apagada."
}

if (-not $Quiet) {
    Write-Host ''
    Write-Host 'Pressione uma tecla para fechar.'
    try { [void][Console]::ReadKey($true) } catch { }
}
exit 0
