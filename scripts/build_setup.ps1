<#
.SYNOPSIS
    Gera dist\talktype-setup.exe, o instalador pequeno que se manda para outras pessoas.

.DESCRIPTION
    Empacota scripts\setup.ps1 com o IExpress (vem com o Windows). O .exe abre uma janela
    de console, roda o setup, que baixa a última versão publicada no GitHub, e mostra o
    progresso. Não precisa de administrador.

    O .exe não é assinado: na primeira vez o Windows SmartScreen pode mostrar "O Windows
    protegeu o computador". Nesse caso, clique em "Mais informações" e "Executar assim mesmo".

.PARAMETER SetupArgs
    Só para testes: argumentos extras para setup.ps1 (por exemplo "-Source C:\x.zip").

.PARAMETER Output
    O .exe gerado. O padrão é dist\talktype-setup.exe.
#>
[CmdletBinding()]
param(
    [string]$SetupArgs = '',
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..')).Path
if (-not $Output) { $Output = Join-Path $root 'dist\talktype-setup.exe' }
$Output = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null

# IExpress packs the files of one folder; stage setup.ps1 alone.
$stage = Join-Path ([IO.Path]::GetTempPath()) ('talktype-build-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $stage | Out-Null
try {
    Copy-Item -LiteralPath (Join-Path $root 'scripts\setup.ps1') -Destination $stage
    $command = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File setup.ps1 -PauseOnError'
    if ($SetupArgs) { $command += " $SetupArgs" }
    # IExpress runs AppLaunched through its own parser; "cmd /c" keeps the arguments intact.
    $sed = @"
[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=%InstallPrompt%
DisplayLicense=%DisplayLicense%
FinishMessage=%FinishMessage%
TargetName=%TargetName%
FriendlyName=%FriendlyName%
AppLaunched=%AppLaunched%
PostInstallCmd=%PostInstallCmd%
AdminQuietInstCmd=%AdminQuietInstCmd%
UserQuietInstCmd=%UserQuietInstCmd%
SourceFiles=SourceFiles
[Strings]
InstallPrompt=
DisplayLicense=
FinishMessage=
TargetName=$Output
FriendlyName=talktype
AppLaunched=cmd /c $command
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
FILE0="setup.ps1"
[SourceFiles]
SourceFiles0=$stage\
[SourceFiles0]
%FILE0%=
"@
    $sedPath = Join-Path $stage 'talktype.sed'
    Set-Content -LiteralPath $sedPath -Value $sed -Encoding ASCII
    if (Test-Path -LiteralPath $Output) { Remove-Item -LiteralPath $Output -Force }
    $proc = Start-Process -FilePath "$env:WINDIR\System32\iexpress.exe" -ArgumentList '/N', '/Q', $sedPath -Wait -PassThru -WindowStyle Hidden
    if ($proc.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $Output)) {
        Write-Host "ERRO: o IExpress falhou (código $($proc.ExitCode))."
        exit 1
    }
} finally {
    Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
}
$size = [math]::Round((Get-Item -LiteralPath $Output).Length / 1KB)
Write-Host "Gerado: $Output ($size KB)"
exit 0
