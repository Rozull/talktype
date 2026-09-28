<#
.SYNOPSIS
    Instala ou atualiza o talktype a partir do GitHub, sem administrador e sem Git.

.DESCRIPTION
    Passos, todos por usuário:
    1. Confere o Windows (10 ou 11) e instala o uv se ele faltar.
    2. Baixa a versão publicada (Releases do GitHub) e a coloca em
       "%LOCALAPPDATA%\talktype\app". Numa atualização, o app aberto é encerrado e o
       ambiente .venv é reaproveitado.
    3. Roda "uv sync". As bibliotecas CUDA (cerca de 2 GB) só entram em máquinas com GPU
       NVIDIA; sem ela, a primeira instalação configura o modelo "parakeet-v3", o mais
       rápido na CPU. Uma configuração existente nunca é alterada.
    4. Cria o atalho no menu Iniciar (scripts\install.ps1), registra o desinstalador em
       "Aplicativos instalados" e abre o talktype.

    A configuração e o histórico ficam em "%APPDATA%\talktype" e sobrevivem a atualizações.
    Compatível com o Windows PowerShell 5.1. Rode como arquivo (-File): por causa do BOM
    UTF-8 dos acentos, o script não funciona colado em "irm | iex".

.PARAMETER Version
    A tag da versão (por exemplo "v0.2.0"). O padrão é a última versão publicada.

.PARAMETER Source
    Só para testes: um .zip local do código no lugar do download.

.PARAMETER InstallDir
    Onde o app fica. O padrão é "%LOCALAPPDATA%\talktype\app".

.PARAMETER Device
    "auto" (padrão) detecta uma GPU NVIDIA; "gpu" e "cpu" forçam a escolha.

.PARAMETER NoLaunch
    Não abre o talktype no fim.

.PARAMETER PauseOnError
    Espera uma tecla quando algo falha, para a mensagem não sumir com a janela (usado pelo
    talktype-setup.exe). Num sucesso, a janela fica aberta por alguns segundos.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1
#>
[CmdletBinding()]
param(
    [string]$Version = 'latest',
    [string]$Source = '',
    [string]$InstallDir = '',
    [ValidateSet('auto', 'gpu', 'cpu')]
    [string]$Device = 'auto',
    [switch]$NoLaunch,
    [switch]$PauseOnError
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest is very slow with the progress bar
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Repo = 'Rozull/talktype'
$Marker = '.talktype-install'

function Step([string]$Message) { Write-Host ''; Write-Host "==> $Message" -ForegroundColor Cyan }
function Fail([string]$Message) {
    Write-Host ''
    Write-Host "ERRO: $Message" -ForegroundColor Red
    if ($PauseOnError) {
        Write-Host ''
        Write-Host 'Pressione uma tecla para fechar.'
        try { [void][Console]::ReadKey($true) } catch { }
    }
    exit 1
}

# 1. Windows e uv ------------------------------------------------------------------------

Step 'Conferindo o sistema'
if ([Environment]::OSVersion.Version.Build -lt 10240) { Fail 'O talktype precisa do Windows 10 ou 11.' }
if (-not $env:LOCALAPPDATA -or -not $env:APPDATA) { Fail 'As variáveis LOCALAPPDATA e APPDATA não estão definidas.' }
if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA 'talktype\app' }
$InstallDir = [IO.Path]::GetFullPath($InstallDir)

$uvBin = Join-Path $env:USERPROFILE '.local\bin'
if (-not (Get-Command 'uv' -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $uvBin 'uv.exe'))) {
    $env:Path = "$uvBin;$env:Path"
}
if (-not (Get-Command 'uv' -ErrorAction SilentlyContinue)) {
    Step 'Instalando o uv (gerenciador de Python)'
    & powershell -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
    $env:Path = "$uvBin;$env:Path"
    if (-not (Get-Command 'uv' -ErrorAction SilentlyContinue)) { Fail 'Não foi possível instalar o uv. Veja https://docs.astral.sh/uv/.' }
}

# 2. O código ----------------------------------------------------------------------------

$work = Join-Path ([IO.Path]::GetTempPath()) ('talktype-setup-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $work | Out-Null
try {
    if ($Source) {
        $zip = (Resolve-Path -LiteralPath $Source).Path
        $label = Split-Path -Leaf $zip
    } else {
        if ($Version -eq 'latest') {
            Step 'Procurando a versão mais recente'
            try {
                $release = Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ 'User-Agent' = 'talktype-setup' }
            } catch {
                Fail "Não foi possível consultar o GitHub ($($_.Exception.Message)). Confira a conexão."
            }
            $Version = [string]$release.tag_name
        }
        $label = $Version
        Step "Baixando o talktype $Version"
        $zip = Join-Path $work 'talktype.zip'
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/$Repo/archive/refs/tags/$Version.zip" -OutFile $zip
        } catch {
            Fail "Download falhou ($($_.Exception.Message))."
        }
    }

    $extract = Join-Path $work 'src'
    Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
    $top = Get-ChildItem -LiteralPath $extract -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'pyproject.toml') } | Select-Object -First 1
    if (-not $top) {
        if (Test-Path (Join-Path $extract 'pyproject.toml')) { $top = Get-Item -LiteralPath $extract } else { Fail 'O pacote baixado não contém o talktype.' }
    }

    # Uma atualização: encerra o talktype que roda desta pasta (o .venv fica em uso).
    $running = @(Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*$InstallDir*" })
    if ($running.Count -gt 0) {
        Step 'Fechando o talktype aberto'
        $running | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 1
    }

    Step "Copiando para $InstallDir"
    if (Test-Path -LiteralPath $InstallDir) {
        if (-not (Test-Path -LiteralPath (Join-Path $InstallDir $Marker)) -and (Get-ChildItem -LiteralPath $InstallDir -Force | Select-Object -First 1)) {
            Fail "A pasta $InstallDir já existe e não foi criada por este instalador."
        }
        Get-ChildItem -LiteralPath $InstallDir -Force | Where-Object { $_.Name -ne '.venv' } | Remove-Item -Recurse -Force
    } else {
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    }
    Get-ChildItem -LiteralPath $top.FullName -Force | Copy-Item -Destination $InstallDir -Recurse -Force
    Set-Content -LiteralPath (Join-Path $InstallDir $Marker) -Value "talktype $label" -Encoding UTF8
} finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}

# 3. Dependências ------------------------------------------------------------------------

$gpu = switch ($Device) {
    'gpu' { $true }
    'cpu' { $false }
    default { [bool](Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1) }
}
$home_tt = if ($env:TALKTYPE_HOME) { $env:TALKTYPE_HOME } else { Join-Path $env:APPDATA 'talktype' }
$firstRun = -not (Test-Path -LiteralPath (Join-Path $home_tt 'config.toml'))

if ($gpu) { Step 'Instalando as dependências, com suporte à GPU NVIDIA (alguns GB na primeira vez)' }
else { Step 'Instalando as dependências (sem GPU NVIDIA: o talktype vai usar a CPU)' }
# Always uv's own Python: the Microsoft Store Python redirects writes under %APPDATA%.
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$syncArgs = @('sync', '--frozen', '--no-default-groups', '--project', $InstallDir)
if ($gpu) { $syncArgs += @('--group', 'gpu') }
& uv @syncArgs
if ($LASTEXITCODE -ne 0) { Fail "uv sync falhou (código $LASTEXITCODE). Rode o instalador de novo." }

$python = Join-Path $InstallDir '.venv\Scripts\python.exe'
if ($firstRun -and -not $gpu) {
    # Sem GPU: Parakeet é o modelo mais rápido na CPU. Só numa configuração nova.
    & $python -c "import sys; from talktype.paths import Paths; from talktype.config import set_value; p = Paths.resolve().ensure(); sys.exit(0 if set_value(p.config, 'asr.model', 'parakeet-v3').saved else 1)"
    if ($LASTEXITCODE -eq 0) { Write-Host 'Modelo configurado: parakeet-v3 (rápido na CPU; troque em asr.model no config.toml).' }
}

# 4. Atalho, desinstalador e início ------------------------------------------------------

Step 'Criando o atalho no menu Iniciar'
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $InstallDir 'scripts\install.ps1')
if ($LASTEXITCODE -ne 0) { Fail 'Não foi possível criar o atalho.' }

$appVersion = (& $python -c 'import talktype; print(talktype.__version__)').Trim()
$uninstallKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\talktype'
New-Item -Path $uninstallKey -Force | Out-Null
$uninstaller = Join-Path $InstallDir 'scripts\uninstall.ps1'
$values = @{
    DisplayName     = 'talktype'
    DisplayVersion  = $appVersion
    Publisher       = 'Rozull'
    InstallLocation = $InstallDir
    UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$uninstaller`""
    URLInfoAbout    = "https://github.com/$Repo"
}
foreach ($name in $values.Keys) { Set-ItemProperty -Path $uninstallKey -Name $name -Value $values[$name] }
Set-ItemProperty -Path $uninstallKey -Name 'NoModify' -Value 1 -Type DWord
Set-ItemProperty -Path $uninstallKey -Name 'NoRepair' -Value 1 -Type DWord
$icon = Join-Path $InstallDir 'src\talktype\assets\talktype.ico'
if (Test-Path -LiteralPath $icon) { Set-ItemProperty -Path $uninstallKey -Name 'DisplayIcon' -Value $icon }

Write-Host ''
Write-Host "talktype $appVersion instalado." -ForegroundColor Green
Write-Host 'Segure o Ctrl direito por 1 s, fale e solte. Na primeira execução o modelo de voz é baixado.'
if (-not $NoLaunch) {
    Start-Process (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\talktype.lnk')
    Write-Host 'O talktype está abrindo: procure o ícone na bandeja, perto do relógio.'
}
if ($PauseOnError) { Start-Sleep -Seconds 8 }
exit 0
