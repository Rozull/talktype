<#
.SYNOPSIS
    Gera os fixtures de áudio dos testes em tests/fixtures/audio.

.DESCRIPTION
    Sintetiza as frases em português com a voz "Microsoft Maria Desktop" (pt-BR), em WAV
    PCM de 16 kHz, mono e 16 bits. Gera também o silêncio e o ruído branco (semente fixa,
    -40 dBFS) com numpy, e escreve expected.toml com o texto esperado de cada arquivo.

    Os arquivos gerados são versionados; rode este script só para recriá-los.
    Compatível com o Windows PowerShell 5.1 e não precisa de administrador.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\gen_fixtures.ps1
#>
[CmdletBinding()]
param(
    [string]$OutDir = '',
    [string]$Voice = 'Microsoft Maria Desktop'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $scriptDir '..')).Path
if (-not $OutDir) { $OutDir = Join-Path $root 'tests\fixtures\audio' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

# Frases faladas. KeyTerms são os termos que os testes exigem na transcrição normalizada.
$spoken = @(
    [ordered]@{
        Name     = 'deploy_pr'
        Rate     = 1
        Text     = 'Então, antes de fazer o deploy, abre um pull request no GitHub com as mudanças da pipeline, espera o CI passar em todos os testes e, se estiver tudo verde, faz o merge na main e avisa o time no canal.'
        KeyTerms = @('deploy', 'pull request', 'github', 'pipeline', 'ci')
    },
    [ordered]@{
        Name     = 'curta_sim'
        Rate     = 0
        Text     = 'Sim.'
        KeyTerms = @('sim')
    },
    [ordered]@{
        Name     = 'nova_linha'
        Rate     = 0
        Text     = 'Primeiro item da lista nova linha segundo item da lista'
        KeyTerms = @('primeiro item', 'nova linha', 'segundo item')
    },
    [ordered]@{
        Name     = 'hesitacao'
        Rate     = -1
        Text     = 'Hum, eu acho que, ahn, a gente pode revisar isso amanhã, hum, depois da reunião.'
        KeyTerms = @('revisar', 'amanhã', 'reunião')
    }
)

Add-Type -AssemblyName System.Speech
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SelectVoice($Voice)
    foreach ($item in $spoken) {
        $path = Join-Path $OutDir ($item.Name + '.wav')
        $synth.Rate = $item.Rate
        $synth.SetOutputToWaveFile($path, $format)
        $synth.Speak($item.Text)
        $synth.SetOutputToNull()
        Write-Host "gerado $path"
    }
}
finally {
    $synth.Dispose()
}

# Silêncio e ruído: numpy com semente fixa, para serem reproduzíveis byte a byte.
$python = @'
import sys
import wave
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
rate = 16000
samples = 3 * rate


def write(name, data):
    pcm = np.clip(np.round(data * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(out / name), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    print("gerado", out / name)


write("silencio_3s.wav", np.zeros(samples))
rms = 10 ** (-40 / 20)  # -40 dBFS
write("ruido_3s.wav", np.random.default_rng(20260928).normal(0.0, rms, samples))
'@
$python | uv run --project $root python - $OutDir
if ($LASTEXITCODE -ne 0) { throw "falha ao gerar silêncio e ruído (código $LASTEXITCODE)" }

function Get-WavSeconds([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $reader = New-Object System.IO.BinaryReader($stream)
        $stream.Position = 28
        $byteRate = $reader.ReadUInt32()
        $stream.Position = 12
        while ($stream.Position -lt $stream.Length) {
            $chunkId = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
            $chunkSize = $reader.ReadUInt32()
            if ($chunkId -eq 'data') { return [math]::Round($chunkSize / $byteRate, 2) }
            $stream.Position += $chunkSize
        }
        throw "sem bloco data: $Path"
    }
    finally {
        $stream.Dispose()
    }
}

function ConvertTo-TomlString([string]$Value) {
    return '"' + $Value.Replace('\', '\\').Replace('"', '\"') + '"'
}

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('# Gerado por scripts/gen_fixtures.ps1. Não edite à mão; rode o script de novo.')
$lines.Add("# Voz: $Voice. Formato: WAV PCM 16 kHz, mono, 16 bits.")
foreach ($item in $spoken) {
    $file = $item.Name + '.wav'
    $terms = ($item.KeyTerms | ForEach-Object { ConvertTo-TomlString $_ }) -join ', '
    $lines.Add('')
    $lines.Add("[$($item.Name)]")
    $lines.Add("file = $(ConvertTo-TomlString $file)")
    $lines.Add("text = $(ConvertTo-TomlString $item.Text)")
    $lines.Add("key_terms = [$terms]")
    $lines.Add('speech = true')
    $seconds = Get-WavSeconds (Join-Path $OutDir $file)
    $lines.Add('duration_s = ' + $seconds.ToString([System.Globalization.CultureInfo]::InvariantCulture))
}
foreach ($name in @('silencio_3s', 'ruido_3s')) {
    $lines.Add('')
    $lines.Add("[$name]")
    $lines.Add("file = `"$name.wav`"")
    $lines.Add('text = ""')
    $lines.Add('key_terms = []')
    $lines.Add('speech = false')
    $lines.Add('duration_s = 3.0')
}

$expected = Join-Path $OutDir 'expected.toml'
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($expected, (($lines -join "`n") + "`n"), $utf8)
Write-Host "gerado $expected"
