<#
One-time setup on the Windows PC (RTX 3060). Run in an *Administrator* PowerShell from this folder:

    powershell -ExecutionPolicy Bypass -File .\setup.ps1

Optional parameters:
    -Root D:\leafspike        where data, checkpoints, caches, venv and results live (default: %USERPROFILE%\leafspike)
    -Trigger Logon            start at user logon instead of at boot (use if the boot task cannot see the GPU)
    -SkipTask                 do not register the start-up task
    -SkipSmoke                do not run the smoke test

What it does: Python venv + PyTorch (CUDA 12.4) + packages, SDT-v3 checkpoint, PlantVillage (via Kaggle API if
%USERPROFILE%\.kaggle\kaggle.json exists), ViT weight pre-download, config.json, power settings (no sleep on AC),
a smoke test, and the "LeafSpikeExperiments" scheduled task that runs/continues the queue after every boot.
#>
param(
    [string]$Root = "$env:USERPROFILE\leafspike",
    [ValidateSet('Startup', 'Logon')][string]$Trigger = 'Startup',
    [switch]$SkipTask,
    [switch]$SkipSmoke
)
$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Exp = Split-Path -Parent $Here
$TaskName = 'LeafSpikeExperiments'

function Say($m) { Write-Host "`n=== $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[warn] $m" -ForegroundColor Yellow }
function Probe([string]$exe, [string[]]$a) {
    # true if the command exits with 0; output and errors are swallowed
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $exe @a *> $null; return ($LASTEXITCODE -eq 0) } catch { return $false } finally { $ErrorActionPreference = $old }
}
function Capture([string]$exe, [string[]]$a) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $o = & $exe @a 2>$null; if ($LASTEXITCODE -eq 0) { return ($o | Select-Object -Last 1) } return $null }
    catch { return $null } finally { $ErrorActionPreference = $old }
}
function Run([string]$exe, [string[]]$a) {
    & $exe @a
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $exe $($a -join ' ')" }
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $SkipTask -and -not $isAdmin) {
    throw 'Please run this script from an Administrator PowerShell (needed to register the start-up task).'
}

Say "Folders under $Root"
foreach ($d in 'data', 'checkpoints', 'cache', 'runs') { New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null }
Set-Content -Path (Join-Path $Here 'local_env.bat') -Value "@set `"LEAFSPIKE_ROOT=$Root`"" -Encoding Ascii
$env:LEAFSPIKE_ROOT = $Root
$env:TORCH_HOME = Join-Path $Root 'cache\torch'
$env:HF_HOME = Join-Path $Root 'cache\huggingface'

Say 'Python'
$PyExe = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in '3.12', '3.11', '3.10') {
        $out = Capture 'py' @("-$v", '-c', 'import sys; print(sys.executable)')
        if ($out) { $PyExe = $out.Trim(); break }
    }
}
if (-not $PyExe) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $PyExe = $cmd.Source }
}
if (-not $PyExe) { throw 'Python 3.10-3.12 not found. Install it from python.org (tick "Add to PATH") and re-run.' }
Write-Host "using $PyExe"

$Venv = Join-Path $Root 'venv'
$VPy = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $VPy)) { Run $PyExe @('-m', 'venv', $Venv) }
Run $VPy @('-m', 'pip', 'install', '--upgrade', 'pip')

Say 'PyTorch with CUDA 12.4 (large download)'
if (-not (Probe $VPy @('-c', 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'))) {
    Run $VPy @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu124')
}
Run $VPy @('-m', 'pip', 'install', '-r', (Join-Path $Exp 'requirements.txt'))
Run $VPy @('-c', "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')")
if (-not (Probe $VPy @('-c', 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'))) { throw 'PyTorch cannot see the GPU. Update the NVIDIA driver, then re-run this script.' }

Say 'SDT-v3 ImageNet checkpoint (5.1M)'
$Ckpt = Join-Path $Root 'checkpoints\sdt_v3_s_imagenet.pth'
if (-not (Test-Path $Ckpt)) {
    $ok = Probe $VPy @('-c', "import gdown, sys; r = gdown.download(url='https://drive.google.com/uc?id=1LMkOTPehDNpQE79bvB7jFTf6UzDjpAHQ', output=sys.argv[1], quiet=True); sys.exit(0 if r else 1)", $Ckpt)
    if (-not $ok -or -not (Test-Path $Ckpt)) {
        Warn "Checkpoint download failed. Copy sdt_v3_s_imagenet.pth (e.g. from your Kaggle run's /kaggle/working) to $Ckpt"
    }
}
if (Test-Path $Ckpt) { Write-Host "checkpoint: $Ckpt ($([math]::Round((Get-Item $Ckpt).Length / 1MB, 1)) MB)" }

Say 'PlantVillage data'
$Lab = Join-Path $Root 'data\plantvillage'
$haveData = (Test-Path $Lab) -and (Get-ChildItem -Path $Lab -Recurse -Filter '*.JPG' -ErrorAction SilentlyContinue | Select-Object -First 1)
if (-not $haveData) {
    $kj = Join-Path $env:USERPROFILE '.kaggle\kaggle.json'
    $ka = Join-Path $env:USERPROFILE '.kaggle\access_token'
    if ((Test-Path $kj) -or (Test-Path $ka) -or $env:KAGGLE_API_TOKEN) {
        Run $VPy @('-m', 'pip', 'install', 'kaggle')
        Run (Join-Path $Venv 'Scripts\kaggle.exe') @('datasets', 'download', '-d', 'abdallahalidev/plantvillage-dataset', '-p', $Lab, '--unzip')
    } else {
        Warn "PlantVillage not found. Either put kaggle.json in $env:USERPROFILE\.kaggle and re-run, or download"
        Warn "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset and extract it into $Lab"
        Warn '(only the color/Tomato___* folders are used; duplicates are removed automatically)'
    }
} else { Write-Host "found data in $Lab" }

Say 'Pre-downloading ViT ImageNet weights (cached in the root folder)'
$ok = Probe $VPy @('-c', "import timm; [timm.create_model(n, pretrained=True) for n in ['deit_small_patch16_224','swin_tiny_patch4_window7_224','mobilevit_xs','mobilevit_xxs']]")
if ($ok) { Write-Host 'ok' } else { Warn 'ViT weight download failed; the runner will retry (or train from scratch if offline).' }

Say 'config.json'
$Cfg = Join-Path $Exp 'config.json'
if (-not (Test-Path $Cfg)) {
    $workers = 2
    $obj = [ordered]@{ root = $Root; num_workers = $workers; batch_size = 64 }
    ($obj | ConvertTo-Json) | Set-Content -Path $Cfg -Encoding UTF8
}
Get-Content $Cfg

Say 'Power: never sleep / hibernate on AC power (so long runs are not interrupted)'
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

$haveData = (Test-Path $Lab) -and (Get-ChildItem -Path $Lab -Recurse -Filter '*.JPG' -ErrorAction SilentlyContinue | Select-Object -First 1)
if (-not $SkipSmoke -and $haveData) {
    Say 'Smoke test (about 5 minutes; separate runs_smoke folder)'
    Run $VPy @((Join-Path $Exp 'run_all.py'), '--smoke')
}

if (-not $SkipTask) {
    Say "Scheduled task '$TaskName' ($Trigger)"
    $user = (whoami).Trim()
    $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument ('/c "' + (Join-Path $Here 'run_experiments.bat') + '"') -WorkingDirectory $Exp
    if ($Trigger -eq 'Startup') {
        $trig = New-ScheduledTaskTrigger -AtStartup
        $trig.Delay = 'PT2M'
        $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
    } else {
        $trig = New-ScheduledTaskTrigger -AtLogOn -User $user
        $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    }
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 10) -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trig -Principal $principal -Settings $settings -Force | Out-Null
    Write-Host "registered. Start now with:  schtasks /Run /TN $TaskName"
}

Say 'Done'
Write-Host "Results: $Root\runs\report\summary.md   Log: $Root\runs\runner.log   Status: windows\status.bat"
