# Checks xtp-pix and xtp-rdoc on a frame whose values are known (selftest.cpp,
# WARP only: no hardware GPU is touched): capture, replay, readback and the
# statistics. Prints PASS or FAIL per check; exit code 1 on any FAIL.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\pix\selftest.ps1 [-Bin E:\xtp-data\pix\bin]
# PIX's replay needs Windows Developer Mode: without it the PIX walk is
# reported as skipped, and only its capture and its walk without replay are
# checked.
param([string]$Bin = 'E:\xtp-data\pix\bin', [string]$Out = 'E:\xtp-data\pix\selftest')
$ErrorActionPreference = 'Stop'
$failed = 0
function Check([string]$what, [bool]$ok, [string]$detail = '') {
    "{0} {1}{2}" -f ($(if ($ok) { 'PASS' } else { 'FAIL' })), $what, $(if ($detail) { ": $detail" } else { '' })
    if (-not $ok) { $script:failed++ }
}
function Rows([string]$tsv) { Import-Csv -Delimiter "`t" $tsv }
function Near([string]$value, [double]$want) {
    $v = 0.0
    [double]::TryParse($value, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$v) -and
        [math]::Abs($v - $want) -lt 1e-4
}

foreach ($tool in 'xtp-pix', 'xtp-rdoc') {
    $exe = Join-Path $Bin "$tool.exe"
    if (-not (Test-Path $exe)) { Check "$tool built" $false "no $exe (tools\pix\build.ps1)"; continue }
    $dir = Join-Path $Out $tool
    Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $dir | Out-Null
    $ext = if ($tool -eq 'xtp-pix') { 'wpix' } else { 'rdc' }
    $cap = Join-Path $dir "self.$ext"
    cmd /c "$exe capture --exe $Bin\selftest.exe --cwd $dir --args $dir --trigger-dir $dir --settle-s 2 --timeout-s 60 --out $cap > $dir\capture.txt 2>&1"
    Check "$tool capture" ($LASTEXITCODE -eq 0 -and (Test-Path $cap)) "exit $LASTEXITCODE"
    if (-not (Test-Path $cap)) { continue }
    if ($tool -eq 'xtp-pix') {
        cmd /c "$exe walk $cap $dir\views.tsv --no-replay --each > $dir\views.txt 2>&1"
        $v = @(Rows "$dir\views.tsv")
        Check "xtp-pix walk without replay" ($v.Count -eq 3 -and ($v | Where-Object { $_.kind -eq 'DSV' -and $_.call -match 'depth on write on' })) "$($v.Count) rows"
        cmd /c "$exe walk $cap $dir\walk.tsv --adapter `"Basic Render`" --inputs --depth-diff --passes --accessed > $dir\walk.txt 2>&1"
        if ((Get-Content "$dir\walk.txt" -Raw) -match 'StartAnalysis.*failed') {
            "SKIP xtp-pix walk with replay: PIX did not start its replay (Developer Mode off?)"
            continue
        }
    } else {
        cmd /c "$exe walk $cap $dir\walk.tsv --software --inputs --depth-diff --passes > $dir\walk.txt 2>&1"
    }
    $r = @(Rows "$dir\walk.tsv")
    $a = $r | Where-Object { $_.kind -eq 'RTV' -and $_.when -eq 'after' } | Select-Object -First 1
    Check "$tool draw output A" ($a -and (Near $a.mean_lum 0.294125) -and (Near $a.max_lum 0.294125)) "mean $($a.mean_lum)"
    $d = $r | Where-Object { $_.when -eq 'depth-diff' } | Select-Object -First 1
    Check "$tool depth written" ($d -and $d.note -match 'changed: 32768' -and (Near $d.mean_lum 0.25)) "$($d.note)"
    $c = $r | Where-Object { $_.kind -eq 'UAV' -and $_.when -eq 'after' } | Select-Object -First 1
    Check "$tool dispatch output C" ($c -and (Near $c.mean_lum 2.0)) "mean $($c.mean_lum)"
    $s = $r | Where-Object { $_.kind -eq 'SRV' -and $_.when -eq 'before' } | Select-Object -First 1
    Check "$tool dispatch input A" ($s -and (Near $s.mean_lum 0.294125)) "mean $($s.mean_lum)"
}
Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'qrenderdoc.exe', 'renderdoccmd.exe' -and $_.CommandLine -match '--crash' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
if ($failed) { "$failed check(s) failed"; exit 1 }
"all checks passed"
