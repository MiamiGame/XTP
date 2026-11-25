# One game session under PIX (or RenderDoc) on the PC: the title from a copy
# of the golden data into the world, a one-frame GPU capture of a world
# frame, then the title ends. Everything it writes goes to E:\xtp-data\pix\<Name>\, never
# into the repository (docs/analysis/pix-world-frame.md).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\pix\capture-world.ps1 `
#       -Name world-1 [-Worktree <XTP checkout with target\diag\xtp.exe>] [-Settle 8] `
#       [-WorldDraws 900] [-WorldAfter 20] [-EnvPairs 'XTP_X=1;XTP_Y=2'] [-Detach] `
#       [-With pix|renderdoc]
#
# - Waits for the PC's game slot as session.ps1 and cargo xtask play do: no
#   live holder of E:\xtp-data\measure.lock and no game xtp.exe running. Then
#   takes the lock (this script's pid on its first line) and keeps it until
#   the end, whatever happens.
# - Copies the golden data folder (XTP_GOLDEN_DATA or E:\xtp-data\golden\data)
#   to <run>\data, writes the autopilot script autopilot\gameplay.txt as
#   <run>\autopilot.txt (no exit: the capture ends the title), and runs the
#   precook as cargo xtask play does (a copy of xtp.exe named xtp-precook.exe).
# - Starts xtp.exe through xtp-pix (E:\xtp-data\pix\bin\xtp-pix.exe, built by
#   tools\pix\build.ps1) for GPU capture (or through xtp-rdoc with -With
#   renderdoc), with the session environment of
#   session.ps1: Series S, XTP_SKIP_INTRO=1, XTP_RAW_STRUCTURED=1, plus
#   XTP_D3D12_DRED=1 (named host objects, a marker per draw and dispatch),
#   XTP_CAPTURE=<run>\frames and XTP_CAPTURE_WORLD=<draws>:<after>:<count>:1,
#   whose first PNG says when the world draws (with XTP_PASS_DUMP in
#   -EnvPairs, the -WorldCount frames from it get XTP's own pass dump, in
#   <run>\frames\passes-<n>.txt). -Settle seconds later it takes the capture
#   <run>\world.wpix (world.rdc with RenderDoc; PIX also saves the presented
#   frame as world.png), then ends the title. Its audio is muted
#   (E:\tools\stall-probe\mute-xtp.ps1).
# - With PIX, lists the capture's events in <run>\events.tsv (no replay).
#   Then runs the walks listed in <run>\walks.txt (xtp-pix or xtp-rdoc walk
#   options, one walk a line), each stopped after -WalkMinutes, before it
#   gives the slot back. PIX replays only with Windows Developer Mode on;
#   RenderDoc needs nothing.
# - -Detach starts the whole thing outside the caller's process tree (WMI) and
#   returns at once; <run>\done.txt holds the exit code at the end.
param(
    [Parameter(Mandatory)][string]$Name,
    [string]$Worktree = '',
    [int]$Settle = 8,
    [int]$WorldDraws = 900,
    [int]$WorldAfter = 20,
    [int]$WorldCount = 1,
    [int]$Timeout = 900,
    [string]$EnvPairs = '',
    [ValidateSet('pix', 'renderdoc')][string]$With = 'pix',
    [string]$Tool = '',
    [int]$WalkMinutes = 20,
    [switch]$NoPrecook,
    [switch]$Detach
)
$ErrorActionPreference = 'Stop'
if (-not $Worktree) { $Worktree = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..\..')).Path }
$run = "E:\xtp-data\pix\$Name"
$exe = if ($With -eq 'pix') { 'xtp-pix' } else { 'xtp-rdoc' }
if (-not $Tool) { $Tool = "E:\xtp-data\pix\bin\$exe.exe" }
$capture = if ($With -eq 'pix') { "$run\world.wpix" } else { "$run\world.rdc" }
$lock = 'E:\xtp-data\measure.lock'

if ($Detach) {
    New-Item -ItemType Directory -Force $run | Out-Null
    Remove-Item "$run\done.txt" -ErrorAction SilentlyContinue
    $flags = @("-Name '$Name'", "-Worktree '$Worktree'", "-Settle $Settle", "-WorldDraws $WorldDraws",
        "-WorldAfter $WorldAfter", "-WorldCount $WorldCount", "-Timeout $Timeout", "-EnvPairs '$($EnvPairs -replace "'", "''")'", "-Tool '$Tool'",
        "-WalkMinutes $WalkMinutes", "-With $With")
    if ($NoPrecook) { $flags += '-NoPrecook' }
    $wrapper = Join-Path $run 'detached.ps1'
    @"
`$code = 1
try {
    & '$PSCommandPath' $($flags -join ' ') *> '$run\session.out.txt'
    `$code = if (`$?) { 0 } else { 1 }
} catch { `$_ | Out-String | Add-Content '$run\session.out.txt' }
"exit `$code at `$((Get-Date).ToString('o'))" | Set-Content '$run\done.txt'
"@ | Set-Content -Encoding UTF8 $wrapper
    # conhost --headless: no window on the person's desktop (as session-detached.ps1).
    $cmd = "conhost.exe --headless powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$wrapper`""
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $Worktree }
    if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
    "started detached: pid $($r.ProcessId); wait for $run\done.txt"
    exit 0
}

function Get-Games {
    @(Get-CimInstance Win32_Process -Filter "Name='xtp.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match '\s(run|play|precook)\b' })
}
function Test-LockHeld {
    if (-not (Test-Path $lock)) { return $false }
    $holder = Get-Content $lock -TotalCount 1 -ErrorAction SilentlyContinue
    if ($holder -match '^\d+$' -and -not (Get-Process -Id ([int]$holder) -ErrorAction SilentlyContinue)) {
        Remove-Item $lock -Force -ErrorAction SilentlyContinue
        return $false
    }
    return $true
}
function Note([string]$text) {
    $line = "$((Get-Date).ToString('o')) $text"
    Add-Content -Encoding UTF8 -Path "$run\times.txt" -Value $line
    $line
}

$xtp = Join-Path $Worktree 'target\diag\xtp.exe'
if (-not (Test-Path $xtp)) { throw "no $xtp (build it: cargo build --profile diag -p xtp --features diagnostics)" }
if (-not (Test-Path $Tool)) { throw "no $Tool (build it: tools\pix\build.ps1)" }
$config = if ($env:XTP_CONFIG) { $env:XTP_CONFIG } else { 'E:\xtp-data\xtp.toml' }
$gameDir = (Select-String -Path $config -Pattern "^\s*game_dir\s*=\s*'([^']+)'").Matches[0].Groups[1].Value
$game = Get-ChildItem $gameDir -Filter 'game_*.exe' | Select-Object -First 1 -ExpandProperty FullName
if (-not $game) { throw "no title executable in $gameDir" }
$golden = if ($env:XTP_GOLDEN_DATA) { $env:XTP_GOLDEN_DATA } else { 'E:\xtp-data\golden\data' }

New-Item -ItemType Directory -Force $run | Out-Null
Note "waiting for the game slot"
$taken = $false
while (-not $taken) {
    while ((Test-LockHeld) -or @(Get-Games).Count) { Start-Sleep -Seconds 5 }
    try {
        $fs = [System.IO.File]::Open($lock, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
        $w = New-Object System.IO.StreamWriter($fs)
        $w.Write("$PID`n$exe capture $Name`n")
        $w.Close()
        $taken = $true
    } catch [System.IO.IOException] { Start-Sleep -Seconds 5 }
}
Note "slot taken (pid $PID)"
$muter = $null
try {
    if (@(Get-Games).Count) { throw 'a game xtp.exe started meanwhile' }
    $data = Join-Path $run 'data'
    Remove-Item $data, "$run\frames", $capture, "$run\world.png" -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item $golden $data -Recurse
    Copy-Item (Join-Path $Worktree 'autopilot\gameplay.txt') "$run\autopilot.txt"
    New-Item -ItemType Directory -Force "$run\frames" | Out-Null

    $env:TMP = $env:TEMP = 'E:\xtp-data\tmp'
    # XTP_D3D12_DRED=1: XTP names its host textures and buffers and puts a
    # marker before each draw and dispatch (its program hashes and slots),
    # which the capture keeps: that is how a pass is tied to a program.
    $vars = [ordered]@{
        XTP_CONFIG = $config; XTP_DATA_DIR = $data; XTP_CONSOLE = 'series-s'; XTP_SKIP_INTRO = '1'
        XTP_RAW_STRUCTURED = '1'; XTP_CAPTURE = "$run\frames"; XTP_CAPTURE_WORLD = "${WorldDraws}:${WorldAfter}:${WorldCount}:1"
        XTP_D3D12_DRED = '1'; TMP = 'E:\xtp-data\tmp'; TEMP = 'E:\xtp-data\tmp'
    }
    foreach ($pair in ($EnvPairs -split ';' | Where-Object { $_ })) { $k, $v = $pair -split '=', 2; $vars[$k] = $v }
    $vars.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" } | Set-Content "$run\env.txt"
    foreach ($k in $vars.Keys) { Set-Item "env:$k" $vars[$k] }

    # Native programs through cmd: Windows PowerShell turns their redirected
    # stderr lines into errors, and it does not escape quotes inside
    # arguments, so no path below may hold a space.
    foreach ($p in @($xtp, $game, $run, $Worktree, $Tool)) { if ($p -match '\s') { throw "a path with a space: $p" } }
    if (-not $NoPrecook) {
        $pack = Join-Path $gameDir 'shaders.rpf'
        $precook = Join-Path (Split-Path $xtp) 'xtp-precook.exe'
        Copy-Item $xtp $precook -Force
        Note "precook"
        $env:XTP_SESSION_LOCK = $lock
        cmd /c "$precook precook $pack --lock-holder $PID > $run\precook.log 2>&1"
        Note "precook exit $LASTEXITCODE"
        Remove-Item env:XTP_SESSION_LOCK
    }

    $muter = Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', 'E:\tools\stall-probe\mute-xtp.ps1', '-Watch', "$($Timeout + 900)")
    $envArgs = ($vars.Keys | ForEach-Object { "--env $_=$($vars[$_])" }) -join ' '
    $toolCmd = "$Tool capture --exe $xtp --cwd $Worktree --args `"run $game --execute --autopilot $run\autopilot.txt --console series-s`" " +
        "--trigger-dir $run\frames --settle-s $Settle --timeout-s $Timeout --out $capture --log $run\$exe.log $envArgs"
    # RenderDoc keeps copies of mapped memory; XTP maps gigabytes of it.
    if ($With -eq 'renderdoc') { $toolCmd += ' --soft-memory-mb 6000' }
    $toolCmd | Set-Content "$run\capture.cmd.txt"
    Note "capture"
    cmd /c "$toolCmd > $run\capture.out.txt 2>&1"
    $code = $LASTEXITCODE
    Note "capture exit $code"
    Start-Sleep -Seconds 5
    foreach ($g in (Get-Games | Where-Object { $_.ExecutablePath -ieq $xtp })) { Stop-Process -Id $g.ProcessId -Force -ErrorAction SilentlyContinue }
    if ($code -eq 0) {
        if ($With -eq 'pix') {
            cmd /c "$Tool events $capture $run\events.tsv > $run\events.out.txt 2>&1"
            Note "events exit $LASTEXITCODE"
        }
        # Walks of the capture, still under the lock (a replay takes the GPU
        # and gigabytes of its memory): one per line of <run>\walks.txt, read
        # only now, so it can be written while the session waits or runs.
        # Each line holds the walk's options; each walk has -WalkMinutes.
        # The file is read again after each walk: lines added meanwhile run too.
        $k = 0
        while ($true) {
            $walks = if (Test-Path "$run\walks.txt") { @(Get-Content "$run\walks.txt" | Where-Object { $_ -match '\S' -and $_ -notmatch '^\s*#' }) } else { @() }
            if ($k -ge $walks.Count) { break }
            $line = $walks[$k]
            $k++
            # <tool>-walk.exe beside the tool, when there is one: a newer
            # build of the walk can be put there while the capture runs.
            $walker = Join-Path (Split-Path $Tool) "$exe-walk.exe"
            if (-not (Test-Path $walker)) { $walker = $Tool }
            Note "walk $k : $line"
            $p = Start-Process -FilePath $walker -PassThru -WindowStyle Hidden -RedirectStandardOutput "$run\walk-$k.out.txt" -RedirectStandardError "$run\walk-$k.err.txt" `
                -ArgumentList "walk $capture $run\walk-$k.tsv $line --log $run\walk-$k.log"
            if (-not $p.WaitForExit($WalkMinutes * 60000)) {
                Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
                Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'WinPixEngineHost.exe', 'renderdoccmd.exe' -and $_.ParentProcessId -eq $p.Id } |
                    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                Note "walk $k stopped after $WalkMinutes min"
            } else { Note "walk $k exit $($p.ExitCode)" }
        }
    }
    if ($code) { exit $code }
} finally {
    if ($muter -and -not $muter.HasExited) { Stop-Process -Id $muter.Id -Force -ErrorAction SilentlyContinue }
    # A RenderDoc crash in a replay opens its crash dialog (qrenderdoc --crash):
    # none stays on the desktop.
    Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'qrenderdoc.exe', 'renderdoccmd.exe' -and $_.CommandLine -match '--crash' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    $holder = Get-Content $lock -TotalCount 1 -ErrorAction SilentlyContinue
    if ($holder -eq "$PID") { Remove-Item $lock -Force }
    Note "slot released"
}
