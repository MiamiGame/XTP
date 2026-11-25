# Builds the PIX and RenderDoc clients (xtp-pix.exe, xtp-rdoc.exe) and the
# WARP test frame (selftest.exe) into E:\xtp-data\pix\bin, never into the
# repository, with the Visual Studio C++ tools, the PIX API headers and
# RenderDoc's replay API. Their C++ sources live outside the repository too,
# in E:\xtp-data\pix\src (the repository's file policy admits no C++ yet;
# docs\analysis\pix-world-frame.md section 8 describes them).
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\pix\build.ps1 [-Src <dir>] [-Out <dir>] [-Pix <PIX folder>]
param([string]$Src = 'E:\xtp-data\pix\src', [string]$Out = 'E:\xtp-data\pix\bin', [string]$Pix = '',
      [string]$RenderDoc = 'E:\tools\gpu\RenderDoc')
$ErrorActionPreference = 'Stop'
if (-not (Test-Path (Join-Path $Src 'xtp-pix.cpp'))) { throw "no xtp-pix.cpp in $Src" }
if (-not $Pix) {
    $Pix = Get-ChildItem 'C:\Program Files\Microsoft PIX' -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName 'include\PixApi.h') } |
        Sort-Object Name | Select-Object -Last 1 -ExpandProperty FullName
}
if (-not $Pix) { throw 'no PIX folder with include\PixApi.h (PIX 2611 or newer)' }
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'no Visual Studio with the C++ tools' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Force $Out | Out-Null
$main = Join-Path $Src 'xtp-pix.cpp'
$obj = Join-Path $Out 'obj'
New-Item -ItemType Directory -Force $obj | Out-Null
$cl = "cl /nologo /std:c++20 /EHsc /O2 /W3 /utf-8 /I`"$Pix\include`" /I`"$Src`" /Fo`"$obj\\`" /Fe`"$Out\xtp-pix.exe`" `"$main`" ole32.lib shlwapi.lib"
cmd /c "`"$vcvars`" >nul && $cl"
if ($LASTEXITCODE) { throw "cl failed ($LASTEXITCODE)" }
"built $Out\xtp-pix.exe against $Pix"
# The known frame of tools\pix\selftest.ps1 (WARP only).
$test = Join-Path $Src 'selftest.cpp'
$cl = "cl /nologo /std:c++20 /EHsc /O2 /W3 /utf-8 /Fo`"$obj\\`" /Fe`"$Out\selftest.exe`" `"$test`" d3d12.lib dxgi.lib d3dcompiler.lib user32.lib"
cmd /c "`"$vcvars`" >nul && $cl"
if ($LASTEXITCODE) { throw "cl failed on selftest.cpp ($LASTEXITCODE)" }
"built $Out\selftest.exe"

# xtp-rdoc.exe, against RenderDoc's replay API: its headers (MIT) at the
# commit renderdoc.dll was built from, fetched into E:\xtp-data\pix\rdapi\
# (outside the repository), and an import library made from the DLL's
# exports. renderdoc.dll itself is loaded at run time from its own folder.
if (-not (Test-Path (Join-Path $RenderDoc 'renderdoc.dll'))) { "no RenderDoc at ${RenderDoc}: xtp-rdoc not built"; return }
$commit = ((& (Join-Path $RenderDoc 'renderdoccmd.exe') version) -join ' ') -replace '.*built from ([0-9a-f]{40}).*', '$1'
if ($commit -notmatch '^[0-9a-f]{40}$') { throw "cannot read RenderDoc's commit from renderdoccmd version" }
$api = "E:\xtp-data\pix\rdapi\$($commit.Substring(0, 7))"
$replay = Join-Path $api 'api\replay'
if (-not (Test-Path (Join-Path $replay 'renderdoc_replay.h'))) {
    New-Item -ItemType Directory -Force $replay | Out-Null
    $list = Invoke-RestMethod -UserAgent 'xtp-pix-build' "https://api.github.com/repos/baldurk/renderdoc/contents/renderdoc/api/replay?ref=$commit"
    foreach ($f in $list) {
        Invoke-WebRequest -UserAgent 'xtp-pix-build' -UseBasicParsing -OutFile (Join-Path $replay $f.name) `
            "https://raw.githubusercontent.com/baldurk/renderdoc/$commit/renderdoc/api/replay/$($f.name)"
    }
}
$def = Join-Path $obj 'renderdoc.def'
$exports = cmd /c "`"$vcvars`" >nul && dumpbin /nologo /exports `"$(Join-Path $RenderDoc 'renderdoc.dll')`""
$names = $exports | ForEach-Object { if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)') { $Matches[1] } }
@('LIBRARY renderdoc', 'EXPORTS') + $names | Set-Content -Encoding ASCII $def
cmd /c "`"$vcvars`" >nul && lib /nologo /machine:x64 /def:`"$def`" /out:`"$obj\renderdoc.lib`""
if ($LASTEXITCODE) { throw "lib failed on renderdoc.def ($LASTEXITCODE)" }
$rdoc = Join-Path $Src 'xtp-rdoc.cpp'
$cl = "cl /nologo /std:c++17 /EHsc /O2 /W3 /utf-8 /wd4996 /I`"$api`" /I`"$Src`" /Fo`"$obj\\`" /Fe`"$Out\xtp-rdoc.exe`" `"$rdoc`" `"$obj\renderdoc.lib`" delayimp.lib /link /DELAYLOAD:renderdoc.dll"
cmd /c "`"$vcvars`" >nul && $cl"
if ($LASTEXITCODE) { throw "cl failed on xtp-rdoc.cpp ($LASTEXITCODE)" }
"built $Out\xtp-rdoc.exe against RenderDoc $commit"
