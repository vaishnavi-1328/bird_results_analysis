# Waits for the bird batch and the running 09-29_B ffmpeg to finish, then:
#  1. moves the finished 09-29_B 1080p file into its own folder
#  2. converts 09-30_A and 10-09_B to 1080p
#  3. runs bird detection on both 1080p files (results in logs\<key>_1080.*; results.csv is not changed)
# Safe to re-run: finished steps are skipped and detection resumes from its checkpoint.
$bin     = "C:\Users\DriveLab\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
$ffmpeg  = "$bin\ffmpeg.exe"
$ffprobe = "$bin\ffprobe.exe"
$python  = "C:\Python314\python.exe"
$here    = "C:\Bird_influenza_analysis\batch_runs"
$log     = "$here\compress_4k.log"
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File $log -Append -Encoding utf8 }
function Duration($f) { [double](& $ffprobe -v error -show_entries format=duration -of csv=p=0 $f) }

function Busy {   # is the bird batch or the 09-29_B compression still running?
    $p = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='ffmpeg.exe'"
    $p | Where-Object { ($_.CommandLine -match 'run_all\.py' -and $_.CommandLine -notmatch '_1080') -or $_.CommandLine -match '09292026\\locationB' }
}

function Compress($in) {   # $in -> $in_1080.mp4 with the agreed settings; returns the output path
    $out  = $in -replace '\.mp4$', '_1080.mp4'
    $part = $in -replace '\.mp4$', '_1080.part.mp4'
    if (Test-Path $out) { Log "already compressed $out"; return $out }
    Log "compress start $in"
    & $ffmpeg -hide_banner -nostdin -y -i $in -vf "scale=1920:1080:flags=lanczos" `
        -c:v libx265 -preset medium -crf 23 -pix_fmt yuv420p `
        -c:a aac -b:a 128k -movflags +faststart $part 2>> ($in -replace '\.mp4$', '_1080.ffmpeg.log')
    if ($LASTEXITCODE -ne 0) { Log "compress FAILED (exit $LASTEXITCODE) $in - see _1080.ffmpeg.log"; exit 1 }
    Move-Item $part $out
    Log "compress done $out"
    return $out
}

function StartDetect($name, $video) {   # background detection, same code/settings as the batch
    $jl = "$here\logs\${name}_1080.jsonl"
    if ((Test-Path $jl) -and (Select-String -Path $jl -Pattern '"event": "result"' -Quiet)) { Log "detection already done - $jl"; return $null }
    Log "detection start $video"
    $env:BIRD_CV_THREADS = "4"
    $p = Start-Process $python -ArgumentList "`"$here\run_all.py`"", '--child', "`"$video`"", "`"$jl`"", "`"$here\logs\${name}_1080.ckpt`"" `
        -RedirectStandardError "$here\logs\${name}_1080.stderr.txt" -WindowStyle Hidden -PassThru
    $null = $p.Handle   # needed so ExitCode is available later
    return $p
}

function FinishDetect($name, $p, $date, $orig) {   # wait, then write the per-minute CSV
    if ($p) { $p.WaitForExit()
        if ($p.ExitCode -ne 0) { Log "detection FAILED $name (exit $($p.ExitCode)) - re-run this script to resume"; return } }
    Remove-Item "$here\logs\${name}_1080.ckpt" -ErrorAction SilentlyContinue
    Push-Location $here
    & $python -c "import run_all as r; r.write_per_minute({'date': '$date', 'timecode': r.read_starttimes()[r.os.path.normcase(r'$orig')]}, r'$here\logs\${name}_1080.jsonl')"
    Pop-Location
    Log "detection done $name - logs\${name}_1080.jsonl and ${name}_1080_per_minute.csv"
}

Log "waiting for bird batch and 09-29_B compression to finish"
while (Busy) { Start-Sleep 60 }
if (-not (Select-String -Path "$here\run_all.out.txt" -Pattern '^all done' -Quiet)) {
    Log "bird batch stopped without 'all done' (crash/reboot?) - stopping; restart run_all.py first"
    exit 1
}

# 1. Move the 09-29_B 1080p file next to its original, if it is complete
$src  = "C:\Farm_videos\farm_video_1080p.mp4"
$orig = "C:\Farm_videos\09292026\locationB\farm_video_compressed.mp4"
$dst  = "C:\Farm_videos\09292026\locationB\farm_video_compressed_1080.mp4"
if ((Test-Path $src) -and -not (Test-Path $dst)) {
    if ([math]::Abs((Duration $src) - (Duration $orig)) -lt 60) { Move-Item $src $dst; Log "moved $src -> $dst" }
    else { Log "NOT moving $src - its length does not match the original (ffmpeg stopped early?)" }
}

# 2-3. Compress 09-30_A, count it while 10-09_B compresses, then count 10-09_B
$a_orig = "C:\Farm_videos\09302026\locationA\farm_video_compressed.mp4"
$b_orig = "C:\Farm_videos\10092026\locationb\merged.mp4"
$a = StartDetect "2026-09-30_A" (Compress $a_orig)
$b_video = Compress $b_orig
$b = StartDetect "2026-10-09_B" $b_video
FinishDetect "2026-09-30_A" $a "2026-09-30" $a_orig
FinishDetect "2026-10-09_B" $b "2026-10-09" $b_orig
Log "all done"
