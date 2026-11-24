<#
start_translator.ps1
--------------------
Runs the translator on this PC for the Learnut app. Double-click
start-translator.cmd in the project folder; it runs this script.

Every run:
  1. Installs what is missing (Python 3.12, Ollama, cloudflared). If there is a
     new translator-lora*.zip in Downloads, it unpacks it, merges it into the
     base model, converts it to GGUF and loads it into Ollama. The first run
     takes 15-30 minutes. Later runs skip this unless a newer zip appears.
  2. Starts scripts/serve.py in backend mode on 127.0.0.1:8000, in front of
     Ollama.
  3. Opens a Cloudflare quick tunnel, which gives a public
     https://*.trycloudflare.com URL.
  4. If the owner said yes (asked once), points Learnut at that URL: Supabase
     table app_config, row app_translator, field get_api_translator. Learnut
     has one production Supabase project, so this switches EVERY user,
     testers included.
  5. Watches everything. Restarts what stops and publishes a new tunnel URL
     when it changes. Keeps Windows awake.
  6. Ctrl+C stops it all and points Learnut back to the URL it found.

Options:
  -Rebuild            rebuild the GGUF and the Ollama model even if nothing changed
  -SetupOnly          do step 1, translate one test sentence locally, then exit
  -NoSupabase         leave Supabase alone this run and just print the URL
  -AddToStartup       start the translator (minimized) every time you sign in to Windows
  -RemoveFromStartup  stop doing that

The Supabase URL and secret key are stored encrypted (Windows DPAPI, this
user only) in %LOCALAPPDATA%\translator-server\settings.json. Delete that file
to be asked again. Logs are in outputs\server\.

This file must stay plain ASCII: Windows PowerShell 5.1 reads a script without
a BOM in the ANSI code page.
#>
param(
    [switch]$Rebuild,
    [switch]$SetupOnly,
    [switch]$NoSupabase,
    [switch]$AddToStartup,
    [switch]$RemoveFromStartup,
    [switch]$PauseOnError
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with a progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$ServerDir     = Join-Path $Root "outputs\server"
$AdapterDir    = Join-Path $Root "outputs\translator-lora"
$MergedDir     = Join-Path $Root "outputs\translator-merged"
$Gguf          = Join-Path $MergedDir "translator-q8_0.gguf"
$ExportVenv    = Join-Path $Root "outputs\export-venv"
$LlamaCpp      = Join-Path $Root "outputs\llama.cpp"
$ToolsDir      = Join-Path $Root "outputs\tools"
$ServeVenv     = Join-Path $Root ".venv"
$SettingsDir   = Join-Path $env:LOCALAPPDATA "translator-server"
$SettingsFile  = Join-Path $SettingsDir "settings.json"
$OllamaUrl    = "http://127.0.0.1:11434"
$ServePort     = 8000
$ServeUrl      = "http://127.0.0.1:$ServePort"
$ModelName     = "translator"
$MaxInputChars = "64"   # Learnut caps input at app_translator.translation_string_length (30)
# Supabase refuses secret keys from anything that looks like a browser, judged by the
# User-Agent. Windows PowerShell's default one starts with "Mozilla/5.0" (401).
$SupabaseUserAgent = "learnut-translator/1.0"
$StartupShortcut   = Join-Path ([Environment]::GetFolderPath("Startup")) "Learnut translator.lnk"

New-Item -ItemType Directory -Force $ServerDir | Out-Null
$env:OLLAMA_KEEP_ALIVE = "-1"                                     # anything started from here keeps the model loaded
Remove-Item Env:TRANSLATOR_API_KEY -ErrorAction SilentlyContinue  # Learnut sends no key; a key would make it fail

# ---- Output ---------------------------------------------------------------------

function Write-Step([string]$Message) { Write-Host ""; Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Info([string]$Message) { Write-Host "    $Message" }
function Write-Warn([string]$Message) { Write-Host "    WARNING: $Message" -ForegroundColor Yellow }
function Write-Status([string]$Message, [string]$Color = "Gray") {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message) -ForegroundColor $Color
}

function Get-LogTail([string]$Path, [int]$Lines = 15) {
    # Reads a log another process still has open.
    if (-not (Test-Path $Path)) { return "" }
    try {
        $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
        try { $text = (New-Object IO.StreamReader($stream)).ReadToEnd() } finally { $stream.Dispose() }
        return ((($text -split "`r?`n") | Select-Object -Last $Lines) -join "`n")
    } catch { return "" }
}

# ---- Small helpers --------------------------------------------------------------

function Invoke-Native([scriptblock]$Command, [string]$What) {
    # Native tools print progress on stderr; only the exit code means failure.
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command | Out-Host; $exitCode = $LASTEXITCODE } finally { $ErrorActionPreference = $saved }
    if ($exitCode -ne 0) { throw "$What failed (exit code $exitCode)." }
}

function Test-PythonCode([string]$Python, [string]$Code) {
    # Python code passed here must not contain double quotes (PowerShell 5.1 mangles them).
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Python -c $Code 2>$null | Out-Null; return ($LASTEXITCODE -eq 0) }
    catch { return $false }
    finally { $ErrorActionPreference = $saved }
}

function Test-Http([string]$Url, [int]$TimeoutSec = 5) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    } catch { return $false }
}

function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}

function Stop-Child($Process) {
    if ($Process -and -not $Process.HasExited) {
        try { Stop-Process -Id $Process.Id -Force -ErrorAction Stop } catch { }
    }
}

# ---- Installing tools -----------------------------------------------------------

function Install-WithWinget([string]$Id, [string]$Name, [string]$Override = "", [switch]$UserScope) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is missing, so $Name can't be installed automatically. Install $Name yourself, then run this again."
    }
    Write-Step "Installing $Name (one time)"
    $wingetArgs = @("install", "--id", $Id, "--exact", "--source", "winget",
                    "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity")
    if ($Override) { $wingetArgs += @("--override", $Override) } else { $wingetArgs += "--silent" }
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $exitCode = 1
        if ($UserScope) { & winget @wingetArgs --scope user | Out-Host; $exitCode = $LASTEXITCODE }
        if ($exitCode -ne 0) { & winget @wingetArgs | Out-Host; $exitCode = $LASTEXITCODE }
    } finally { $ErrorActionPreference = $saved }
    Update-SessionPath
    return $exitCode
}

function Find-PythonExe([string]$Version) {
    $short = $Version.Replace(".", "")
    foreach ($candidate in @((Join-Path $env:LOCALAPPDATA "Programs\Python\Python$short\python.exe"),
                             (Join-Path $env:ProgramFiles "Python$short\python.exe"))) {
        if ((Test-Path $candidate) -and (Test-PythonCode $candidate "import sys")) { return $candidate }
    }
    return $null
}

function Get-BasePython {
    # 3.12 or 3.11: both have wheels for every pinned package (torch 2.5.1 included).
    foreach ($version in @("3.12", "3.11")) {
        $exe = Find-PythonExe $version
        if ($exe) { return $exe }
    }
    # Per-user, no py launcher: needs no administrator rights.
    $exitCode = Install-WithWinget "Python.Python.3.12" "Python 3.12" -UserScope `
        -Override "/quiet InstallAllUsers=0 Include_launcher=0 PrependPath=1 Include_test=0"
    $exe = Find-PythonExe "3.12"
    if (-not $exe) {
        throw "Python 3.12 did not install (winget exit code $exitCode). Install it from python.org, then run this again."
    }
    return $exe
}

function Get-OllamaExe {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $exe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $exe) { return $exe }
    return $null
}

function Install-Ollama {
    $exe = Get-OllamaExe
    if ($exe) { return $exe }
    $exitCode = Install-WithWinget "Ollama.Ollama" "Ollama"
    $exe = Get-OllamaExe
    if (-not $exe) { throw "Ollama did not install (winget exit code $exitCode). Install it from ollama.com, then run this again." }
    return $exe
}

function Start-Ollama([string]$OllamaExe) {
    if (Test-Http "$OllamaUrl/api/version") { return }
    Write-Info "Starting Ollama ..."
    $app = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
    if (Test-Path $app) { Start-Process -FilePath $app | Out-Null }
    else { Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden | Out-Null }
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        if (Test-Http "$OllamaUrl/api/version") { return }
        Start-Sleep -Seconds 2
    }
    throw "Ollama did not start. Open Ollama from the Start menu, then run this again."
}

function Set-OllamaKeepAlive {
    # Without this, Ollama unloads the model after 5 idle minutes and the next
    # translation waits for it to load again.
    if ([Environment]::GetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "User") -eq "-1") { return }
    [Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "-1", "User")
    if (Test-Http "$OllamaUrl/api/version") {
        Write-Info "Restarting Ollama once so it keeps the model loaded ..."
        Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
    }
}

function Test-OllamaModel {
    try { $tags = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -TimeoutSec 10 } catch { return $false }
    foreach ($model in @($tags.models)) {
        if ($model.name -eq "${ModelName}:latest" -or $model.name -eq $ModelName) { return $true }
    }
    return $false
}

function Get-Cloudflared {
    $command = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $exe = Join-Path $ToolsDir "cloudflared.exe"
    if (Test-Path $exe) { return $exe }
    Write-Step "Downloading cloudflared (one time)"
    New-Item -ItemType Directory -Force $ToolsDir | Out-Null
    $download = Join-Path $ToolsDir "cloudflared.download.exe"
    Invoke-WebRequest -UseBasicParsing -OutFile $download `
        -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    $signature = Get-AuthenticodeSignature $download
    if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notmatch "Cloudflare") {
        Remove-Item $download -Force
        throw "The downloaded cloudflared.exe is not signed by Cloudflare (signature: $($signature.Status)), so it was deleted."
    }
    Move-Item $download $exe
    return $exe
}

# ---- The model ------------------------------------------------------------------

function Sync-AdapterFromDownloads {
    # Unpacks the newest translator-lora*.zip from Downloads when it is not the
    # one unpacked last time. Returns $true when the model must be rebuilt.
    $downloads = Join-Path $env:USERPROFILE "Downloads"
    $zip = Get-ChildItem -Path $downloads -Filter "translator-lora*.zip" -File -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $stampFile = Join-Path $ServerDir "adapter-source.txt"
    $haveAdapter = Test-Path (Join-Path $AdapterDir "adapter_model.safetensors")
    if (-not $zip) {
        if ($haveAdapter) { return $false }
        throw "No trained model found. Put translator-lora.zip (from Kaggle or Colab) in your Downloads folder, then run this again."
    }
    $stamp = "{0}|{1}|{2}" -f $zip.Name, $zip.Length, $zip.LastWriteTimeUtc.Ticks
    $previous = ""
    if (Test-Path $stampFile) { $previous = (Get-Content $stampFile -Raw).Trim() }
    if ($haveAdapter -and $stamp -eq $previous) { return $false }

    Write-Step "Unpacking $($zip.Name) from Downloads"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($zip.FullName)
    try { $names = @($archive.Entries | ForEach-Object { $_.FullName }) } finally { $archive.Dispose() }
    if ($names -notcontains "adapter_model.safetensors" -or $names -notcontains "adapter_config.json") {
        throw "$($zip.Name) is not a trained adapter (no adapter_model.safetensors at its top level)."
    }
    if (Test-Path $AdapterDir) { Remove-Item $AdapterDir -Recurse -Force }
    Expand-Archive -Path $zip.FullName -DestinationPath $AdapterDir
    Set-Content -Path $stampFile -Value $stamp -Encoding ASCII
    return $true
}

function Register-Model([string]$OllamaExe) {
    Write-Step "Loading the model into Ollama"
    Push-Location $MergedDir
    try { Invoke-Native { & $OllamaExe create $ModelName -f Modelfile } "ollama create" } finally { Pop-Location }
}

function Build-Model([string]$BasePython, [string]$OllamaExe) {
    # adapter -> merged model (scripts/export_model.py) -> q8_0 GGUF -> Ollama.
    $py = Join-Path $ExportVenv "Scripts\python.exe"
    if ((Test-Path $py) -and -not (Test-PythonCode $py "import sys")) { Remove-Item $ExportVenv -Recurse -Force }
    if (-not (Test-Path $py)) {
        Write-Step "Creating the model-building environment (one time)"
        Invoke-Native { & $BasePython -m venv $ExportVenv } "Creating outputs\export-venv"
    }
    Write-Step "Installing the model-building tools (the first time takes a few minutes)"
    if (-not (Test-PythonCode $py "import torch")) {
        Invoke-Native { & $py -m pip install --disable-pip-version-check torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu } "Installing torch"
    }
    # The same versions as training (requirements.txt). The GGUF converter below may
    # upgrade transformers afterwards, so check them again on every rebuild.
    if (-not (Test-PythonCode $py "import transformers, peft, accelerate; assert transformers.__version__ == '4.46.3' and peft.__version__ == '0.13.2'")) {
        Invoke-Native { & $py -m pip install --disable-pip-version-check transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 } "Installing transformers and peft"
    }

    Write-Step "Merging the adapter into the base model (downloads Qwen2.5-1.5B, about 3 GB, the first time)"
    if (Test-Path $MergedDir) { Remove-Item $MergedDir -Recurse -Force }
    Invoke-Native { & $py scripts\export_model.py --adapter-dir $AdapterDir --out-dir $MergedDir } "Merging the adapter"

    if (-not (Test-Path (Join-Path $LlamaCpp "convert_hf_to_gguf.py"))) {
        Write-Step "Downloading the llama.cpp converter (one time)"
        if (Test-Path $LlamaCpp) { Remove-Item $LlamaCpp -Recurse -Force }
        Invoke-Native { git clone --depth 1 https://github.com/ggml-org/llama.cpp $LlamaCpp } "Downloading llama.cpp"
    }
    $requirements = Join-Path $LlamaCpp "requirements\requirements-convert_hf_to_gguf.txt"
    if (-not (Test-Path $requirements)) { $requirements = Join-Path $LlamaCpp "requirements.txt" }
    Invoke-Native { & $py -m pip install --disable-pip-version-check -r $requirements } "Installing the GGUF converter's packages"

    Write-Step "Converting to GGUF (q8_0, about 1.6 GB)"
    Invoke-Native { & $py (Join-Path $LlamaCpp "convert_hf_to_gguf.py") $MergedDir --outtype q8_0 --outfile $Gguf } "Converting to GGUF"
    Register-Model $OllamaExe
}

function Initialize-ServeEnv([string]$BasePython) {
    # serve.py in backend mode needs only fastapi and uvicorn (no torch).
    $py = Join-Path $ServeVenv "Scripts\python.exe"
    if ((Test-Path $py) -and -not (Test-PythonCode $py "import sys")) {
        Write-Info "Replacing a .venv whose Python was uninstalled ..."
        Remove-Item $ServeVenv -Recurse -Force
    }
    if (-not (Test-Path $py)) {
        Write-Step "Creating the server environment (one time)"
        Invoke-Native { & $BasePython -m venv $ServeVenv } "Creating .venv"
    }
    if (-not (Test-PythonCode $py "import fastapi, uvicorn")) {
        Invoke-Native { & $py -m pip install --disable-pip-version-check fastapi==0.115.5 uvicorn==0.32.1 } "Installing fastapi and uvicorn"
    }
    return $py
}

# ---- Running ----------------------------------------------------------------------

function Stop-ServeListeners {
    # .venv\Scripts\python.exe is a small launcher; the real interpreter is its
    # child, so find it by the port and its command line.
    foreach ($connection in @(Get-NetTCPConnection -LocalPort $ServePort -State Listen -ErrorAction SilentlyContinue)) {
        $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($connection.OwningProcess)" -ErrorAction SilentlyContinue
        if ($owner -and $owner.CommandLine -like "*serve.py*") {
            Stop-Process -Id $owner.ProcessId -Force -ErrorAction SilentlyContinue
        } else {
            $name = "another program"
            if ($owner) { $name = $owner.Name }
            throw "Port $ServePort is already used by $name. Close it, then run this again."
        }
    }
}

function Start-Server([string]$Python) {
    Stop-ServeListeners
    $env:TRANSLATOR_BACKEND_URL = "$OllamaUrl/v1"
    $env:TRANSLATOR_BACKEND_MODEL = $ModelName
    $env:MAX_INPUT_CHARS = $MaxInputChars
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $log = Join-Path $ServerDir "serve-$stamp.log"
    $process = Start-Process -FilePath $Python -ArgumentList "scripts\serve.py --host 127.0.0.1 --port $ServePort" `
        -WorkingDirectory $Root -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $ServerDir "serve-$stamp-requests.log") -RedirectStandardError $log
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline -and -not $process.HasExited) {
        if (Test-Http "$ServeUrl/health") { return $process }
        Start-Sleep -Seconds 1
    }
    Write-Host (Get-LogTail $log 20) -ForegroundColor Yellow
    Stop-Child $process
    throw "serve.py did not start (log: $log)."
}

function Stop-Server($Process) {
    Stop-Child $Process
    try { Stop-ServeListeners } catch { }
}

function Invoke-TestTranslation {
    $body = '{"sentence": "Saan ka pupunta?", "source_lang": "Tagalog", "target_lang": "Cuyonon", "translated": null}'
    $answer = Invoke-RestMethod -Method Post -Uri "$ServeUrl/translate" -TimeoutSec 180 `
        -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
    return $answer.translated_sentence
}

function Wait-PublicUrl([string]$Url, [int]$TimeoutSec) {
    $hostName = ([Uri]$Url).Host
    $start = Get-Date
    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSec) {
        # Asking Windows for a name before it exists makes Windows remember "not
        # found" for minutes, so wait until a public resolver knows it.
        $known = $false
        try { Resolve-DnsName -Name $hostName -Server 1.1.1.1 -DnsOnly -QuickTimeout -ErrorAction Stop | Out-Null; $known = $true } catch { }
        if ($known -or ((Get-Date) - $start).TotalSeconds -gt 30) {
            if (Test-Http "$Url/health" 15) { return $true }
        }
        Start-Sleep -Seconds 3
    }
    return $false
}

function Start-Tunnel([string]$Cloudflared) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $log = Join-Path $ServerDir "tunnel-$stamp.log"
    $process = Start-Process -FilePath $Cloudflared -ArgumentList "tunnel --no-autoupdate --url $ServeUrl" `
        -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $ServerDir "tunnel-$stamp-out.log") -RedirectStandardError $log
    $url = $null
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline -and -not $process.HasExited) {
        $found = [regex]::Match((Get-LogTail $log 400), 'https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com')
        if ($found.Success) { $url = $found.Value; break }
        Start-Sleep -Seconds 1
    }
    if (-not $url) {
        Write-Host (Get-LogTail $log 15) -ForegroundColor Yellow
        Stop-Child $process
        throw "cloudflared gave no public URL (log: $log)."
    }
    if (-not (Wait-PublicUrl $url 120)) {
        Stop-Child $process
        throw "The tunnel URL $url never answered (log: $log)."
    }
    return @{ Process = $process; Url = $url }
}

function Show-Banner([string]$Url, [bool]$Switched, $Settings) {
    Write-Host ""
    Write-Host "  ==================================================================" -ForegroundColor Green
    Write-Host "   The translator is running on this PC" -ForegroundColor Green
    Write-Host "   Public URL : $Url"
    if ($Switched) { Write-Host "   Learnut    : uses this PC now (people get it when they reopen the app)" }
    else { Write-Host "   Learnut    : not switched (it still uses its usual translator)" }
    Write-Host "   Logs       : $ServerDir"
    Write-Host "   Keep this window open. Press Ctrl+C to stop."
    if ($Switched -and $Settings.fallbackUrl) { Write-Host "   Stopping switches Learnut back to $($Settings.fallbackUrl)" }
    Write-Host "  ==================================================================" -ForegroundColor Green
    Write-Host ""
}

function Disable-Sleep {
    try {
        Add-Type -Namespace TranslatorServer -Name Power -MemberDefinition `
            '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
        [void][TranslatorServer.Power]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    } catch { Write-Warn "Could not keep Windows awake: $($_.Exception.Message)" }
}

function Enable-Sleep {
    try { [void][TranslatorServer.Power]::SetThreadExecutionState([uint32]2147483648) } catch { }   # ES_CONTINUOUS
}

# ---- Supabase (Learnut's app_config) ----------------------------------------------

function ConvertTo-PlainText([Security.SecureString]$Secure) {
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}

function Unprotect-Text([string]$Encrypted) {
    return (ConvertTo-PlainText (ConvertTo-SecureString $Encrypted))
}

function Get-SupabaseHeaders($Settings) {
    $key = Unprotect-Text $Settings.key
    $headers = @{ apikey = $key }
    # Legacy keys are JWTs and also go in Authorization. New sb_secret_ keys go in apikey only.
    if ($key.StartsWith("eyJ")) { $headers["Authorization"] = "Bearer $key" }
    return $headers
}

function Invoke-Supabase($Settings, [string]$Method, [string]$PathAndQuery, [string]$Json = "", [hashtable]$ExtraHeaders = @{}) {
    $headers = Get-SupabaseHeaders $Settings
    foreach ($name in $ExtraHeaders.Keys) { $headers[$name] = $ExtraHeaders[$name] }
    $params = @{ Method = $Method; Uri = "$($Settings.url)$PathAndQuery"; Headers = $headers
                 TimeoutSec = 20; UserAgent = $SupabaseUserAgent }
    if ($Json) {
        $params.ContentType = "application/json; charset=utf-8"
        $params.Body = [Text.Encoding]::UTF8.GetBytes($Json)
    }
    try { return (Invoke-RestMethod @params) }
    catch {
        $status = ""
        if ($_.Exception.Response) { $status = [int]$_.Exception.Response.StatusCode }
        $detail = $_.ErrorDetails.Message
        if (-not $detail) { $detail = $_.Exception.Message }
        if ($status -eq 401) {
            throw ("Supabase refused the key (401): $detail Copy the whole secret key of the Learnut " +
                   "project (Project Settings > API Keys; reveal it before copying).")
        }
        throw "Supabase request failed ($status): $detail"
    }
}

function Get-TranslatorConfig($Settings) {
    $response = Invoke-Supabase $Settings "Get" "/rest/v1/app_config?key=eq.app_translator&select=value"
    $row = @($response) | Select-Object -First 1
    if ($null -eq $row -or $null -eq $row.value) { throw "Supabase has no app_config row with key 'app_translator'." }
    return $row.value
}

function Set-TranslatorUrl($Settings, [string]$NewUrl) {
    # Changes only get_api_translator and keeps the row's other fields. Returns the old URL.
    $value = Get-TranslatorConfig $Settings
    $previous = [string]$value.get_api_translator
    $value | Add-Member -NotePropertyName "get_api_translator" -NotePropertyValue $NewUrl -Force
    $json = @{ value = $value } | ConvertTo-Json -Depth 20 -Compress
    $response = Invoke-Supabase $Settings "Patch" "/rest/v1/app_config?key=eq.app_translator" $json `
        @{ Prefer = "return=representation" }
    $row = @($response) | Select-Object -First 1
    if ($null -eq $row -or [string]$row.value.get_api_translator -ne $NewUrl) {
        throw "Supabase did not save the new URL. Use the secret (service role) key, not the publishable one."
    }
    if ($row.value.api_translator_enabled -eq $false) {
        Write-Warn "api_translator_enabled is false in Supabase, so Learnut shows 'service unavailable' instead of translating."
    }
    return $previous
}

function Read-Settings {
    if (-not (Test-Path $SettingsFile)) { return $null }
    return (Get-Content $SettingsFile -Raw | ConvertFrom-Json)
}

function Save-Settings($Settings) {
    New-Item -ItemType Directory -Force $SettingsDir | Out-Null
    $Settings | ConvertTo-Json | Set-Content -Path $SettingsFile -Encoding UTF8
}

function Find-LearnutSupabaseUrl {
    # Learnut is normally checked out next to this repo's parent folder (see CLAUDE.md).
    try {
        $projects = Split-Path -Parent (Split-Path -Parent $Root)
        $file = Join-Path $projects "mobile app\learnut\shared\src\commonMain\kotlin\com\app\tagacuyo\config\AppConfig.kt"
    } catch { return "" }
    if (-not (Test-Path $file)) { return "" }
    $found = [regex]::Match((Get-Content $file -Raw), 'https://[a-z0-9]+\.supabase\.co')
    if ($found.Success) { return $found.Value }
    return ""
}

function New-Settings {
    Write-Step "One-time question: switch the Learnut app to this PC?"
    Write-Host "    While this window is open, this script can point Learnut's translator at this PC."
    Write-Host "    Learnut has one production Supabase project, so that switches EVERY Learnut user,"
    Write-Host "    testers included. Ctrl+C switches them back to the translator they use now."
    Write-Host ""
    $settings = [pscustomobject]@{ url = ""; key = ""; fallbackUrl = "" }
    $answer = Read-Host "    Switch Learnut to this PC while it runs? [y/N]"
    if ($answer -notmatch '^\s*(y|yes)\s*$') {
        Save-Settings $settings
        Write-Info "OK. Learnut is left alone. To be asked again, delete $SettingsFile"
        return $settings
    }
    $default = Find-LearnutSupabaseUrl
    $prompt = "    Supabase project URL"
    if ($default) { $prompt += " [$default]" }
    $url = (Read-Host $prompt).Trim()
    if (-not $url) { $url = $default }
    $url = $url.TrimEnd("/")
    if ($url -notmatch '^https://[a-z0-9]+\.supabase\.co$') {
        throw "'$url' is not a Supabase project URL (https://<project>.supabase.co)."
    }
    Write-Host "    Paste the secret key from Supabase > Project Settings > API Keys: an sb_secret_... key,"
    Write-Host "    or the legacy service_role key. It is saved encrypted, for your Windows user only."
    $key = (ConvertTo-PlainText (Read-Host "    Secret key" -AsSecureString)).Trim()
    if (-not $key) { throw "No key entered." }
    if ($key.StartsWith("sb_publishable_")) {
        throw "That is the publishable key. Copy the secret key instead (it starts with sb_secret_)."
    }
    if (-not ($key.StartsWith("sb_secret_") -or $key.StartsWith("eyJ"))) {
        throw "That doesn't look like a Supabase secret key: it should start with sb_secret_."
    }
    $settings.url = $url
    $settings.key = ConvertFrom-SecureString (ConvertTo-SecureString $key -AsPlainText -Force)
    $key = $null
    $value = Get-TranslatorConfig $settings   # proves the URL and key work before saving them
    Write-Info "Connected. Learnut's translator right now: $($value.get_api_translator)"
    Save-Settings $settings
    return $settings
}

function Publish-Url($Settings, [string]$Url) {
    $previous = Set-TranslatorUrl $Settings $Url
    # Remember the real server to switch back to (never one of our own tunnel URLs).
    if ($previous -and $previous -notmatch 'trycloudflare\.com') {
        $Settings | Add-Member -NotePropertyName "fallbackUrl" -NotePropertyValue $previous -Force
        Save-Settings $Settings
    }
    Write-Status "Learnut now uses $Url" Green
}

function Restore-Url($Settings) {
    if (-not $Settings.fallbackUrl) {
        Write-Warn "No earlier translator URL is saved, so Learnut still points at this PC."
        return
    }
    try {
        Set-TranslatorUrl $Settings $Settings.fallbackUrl | Out-Null
        Write-Status "Learnut switched back to $($Settings.fallbackUrl)" Green
    } catch {
        Write-Warn "Could not switch Learnut back to $($Settings.fallbackUrl): $($_.Exception.Message)"
    }
}

# ---- Main ---------------------------------------------------------------------------

if ($AddToStartup -or $RemoveFromStartup) {
    if ($RemoveFromStartup) {
        Remove-Item $StartupShortcut -ErrorAction SilentlyContinue
        Write-Host "Removed: the translator no longer starts when you sign in."
    } else {
        $link = (New-Object -ComObject WScript.Shell).CreateShortcut($StartupShortcut)
        $link.TargetPath = Join-Path $Root "start-translator.cmd"
        $link.WorkingDirectory = $Root
        $link.WindowStyle = 7   # minimized
        $link.Description = "Runs the Learnut translator on this PC"
        $link.Save()
        Write-Host "Added: the translator starts (minimized) every time you sign in to Windows."
        Write-Host "Shortcut: $StartupShortcut"
    }
    exit 0
}

$server = $null
$tunnel = $null
$settings = $null
$switched = $false
$failed = $false

try {
    Write-Host "Learnut translator" -ForegroundColor Green
    Write-Host "Project: $Root"
    Get-ChildItem $ServerDir -Filter "*.log" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item -Force -ErrorAction SilentlyContinue
    Disable-Sleep   # the first run downloads gigabytes; sleeping would stall it

    # 1. Tools and model
    $basePython = Get-BasePython
    $ollamaExe = Install-Ollama
    Set-OllamaKeepAlive
    Start-Ollama $ollamaExe
    $newAdapter = Sync-AdapterFromDownloads
    if ($Rebuild -or $newAdapter -or -not (Test-Path $Gguf)) {
        Build-Model $basePython $ollamaExe
    } elseif (-not (Test-OllamaModel)) {
        Register-Model $ollamaExe
    }
    $servePy = Initialize-ServeEnv $basePython
    $cloudflared = Get-Cloudflared

    # 2. Server
    Write-Step "Starting the translator server"
    $server = Start-Server $servePy
    $sample = Invoke-TestTranslation
    Write-Info "Test translation, Tagalog 'Saan ka pupunta?' -> Cuyonon: '$sample'"
    if ($SetupOnly) {
        Write-Host ""
        Write-Host "Setup finished. Double-click start-translator.cmd to run the translator." -ForegroundColor Green
        return
    }

    # 3. Learnut: ask once, before anything is switched, and remember the answer
    if (-not $NoSupabase) {
        $settings = Read-Settings
        if ($null -eq $settings) { $settings = New-Settings }
    }
    $useSupabase = [bool]($settings -and $settings.url)

    # 4. Public URL, the Learnut switch, then keep everything up until Ctrl+C. Every
    #    step is retried every 15 seconds, so starting before the internet is up
    #    (at sign-in, after sleep) only delays it.
    $announced = $false
    $publishedUrl = ""
    $tick = 0
    $misses = 0
    while ($true) {
        try {
            if (-not (Test-Http "$OllamaUrl/api/version")) {
                Write-Status "Ollama stopped. Starting it again." Yellow
                Start-Ollama $ollamaExe
            }
            if ($server.HasExited) {
                Write-Status "The server stopped. Starting it again." Yellow
                $server = Start-Server $servePy
                Write-Status "The server is back." Green
            }
            $renew = $false
            if ($null -eq $tunnel -or $tunnel.Process.HasExited) {
                $renew = $true
            } elseif ($tick % 4 -eq 0) {
                if (Test-Http "$($tunnel.Url)/health" 15) { $misses = 0 }
                else {
                    $misses++
                    Write-Status "The public URL did not answer ($misses of 3)." Yellow
                    if ($misses -ge 3) { $renew = $true }
                }
            }
            if ($renew) {
                if ($tunnel) { Stop-Child $tunnel.Process }
                $tunnel = $null
                Write-Status "Opening the public tunnel ..." Cyan
                $tunnel = Start-Tunnel $cloudflared
                $misses = 0
                Write-Status "Public URL: $($tunnel.Url)" Green
            }
            if ($useSupabase -and $tunnel -and $publishedUrl -ne $tunnel.Url) {
                Publish-Url $settings $tunnel.Url
                $publishedUrl = $tunnel.Url
                $switched = $true
            }
            if (-not $announced -and $tunnel) {
                Show-Banner $tunnel.Url $switched $settings
                $announced = $true
            }
        } catch {
            Write-Status "Problem: $($_.Exception.Message) Trying again in 15 seconds." Red
        }
        Start-Sleep -Seconds 15
        $tick++
    }
}
catch {
    $failed = $true
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Logs: $ServerDir"
}
finally {
    if ($tunnel) { Stop-Child $tunnel.Process }
    Stop-Server $server
    if ($switched) { Restore-Url $settings }
    Enable-Sleep
    if ($failed -and $PauseOnError) { [void](Read-Host "Press Enter to close") }
}
if ($failed) { exit 1 }
