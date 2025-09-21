param(
  [string]$AppSource = "",             # chemin complet vers ton app.py personnalisé
  [switch]$SkipIngestion = $false,     # ajoute -SkipIngestion pour sauter ingestion/index
  [switch]$UseVenvBinaries = $true     # force l'usage des binaires du venv (.venv\Scripts\...)
)

# --- Utils ---
function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# --- Project root ---
$ProjectRoot = (Get-Location).Path
Write-Info "Project root = $ProjectRoot"

# --- Paths ---
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvPip    = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
$VenvStreamlit = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"
$ReqFile    = Join-Path $ProjectRoot "requirements.txt"
$AppTarget  = Join-Path $ProjectRoot "src\ui\app.py"
$IngestScript = Join-Path $ProjectRoot "src\ingestion\pipeline.py"
$IndexScript  = Join-Path $ProjectRoot "src\indexing\build_index.py"

# --- Checks ---
if (-not (Test-Path $VenvPython)) {
  Write-Err "Python venv non trouvé: $VenvPython"
  Write-Host "Crée le venv puis réessaie :"
  Write-Host "  python -m venv .venv"
  Write-Host "  .\.venv\Scripts\Activate.ps1"
  exit 1
}
Write-Ok "Venv Python trouvé."

if (-not (Test-Path $ReqFile)) {
  Write-Warn "requirements.txt introuvable. Je continue, mais pense à l’ajouter."
} else {
  Write-Info "Installation des dépendances…"
  & $VenvPip install --upgrade pip
  & $VenvPip install -r $ReqFile
  if ($LASTEXITCODE -ne 0) { Write-Err "pip install a échoué."; exit 1 }
  Write-Ok "Dépendances installées."
}

# --- Copier app.py personnalisé si fourni ---
if ($AppSource -and (Test-Path $AppSource)) {
  Write-Info "Copie de l’app personnalisée: $AppSource -> $AppTarget"
  Copy-Item -Path $AppSource -Destination $AppTarget -Force
  Write-Ok "app.py mis à jour."
} else {
  if ($AppSource) {
    Write-Warn "AppSource fourni mais introuvable: $AppSource"
  } else {
    Write-Warn "Aucun AppSource fourni. (Le placeholder de src\\ui\\app.py sera utilisé s’il existe.)"
  }
}

# --- Ingestion + Index (optionnel) ---
if (-not $SkipIngestion) {
  if (-not (Test-Path $IngestScript)) {
    Write-Warn "Script d’ingestion introuvable: $IngestScript (je passe l’étape)."
  } else {
    Write-Info "Exécution ingestion (pipeline.py)…"
    & $VenvPython $IngestScript
    if ($LASTEXITCODE -ne 0) { Write-Err "Ingestion a échoué."; exit 1 }
    Write-Ok "Ingestion OK."
  }

  if (-not (Test-Path $IndexScript)) {
    Write-Warn "Script d’indexation introuvable: $IndexScript (je passe l’étape)."
  } else {
    Write-Info "Construction de l’index (build_index.py)…"
    & $VenvPython $IndexScript
    if ($LASTEXITCODE -ne 0) { Write-Err "Indexation a échoué."; exit 1 }
    Write-Ok "Indexation OK."
  }
} else {
  Write-Info "SkipIngestion = true → ingestion/index non exécutés."
}

# --- Lancement Streamlit ---
if (Test-Path $VenvStreamlit -and $UseVenvBinaries) {
  Write-Info "Démarrage Streamlit via venv…"
  & $VenvStreamlit run $AppTarget
} else {
  Write-Info "Démarrage Streamlit via PATH global…"
  streamlit run $AppTarget
}
