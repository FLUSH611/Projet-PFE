Write-Host "Python:"
python --version
Write-Host "Venv Python path:"
$p = ".\.venv\Scripts\python.exe"; if (Test-Path $p) { Write-Host (Resolve-Path $p) } else { Write-Host "not found" }
Write-Host "Folders:"
ls data, vectors, out, logs