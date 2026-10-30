# ─────────────────────────────────────────────────────────────────────────────
# Experimentos v2: baseline + variantes, 5 semillas cada uno.
# Uso (desde la raíz del repo, con el entorno `anomalia` activado):
#     .\run_experimentos.ps1
# Para correr solo algunos:  .\run_experimentos.ps1 -Runs cnn_v2,farneback_v2
# ─────────────────────────────────────────────────────────────────────────────
param(
    [string[]]$Runs = @("cnn_v2", "farneback_v2", "sinbrillo_v2", "clip_v2"),
    [int[]]$Seeds = @(1, 2, 3, 4, 5)
)
$ErrorActionPreference = "Stop"

$Configs = @{
    "cnn_v2"       = "configs/hr320_cnn_v2.yaml"        # BASELINE: ConvNeXt-S + ConvNeXt-T sobre |Δframe|
    "farneback_v2" = "configs/hr320_farneback_v2.yaml"  # Variante 1: movimiento con Farnebäck
    "sinbrillo_v2" = "configs/hr320_sinbrillo_v2.yaml"  # Variante 2: sin filtro de brillo
    "clip_v2"      = "configs/hr320_clip_v2.yaml"       # Opcional: CLIP en apariencia (modelo final)
}

foreach ($run in $Runs) {
    $cfg = $Configs[$run]
    Write-Host "`n==================== $run  ($cfg) ====================" -ForegroundColor Cyan

    # La variante sin brillo necesita sus propios splits
    if ($run -eq "sinbrillo_v2") {
        vad --config $cfg preprocess
        if ($LASTEXITCODE -ne 0) { throw "preprocess falló en $run" }
    }

    # Extrae solo los .npz que falten (reanudable)
    vad --config $cfg extract
    if ($LASTEXITCODE -ne 0) { throw "extract falló en $run" }

    # Baseline mínimo (logística sobre las mismas features) → artifacts/seeds/<run>/baseline_logistica/
    python baseline_logistica.py --config $cfg --run $run
    if ($LASTEXITCODE -ne 0) { throw "baseline_logistica falló en $run" }

    foreach ($s in $Seeds) {
        $ck = "artifacts/seeds/$run/s$s"
        Write-Host "--- $run · semilla $s ---" -ForegroundColor Yellow
        vad --config $cfg train --set train.seed=$s --set paths.checkpoints_dir=$ck
        if ($LASTEXITCODE -ne 0) { throw "train falló en $run s$s" }
        vad --config $cfg evaluate --set paths.checkpoints_dir=$ck --set paths.reports_dir=$ck/reports
        if ($LASTEXITCODE -ne 0) { throw "evaluate falló en $run s$s" }
    }
}

Write-Host "`nListo. Resultados en artifacts/seeds/<run>/s<semilla>/reports/" -ForegroundColor Green
Write-Host "Tabla comparativa: abre notebooks/03_experimentos.ipynb y Run All"
