# Corre `vad localize` sobre cada semilla ya entrenada, para tener el tIoU de las cinco semillas.
# Ejecutar desde la raíz de MIA:   .\run_localize.ps1 -Runs cnn_v2, cnn_v2_hpo
# Salta las semillas que ya tienen reports\localization_test\summary.csv.

param(
    [string[]]$Runs = @("cnn_v2", "farneback_v2", "sinbrillo_v2", "clip_v2", "cnnclip_v2"),
    [switch]$PlotAll            # agrega --plot-all (un gráfico por video; tarda más)
)

$cfgs = @{
    "cnn_v2"       = "configs/hr320_cnn_v2.yaml"
    "farneback_v2" = "configs/hr320_farneback_v2.yaml"
    "sinbrillo_v2" = "configs/hr320_sinbrillo_v2.yaml"
    "clip_v2"      = "configs/hr320_clip_v2.yaml"
    "cnnclip_v2"   = "configs/hr320_cnnclip_v2.yaml"
    "cnn_v2_hpo"   = "configs/hr320_cnn_v2_hpo.yaml"
}

foreach ($r in $Runs) {
    $cfg = $cfgs[$r]
    if (-not $cfg -or -not (Test-Path $cfg)) { Write-Host "[$r] no encuentro la config $cfg, se salta" -ForegroundColor Yellow; continue }
    $seeds = Get-ChildItem "artifacts/seeds/$r" -Directory -Filter "s*" -ErrorAction SilentlyContinue
    if (-not $seeds) { Write-Host "[$r] no hay semillas en artifacts/seeds/$r" -ForegroundColor Yellow; continue }
    foreach ($sd in $seeds) {
        $ck = "artifacts/seeds/$r/$($sd.Name)"
        if (Test-Path "$ck/reports/localization_test/summary.csv") { Write-Host "[$r $($sd.Name)] ya tiene localización, se salta"; continue }
        Write-Host "=== [$r $($sd.Name)] $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
        $extra = @(); if ($PlotAll) { $extra += "--plot-all" }
        vad --config $cfg localize --set paths.checkpoints_dir=$ck --set paths.reports_dir=$ck/reports @extra
        if ($LASTEXITCODE -ne 0) { Write-Host "[$r $($sd.Name)] terminó con error" -ForegroundColor Red }
    }
}
Write-Host "Listo. El notebook 03 (sección 6d) resume el tIoU por modelo y semilla." -ForegroundColor Green
