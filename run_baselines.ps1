# Entrena Sultani, RTFM y UR-DMU (3 clases) con 5 semillas sobre las features I3D.
# Ejecutar desde la raíz de MIA:  .\run_baselines.ps1
# Si se corta, vuelve a ejecutarlo: salta las corridas que ya tienen metrics.json.

$man  = "data/splits/manifest_3class_v2.csv"
$feat = "data/processed/i3d_320x240_10crop"
$ann  = "data/annotations/Temporal_Anomaly_Annotation.txt"
$out  = "artifacts/runs_baselines"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null

foreach ($m in "sultani", "rtfm", "urdmu") {
    foreach ($s in 1..5) {
        if (Test-Path "$out/$m/seed$s/metrics.json") {
            Write-Host "[$m seed $s] ya existe, se salta" -ForegroundColor Yellow
            continue
        }
        Write-Host "=== [$m seed $s] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Cyan
        python baselines_mil/train_baseline.py --model $m --manifest $man --feat_dir $feat `
            --out_dir "$out/$m" --seed $s --annotations $ann 2>&1 | Tee-Object -FilePath "$out/logs/${m}_seed$s.log"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[$m seed $s] terminó con error, revisa $out/logs/${m}_seed$s.log" -ForegroundColor Red
        }
    }
}
Write-Host "Listo: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
