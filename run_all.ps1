param(
    [string]$Python = ".venv\\Scripts\\python.exe"
)

$configs = @(
    "configs\\lorenz63_lstm.json",
    "configs\\duffing_transformer.json",
    "configs\\lorenz96_encoder_decoder.json"
)

foreach ($config in $configs) {
    Write-Host "Running $config"
    & $Python -m chaos_ml.cli --config $config
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed on $config with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Write-Host "All runs completed."
