param(
  [string]$ServerExe = ".\llama-server.exe",
  [int]$Port = 8080
)
if (-not (Test-Path $ServerExe)) {
  Write-Host "llama-server.exe not found: $ServerExe" -ForegroundColor Red
  exit 1
}
& $ServerExe `
  -hf "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF" `
  --alias "qwen2.5-vl-3b" `
  --host "127.0.0.1" `
  --port $Port `
  -ngl 99 `
  -c 4096
