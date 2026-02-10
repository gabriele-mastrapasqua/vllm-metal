# vLLM Metal - Setup & Run

## Con uv

vLLM non ha wheel per macOS, va installato da source:

```bash
# 1. Installa le dipendenze del plugin
uv sync

# 2. Scarica e installa vLLM 0.13.0 da source
curl -OL https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0.tar.gz
tar xf vllm-0.13.0.tar.gz
uv pip install -r vllm-0.13.0/requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install vllm-0.13.0/
rm -rf vllm-0.13.0 vllm-0.13.0.tar.gz

# 3. Avvia il server
uv run vllm serve mlx-community/Qwen3-0.6B-8bit --host 0.0.0.0 --port 8000
```

Oppure usa il Makefile:

```bash
make setup    # installa tutto
make serve    # avvia il server
make test-api # testa con curl
```

## Con install.sh

```bash
bash install.sh
source .venv-vllm-metal/bin/activate
vllm serve mlx-community/Qwen3-0.6B-8bit --host 0.0.0.0 --port 8000
```

## Test rapido

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-8bit",
    "messages": [{"role": "user", "content": "Ciao!"}]
  }'
```
