VLLM_VERSION := 0.13.0
VLLM_TARBALL := vllm-$(VLLM_VERSION).tar.gz
VLLM_DIR := vllm-$(VLLM_VERSION)
VLLM_URL := https://github.com/vllm-project/vllm/releases/download/v$(VLLM_VERSION)/$(VLLM_TARBALL)

MODEL := mlx-community/Qwen3-0.6B-8bit
HOST := 0.0.0.0
PORT := 8000

.PHONY: help setup serve test-api clean

.DEFAULT_GOAL := help

help:
	@echo "vLLM Metal - comandi disponibili:"
	@echo ""
	@echo "  make setup      Installa dipendenze + vLLM da source"
	@echo "  make serve      Avvia il server ($(MODEL))"
	@echo "  make test-api   Testa il server con curl"
	@echo "  make clean      Rimuovi file temporanei"
	@echo ""
	@echo "Opzioni:"
	@echo "  MODEL=...       Modello HuggingFace (default: $(MODEL))"
	@echo "  PORT=...        Porta del server (default: $(PORT))"
	@echo ""
	@echo "NOTA: non usare 'uv sync' direttamente, usa sempre 'make setup'."
	@echo "      uv sync cancella vLLM che va installato da source (no wheel macOS)."

setup:
	uv sync
	curl -OL $(VLLM_URL)
	tar xf $(VLLM_TARBALL)
	uv pip install -r $(VLLM_DIR)/requirements/cpu.txt --index-strategy unsafe-best-match
	uv pip install $(VLLM_DIR)/
	rm -rf $(VLLM_DIR) $(VLLM_TARBALL)
	@echo ""
	@echo "Setup completato. Usa 'make serve' per avviare il server."

serve:
	@uv run python -c "import vllm" 2>/dev/null || (echo "Errore: vLLM non installato. Lancia 'make setup' prima." && exit 1)
	uv run vllm serve $(MODEL) --host $(HOST) --port $(PORT)

test-api:
	@curl -s http://localhost:$(PORT)/health > /dev/null 2>&1 || (echo "Errore: server non raggiungibile su porta $(PORT). Lancia 'make serve' prima." && exit 1)
	curl http://localhost:$(PORT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL)", "messages": [{"role": "user", "content": "Ciao!"}]}'

clean:
	rm -rf $(VLLM_DIR) $(VLLM_TARBALL)
