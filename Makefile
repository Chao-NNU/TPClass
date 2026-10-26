.PHONY: help install install-dev install-gpu install-esm download-models test lint format type-check split train-phase1 train-phase2 run-hmo run-squalene run-isobutanol docs clean clean-temp

help:
	@echo "TPClass Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install TPClass in editable mode (CPU)"
	@echo "  install-dev      Install with dev dependencies"
	@echo "  install-gpu      Install with GPU dependencies (faiss-gpu)"
	@echo "  install-esm      Install with fair-esm (needed for the .pt checkpoints)"
	@echo "  download-models  Download all required models and reference data"
	@echo ""
	@echo "Test & lint:"
	@echo "  test             Run pytest"
	@echo "  lint             Run flake8 linter"
	@echo "  format           Run black formatter"
	@echo "  type-check       Run mypy type checker"
	@echo ""
	@echo "Training:"
	@echo "  split            Split a TCDB FASTA into train/val/test"
	@echo "  train-phase1     Train the Phase 1 MLP + XGBoost models"
	@echo "  train-phase2     Train the Phase 2 contrastive head"
	@echo ""
	@echo "Run case studies:"
	@echo "  run-hmo         Run HMO transporter mining"
	@echo "  run-squalene    Run squalene transporter discovery in K. marxianus"
	@echo "  run-isobutanol  Run isobutanol exporter mining"
	@echo ""
	@echo "Clean:"
	@echo "  clean            Remove build artifacts"
	@echo "  clean-temp       Remove temp files in /tmp"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[gpu]"

install-esm:
	pip install -e ".[esm]"

download-models:
	bash models/download.sh

test:
	pytest tests/ -v --tb=short

lint:
	flake8 tpclass/ tests/ --max-line-length=100 --extend-ignore=E203,W503

format:
	black tpclass/ tests/

type-check:
	mypy tpclass/ --ignore-missing-imports

split:
	tpclass split --input $(INPUT) --out_dir $(OUT_DIR) --strategy cluster

train-phase1:
	tpclass train-phase1 --data_dir $(DATA_DIR) --out_dir $(OUT_DIR)

train-phase2:
	tpclass train-phase2 --data_csv $(DATA_CSV) --esm2 $(ESM2) --out_dir $(OUT_DIR)

run-hmo:
	bash examples/hmo/run_hmo.sh

run-squalene:
	bash examples/squalene/run_squalene.sh

run-isobutanol:
	bash examples/isobutanol/run_isobutanol.sh

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-temp:
	find /tmp -maxdepth 1 -name "top_candidates_*" -delete 2>/dev/null || true
	find /tmp -maxdepth 1 -name "blast_out_*" -delete 2>/dev/null || true
	find /tmp -maxdepth 1 -name "ref_db_*" -delete 2>/dev/null || true
