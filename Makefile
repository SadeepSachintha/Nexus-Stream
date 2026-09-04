.PHONY: help install train api dashboard simulate processor test docker-up docker-down clean

PYTHON = .venv/Scripts/python
STREAMLIT = .venv/Scripts/streamlit
PYTEST = .venv/Scripts/pytest

help:
	@echo "Nexus Stream — Production Real-Time Fraud Engine"
	@echo "------------------------------------------------"
	@echo "  make install     : Install all dependencies into virtual environment"
	@echo "  make train       : Synthesize data & train cost-sensitive XGBoost model"
	@echo "  make api         : Start FastAPI inference microservice"
	@echo "  make dashboard   : Launch Streamlit analyst portal"
	@echo "  make simulate    : Run synthetic transaction generator (local mode)"
	@echo "  make processor   : Run streaming state enrichment processor"
	@echo "  make test        : Execute unit and integration tests"
	@echo "  make docker-up   : Launch full distributed multi-container cluster"
	@echo "  make docker-down : Tear down Docker container cluster"
	@echo "  make clean       : Clear local databases and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.inference.train

api:
	$(PYTHON) -m uvicorn src.inference.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(STREAMLIT) run src/dashboard/app.py --server.port 8501

simulate:
	$(PYTHON) -m src.simulator.generate_transactions --rate 50 --target api

processor:
	$(PYTHON) -m src.streaming.processor --local

test:
	$(PYTEST) tests/ -v

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

clean:
	rm -rf *.duckdb *.duckdb.wal .pytest_cache __pycache__ src/**/__pycache__
