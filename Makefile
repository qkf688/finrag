PY ?= python

setup:
	$(PY) -m pip install -e .
	$(PY) -m pip install pytest ruff

model:
	$(PY) -m pip install -e ".[model]"

serve-setup:
	$(PY) -m pip install -e ".[serve]"

serve:
	$(PY) -m uvicorn finrag.serve:app --port 8000

bench-setup:
	$(PY) -m pip install -e ".[bench]"

financebench:
	$(PY) -m finrag.cli financebench --download

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

check-corpus:
	$(PY) -m finrag.cli check-corpus

eval:
	$(PY) -m finrag.cli eval --out results/eval-hash.json

demo:
	$(PY) -m finrag.cli demo "基金的业绩比较基准是什么"
