export PYTHONPATH := src

.PHONY: eval test

eval:
	python -m harness.cli --config configs/default.yaml

test:
	python -m pytest -v