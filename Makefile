.PHONY: install test lint train validate clean

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev,notebook]"

test:
	python -m pytest

lint:
	python -m ruff check .

validate:
	python -m us_housing_price_prediction validate-data

train:
	python -m us_housing_price_prediction train

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'htmlcov']]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
