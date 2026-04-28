.PHONY: install test lint security train validate clean

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev,notebook]"

test:
	python -m pytest --cov=us_housing_price_prediction --cov-report=term-missing --cov-fail-under=75

lint:
	python -m ruff check .

security:
	python -m bandit -r src -c pyproject.toml
	python -m pip_audit -r requirements.txt

validate:
	python -m us_housing_price_prediction validate-data

train:
	python -m us_housing_price_prediction train

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'htmlcov']]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
