FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

COPY data ./data

ENTRYPOINT ["python", "-m", "us_housing_price_prediction"]
CMD ["validate-data"]
