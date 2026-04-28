# Data Card

## Dataset

`data/raw/housing_price_data.csv`

The file contains 545 housing records with city, area, bedroom count, toilet count, story count, renovation status, and sale price.

## Schema

| Column | Canonical name | Type | Notes |
| --- | --- | --- | --- |
| House ID | house_id | integer | Identifier only, excluded from model features |
| City | city | category | Encoded inside the pipeline |
| House Area (sqm) | house_area_sqm | float | Must be positive |
| No. of Bedrooms | bedrooms | integer | Must be positive |
| No. of Toilets | toilets | integer | Must be positive |
| Stories | stories | integer | Must be positive |
| Renovation Status | renovation_status | category | furnished, semi-furnished, or unfurnished |
| Price ($) | price_usd | integer | Regression target |

## Quality Checks

The package validates required columns, numeric ranges, duplicate IDs, missing values, and accepted renovation statuses.

## Limitations

The dataset is small and synthetic or classroom-sized. It should be treated as an educational dataset unless provenance, collection process, and representativeness are verified.
