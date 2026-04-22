# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python data package that provides access to the Complete Journey dataset from 84.51°, originally available as an R package. It serves as a Python equivalent using the more portable Parquet format instead of R datasets. The package provides comprehensive grocery transaction data covering 2,469 households over one year, including transaction records (1.47M), demographics, campaigns, coupons, and product information for retail analytics and data science research.

## Architecture

**Core Structure:**
- `completejourney_py/` - Main package directory
  - `__init__.py` - Exports the main `get_data` function
  - `get_data.py` - Single module containing data loading functionality
  - `data/` - Contains 8 Parquet files with the actual datasets

**Data Access Pattern:**
The entire package revolves around a single function `get_data()` that returns pandas DataFrames loaded from Parquet files. The function accepts a `which` parameter to specify which dataset(s) to load, returning a dictionary mapping dataset names to DataFrames.

**Available Datasets:**
- `transactions` - Complete purchase records (1,469,307 records) - Primary dataset
- `demographics` - Household demographic information (2,469 households)
- `products` - Product metadata and categories
- `campaigns` - Marketing campaigns received by households
- `campaign_descriptions` - Campaign metadata and details
- `promotions` - Product placement in mailers and stores
- `coupons` - Coupon metadata (UPC codes, campaigns)
- `coupon_redemptions` - Detailed coupon usage records

## Development Commands

**Testing:**
```bash
python setup.py test     # Run all tests
pytest                   # Alternative test runner
pytest tests/test_get_data.py  # Run specific test file
```

**Linting:**
```bash
flake8                   # Code style checking (max line length: 120)
```

**Package Installation:**
```bash
pip install -e .         # Install in development mode
python setup.py install  # Standard installation
```

## Dependencies

**Required:**
- pandas >= 0.25.0
- pyarrow >= 0.11.0 (for Parquet file reading)

**Development:**
- pytest (testing)
- flake8 (linting)
- pytest-runner (setup integration)

## Code Patterns

**Data Loading Pattern:**
```python
from completejourney_py import get_data

# Load all datasets
data = get_data()

# Load specific dataset
transactions = get_data("transactions")["transactions"]

# Load multiple datasets
subset = get_data(["transactions", "demographics"])
```

**Common Analysis Patterns:**
```python
# Basic transaction analysis
data = get_data(["transactions", "demographics", "products"])
transactions = data["transactions"]

# Household spending analysis
household_spending = transactions.groupby('household_id')['sales_value'].sum()

# Campaign effectiveness analysis
campaign_data = get_data(["campaigns", "campaign_descriptions"])
```

**File Organization:**
- All data files are embedded as package data in `completejourney_py/data/`
- Uses modern `importlib.resources` for robust file path resolution
- Simple, single-module design focused on data access

## Testing Strategy

Comprehensive test suite with 9 test cases covering:
- **Parameter variations**: All input types (None, string, list, tuple, empty list)
- **Error handling**: Invalid dataset names, wrong parameter types, mixed scenarios
- **Data quality**: DataFrame validation, structure checks, data type verification
- **Edge cases**: Empty inputs, boundary conditions
- **Functionality**: All 8 datasets correctly loaded and validated

Tests ensure robust error handling and comprehensive coverage of all use cases.

## Dataset Column Reference

This section provides the complete, accurate column structure for all datasets. **Use these exact column names in all code and documentation.**

### 1. TRANSACTIONS (1,469,307 records)
Primary dataset containing individual product purchases.

| Column | Data Type | Description | Example Values |
|--------|-----------|-------------|----------------|
| `household_id` | int64 | Household identifier | 900, 1228, 906 |
| `store_id` | int64 | Store location identifier | 330, 406, 319 |
| `basket_id` | int64 | Unique shopping trip identifier | 31198570044, 31198570047 |
| `product_id` | int64 | Product identifier | 1095275, 9878513, 1041453 |
| `quantity` | int64 | Number of items purchased | 1, 2, 3 |
| `sales_value` | float64 | Dollar amount spent on product | 0.50, 0.99, 1.43 |
| `retail_disc` | float64 | Retail discount applied | 0.0, 0.1, 0.15 |
| `coupon_disc` | float64 | Coupon discount applied | 0.0, 0.55, 2.0 |
| `coupon_match_disc` | float64 | Coupon match discount | 0.0, 0.45, 0.25 |
| `week` | int64 | Week number (1-52) | 1, 2, 3 |
| `transaction_timestamp` | datetime64[ns] | Date and time of purchase | 2017-01-01 11:53:26 |

### 2. DEMOGRAPHICS (801 records)
Household demographic and characteristic information.

| Column | Data Type | Description | Example Values |
|--------|-----------|-------------|----------------|
| `household_id` | int64 | Household identifier | 1, 1001, 1003 |
| `age` | object | Age range of household head | "65+", "45-54", "35-44" |
| `income` | object | Household income range | "35-49K", "50-74K", "25-34K" |
| `home_ownership` | object | Home ownership status | "Homeowner", "Probable Renter", "Renter" |
| `marital_status` | object | Marital status | "Married", "Unmarried" |
| `household_size` | object | Number of people in household | "2", "1", "4" |
| `household_comp` | object | Household composition | "2 Adults No Kids", "1 Adult No Kids" |
| `kids_count` | object | Number of children | "0", "2", "3+" |

### 3. PRODUCTS (92,331 records)
Product catalog with categories and brand information.

| Column | Data Type | Description | Example Values |
|--------|-----------|-------------|----------------|
| `product_id` | int64 | Product identifier | 25671, 26081, 26093 |
| `manufacturer_id` | int64 | Manufacturer identifier | 2, 69, 16 |
| `department` | object | Store department | "GROCERY", "PASTRY", "DRUG GM" |
| `brand` | object | Brand type | "National", "Private" |
| `product_category` | object | Product category | "FRZN ICE", "BREAD", "FRUIT - SHELF STABLE" |
| `product_type` | object | Specific product type | "ICE - CRUSHED/CUBED", "APPLE SAUCE" |
| `package_size` | object | Package size information | "22 LB", "50 OZ", "14 OZ" |

### 4. CAMPAIGNS (6,589 records)
Marketing campaigns received by households.

| Column | Data Type | Description | Example Values |
|--------|-----------|-------------|----------------|
| `campaign_id` | int64 | Campaign identifier | 1, 10, 11 |
| `household_id` | int64 | Household identifier | 105, 1238, 1258 |

### 5. CAMPAIGN_DESCRIPTIONS (27 records)
Campaign metadata and details.

| Column | Data Type | Description | Example Values |
|--------|-----------|-------------|----------------|
| `campaign_id` | int64 | Campaign identifier | 1, 2, 3 |
| `campaign_type` | object | Campaign type | "Type B", "Type C", "Type A" |
| `start_date` | datetime64[ns] | Campaign start date | 2017-03-03, 2017-03-08 |
| `end_date` | datetime64[ns] | Campaign end date | 2017-04-09, 2017-05-08 |

### 6. COUPONS (116,204 records)
Coupon metadata with UPC codes and campaigns.

| Column | Data Type | Description | Example Values |
|--------|-----------|-------------|----------------|
| `coupon_upc` | int64 | Coupon UPC identifier | 10000085207, 10000085319 |
| `product_id` | int64 | Associated product | 9676830, 9676943, 9676944 |
| `campaign_id` | int64 | Associated campaign | 26, 27, 8 |

### 7. COUPON_REDEMPTIONS (2,102 records)
Individual coupon usage records.

| Column | Data Type | Description | Example Values |
|--------|-----------|-------------|----------------|
| `household_id` | int64 | Household identifier | 1029, 165, 712 |
| `coupon_upc` | int64 | Coupon UPC identifier | 51380041013, 51380041313 |
| `campaign_id` | int64 | Associated campaign | 26, 25, 27 |
| `redemption_date` | datetime64[ns] | Date of coupon use | 2017-01-01, 2017-01-03 |

### 8. PROMOTIONS (20,940,529 records)
Product placement and promotional activity data.

| Column | Data Type | Description | Example Values |
|--------|-----------|-------------|----------------|
| `product_id` | int64 | Product identifier | 1000050, 1000092, 1000106 |
| `store_id` | int64 | Store identifier | 316, 337, 441 |
| `display_location` | object | In-store display placement | "9", "3", "5" |
| `mailer_location` | object | Mailer placement code | "0", "A", "D" |
| `week` | int64 | Week of promotion | 1, 2, 3 |

## Common Data Joins

**Transaction Analysis with Demographics:**
```python
# Join transactions with household information
transaction_demo = transactions.merge(demographics, on='household_id', how='left')
```

**Product Analysis:**
```python
# Join transactions with product information
transaction_products = transactions.merge(
    products[['product_id', 'product_category', 'department']], 
    on='product_id', how='left'
)
```

**Campaign Analysis:**
```python
# Full campaign analysis
campaign_analysis = (campaigns
    .merge(campaign_descriptions, on='campaign_id')
    .merge(demographics, on='household_id')
)
```

**Coupon Analysis:**
```python
# Coupon redemption with product and household info
coupon_analysis = (coupon_redemptions
    .merge(coupons, on='coupon_upc')
    .merge(products, on='product_id')
    .merge(demographics, on='household_id')
)
```