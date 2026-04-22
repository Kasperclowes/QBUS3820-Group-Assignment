# API Reference: get_data

The `get_data` function is the primary interface for loading Complete Journey datasets.

::: completejourney_py.get_data

## Usage Examples

### Load All Datasets

```python
from completejourney_py import get_data

# Load all 8 datasets
data = get_data()
print(f"Available datasets: {list(data.keys())}")
# Output: ['campaign_descriptions', 'campaigns', 'coupons', 'coupon_redemptions', 
#          'demographics', 'products', 'promotions', 'transactions']
```

### Load Single Dataset

```python
# Load only transactions
transactions_dict = get_data("transactions")
transactions = transactions_dict["transactions"]

print(f"Transactions shape: {transactions.shape}")
print(f"Columns: {list(transactions.columns)}")
```

### Load Multiple Datasets

```python
# Load specific datasets for sales analysis
sales_data = get_data(["transactions", "products", "demographics"])

transactions = sales_data["transactions"]
products = sales_data["products"]  
demographics = sales_data["demographics"]

# Join for comprehensive analysis
transaction_details = (transactions
    .merge(products, on='product_id')
    .merge(demographics, on='household_id'))
```

### Load with Different Input Types

```python
# Using list
data1 = get_data(["transactions", "demographics"])

# Using tuple  
data2 = get_data(("transactions", "demographics"))

# Using single string
data3 = get_data("transactions")

# Empty list returns empty dictionary
data4 = get_data([])
assert len(data4) == 0
```

## Available Datasets

| Dataset Name | Records | Description |
|--------------|---------|-------------|
| `transactions` | 1,469,307 | Individual product purchases with prices, quantities, and timing |
| `demographics` | 801 | Household characteristics including age, income, and composition |
| `products` | 92,353 | Product metadata with categories, brands, and descriptions |
| `campaigns` | 6,627 | Marketing campaigns received by households |
| `campaign_descriptions` | 27 | Campaign metadata and details |
| `promotions` | 20,940,529 | Product placement in mailers and stores |
| `coupons` | 116,364 | Coupon metadata with values and restrictions |
| `coupon_redemptions` | 2,318 | Individual coupon usage records |

## Error Handling

The function will raise exceptions for invalid inputs:

```python
# Invalid dataset name
try:
    get_data("invalid_dataset")
except FileNotFoundError:
    print("Dataset not found")

# Invalid parameter type
try:
    get_data(123)
except (TypeError, AttributeError):
    print("Invalid parameter type")

# Mixed valid/invalid names
try:
    get_data(["transactions", "invalid_dataset"])
except FileNotFoundError:
    print("One or more datasets not found")
```

## Return Value Details

The function always returns a dictionary where:

- **Keys**: Dataset names (strings)
- **Values**: pandas DataFrames containing the dataset
- **Structure**: Each DataFrame has appropriate column names and data types

```python
data = get_data(["transactions", "demographics"])

# Check return structure
assert isinstance(data, dict)
assert isinstance(data["transactions"], pd.DataFrame)
assert isinstance(data["demographics"], pd.DataFrame)

# Verify data quality
transactions = data["transactions"]
assert len(transactions) > 0  # Has data
assert len(transactions.columns) > 0  # Has columns
assert transactions.index is not None  # Has valid index
```

## Performance Considerations

- **Large datasets**: The `promotions` dataset is very large (20.9M records). Consider filtering after loading if memory is limited.
- **Selective loading**: Load only the datasets you need to optimize memory usage and loading time.
- **Caching**: Consider storing frequently used datasets locally for repeated analysis.

```python
# Memory-efficient approach for large analysis
# Load core datasets first
core_data = get_data(["transactions", "products", "demographics"])

# Load additional datasets only if needed
if analysis_requires_promotions:
    promotions = get_data("promotions")["promotions"]
```

## Common Patterns

### Basic Analysis Setup

```python
# Standard setup for transaction analysis
data = get_data(["transactions", "products", "demographics"])
transactions = data["transactions"]
products = data["products"]
demographics = data["demographics"]

# Create enriched transaction dataset
enriched_transactions = (transactions
    .merge(products[['product_id', 'commodity_desc', 'department']], on='product_id')
    .merge(demographics[['household_id', 'income_desc', 'household_size_desc']], on='household_id'))
```

### Campaign Analysis Setup

```python
# Setup for marketing campaign analysis
campaign_data = get_data([
    "campaigns", 
    "campaign_descriptions", 
    "demographics", 
    "transactions"
])

# Create campaign analysis dataset
campaign_analysis = (campaign_data["campaigns"]
    .merge(campaign_data["campaign_descriptions"], on='campaign')
    .merge(campaign_data["demographics"], on='household_id'))
```

### Coupon Analysis Setup

```python
# Setup for coupon effectiveness analysis
coupon_data = get_data([
    "coupons",
    "coupon_redemptions", 
    "products",
    "demographics"
])

# Create coupon analysis dataset
coupon_analysis = (coupon_data["coupon_redemptions"]
    .merge(coupon_data["coupons"], on='coupon_upc')
    .merge(coupon_data["products"], on='product_id')
    .merge(coupon_data["demographics"], on='household_id'))
```

## Data Source Information

All datasets are sourced from 84.51° "The Complete Journey" project:

- **Provider**: [84.51°](http://www.8451.com/area51/)
- **Format**: Parquet files for efficient loading and cross-platform compatibility
- **Scope**: One year of grocery shopping data from 801 households
- **License**: Available for research and educational purposes

## See Also

- [Dataset Overview](../user-guide/datasets.ipynb) - Detailed information about each dataset
- [Getting Started](../user-guide/getting-started.ipynb) - Basic usage patterns
- [Cookbook Examples](../cookbook/top-products.ipynb) - Step-by-step analysis tutorials