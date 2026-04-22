# Installation

This page provides detailed installation instructions for the Complete Journey Python package.

## Requirements

- **Python**: 3.8-3.14
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM recommended for loading all datasets
- **Storage**: Approximately 200MB for package and data

## Standard Installation

### Using pip (Recommended)

```bash
pip install completejourney_py
```

### Verify Installation

```python
# Test the installation
from completejourney_py import get_data

# Load a small dataset to verify
demographics = get_data("demographics")
print(f"✅ Successfully loaded {len(demographics['demographics'])} household records")
```

## Development Installation

For contributors or users who want the latest features:

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/cunningjames/completejourney_py.git
cd completejourney_py

# Install in development mode
pip install -e .
```

### With Development Dependencies

```bash
# Install with all development tools
pip install -e ".[dev]"

# Or install specific groups
pip install -e ".[test]"  # Just testing dependencies
```

## Virtual Environment Setup

It's recommended to use a virtual environment:

### Using venv

```bash
# Create virtual environment
python -m venv completejourney_env

# Activate (Linux/Mac)
source completejourney_env/bin/activate

# Activate (Windows)
completejourney_env\Scripts\activate

# Install package
pip install completejourney_py
```

### Using conda

```bash
# Create conda environment
conda create -n completejourney python=3.9
conda activate completejourney

# Install package
pip install completejourney_py
```

## Troubleshooting

### Common Issues

#### ImportError: Missing pandas/pyarrow

```bash
# Install missing dependencies
pip install pandas>=1.0.0 pyarrow>=1.0.0
```

#### Memory Issues with Large Datasets

```python
# Load datasets selectively
import pandas as pd

# Configure pandas for memory efficiency
pd.set_option('display.precision', 2)

# Load only needed datasets
data = get_data(["transactions"])  # Instead of all datasets
```

#### Permission Errors

```bash
# Install for current user only
pip install --user completejourney_py
```

### Python Version Issues

Check your Python version:

```bash
python --version
# Should be 3.8 or higher (up to 3.14)
```

If you have multiple Python versions:

```bash
# Use specific Python version
python3.14 -m pip install completejourney_py
```

## Dependency Information

### Required Dependencies

- **pandas** (>=1.0.0): Data manipulation and analysis
- **pyarrow** (>=1.0.0): Efficient Parquet file reading

### Optional Dependencies

For development:

- **pytest** (>=6.0): Testing framework
- **black** (>=22.0): Code formatting
- **isort** (>=5.0): Import sorting
- **mypy** (>=0.910): Type checking
- **flake8** (>=4.0): Code linting

For documentation:

- **mkdocs-material**: Documentation site generator
- **mkdocs-jupyter**: Jupyter notebook integration

## Performance Optimization

### Memory Management

For large dataset analysis:

```python
# Monitor memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory before loading: {get_memory_usage():.1f} MB")
data = get_data(["transactions"])
print(f"Memory after loading: {get_memory_usage():.1f} MB")
```

### Storage Optimization

The package data is compressed efficiently:

```python
# Check package size
import pkg_resources
import os

package_path = pkg_resources.resource_filename('completejourney_py', '')
total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(package_path)
                for filename in filenames)

print(f"Package size: {total_size / 1024 / 1024:.1f} MB")
```

## Docker Setup

For containerized environments:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
RUN pip install completejourney_py

# Verify installation
RUN python -c "from completejourney_py import get_data; print('✅ Installation verified')"
```

## Jupyter Notebook Setup

For analysis in Jupyter:

```bash
# Install Jupyter
pip install jupyter

# Install plotting libraries
pip install matplotlib seaborn plotly

# Start Jupyter
jupyter notebook
```

Test in a notebook cell:

```python
import pandas as pd
import matplotlib.pyplot as plt
from completejourney_py import get_data

# Load and visualize
data = get_data(["transactions"])
transactions = data["transactions"]

# Quick visualization
daily_sales = transactions.groupby('day')['sales_value'].sum()
daily_sales.plot(title='Daily Sales Over Time')
plt.show()
```

## Next Steps

After installation:

1. **[Getting Started](../user-guide/getting-started.ipynb)** - Learn basic usage
2. **[Dataset Overview](../user-guide/datasets.ipynb)** - Understand the data structure  
3. **[Cookbook Examples](../cookbook/top-products.ipynb)** - Follow analysis tutorials
4. **[API Reference](../api/get_data.md)** - Detailed function documentation

## Getting Help

If you encounter installation issues:

1. Check the [GitHub Issues](https://github.com/cunningjames/completejourney_py/issues)
2. Verify your Python version and dependencies
3. Try installation in a fresh virtual environment
4. Report bugs with your Python version, OS, and error messages