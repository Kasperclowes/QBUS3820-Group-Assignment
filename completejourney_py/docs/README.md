# Documentation Development

This directory contains the source files for the Complete Journey Python package documentation.

## Setup

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

## Local Development

Serve the documentation locally:

```bash
mkdocs serve
```

Then visit http://127.0.0.1:8000

## Building

Build the documentation:

```bash
mkdocs build
```

The built site will be in the `site/` directory.

## Structure

```
docs/
├── index.md                    # Homepage
├── user-guide/
│   ├── getting-started.ipynb   # Basic usage
│   ├── datasets.ipynb         # Dataset documentation  
│   └── installation.md        # Installation guide
├── cookbook/                   # Analysis examples
│   ├── top-products.ipynb      # Top selling products
│   ├── shopping-frequency.ipynb # Shopping patterns
│   ├── coupon-analysis.ipynb   # Coupon redemption
│   └── traffic-patterns.ipynb  # Traffic analysis
├── api/
│   └── get_data.md            # API reference
└── about/
    ├── changelog.md           # Version history
    └── license.md            # License information
```

## Notebook Guidelines

For cookbook notebooks:

- Include clear explanations and insights
- Use realistic analysis scenarios
- Provide actionable business insights
- Include proper error handling examples