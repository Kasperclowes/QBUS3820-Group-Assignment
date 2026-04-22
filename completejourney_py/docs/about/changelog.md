# Changelog

All notable changes to the Complete Journey Python package are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-11

### 🎉 Major Documentation & CI/CD Release

This release represents a significant enhancement from a basic data package to a comprehensive, production-ready toolkit for retail analytics education and exploration.

### Added

**📚 Complete Documentation Ecosystem**

- Comprehensive MkDocs documentation with Material theme
- **7 Analysis Cookbooks** with detailed business insights:
  - Dataset Summary Analysis: High-level overview of all 8 datasets
  - Top Selling Products: Product performance and ranking analysis  
  - Shopping Frequency Analysis: Customer behavior and visit patterns
  - Coupon Analysis: Promotional effectiveness and redemption patterns
  - Traffic Patterns: Store visit timing and seasonal trends
  - Demographic Product Analysis: Purchase behavior by income/age/family structure
  - Market Basket Analysis: Product associations and cross-selling opportunities
- Interactive Jupyter notebooks with visualizations and strategic recommendations
- User guide with getting started tutorials and dataset exploration
- Complete API reference documentation
- Professional about section with changelog and license

**🔧 Production CI/CD Infrastructure**

- GitHub Actions workflows for automated testing across Python 3.8-3.11
- Automated documentation deployment to GitHub Pages
- Package building and installation verification
- Code quality checks with flake8 linting
- Test coverage reporting with Codecov integration
- Dependabot configuration for automated dependency updates

**📊 Enhanced Package Metadata**

- Professional README with GitHub Actions status badges
- Comprehensive usage examples and business use cases
- Links to live documentation and cookbook examples
- CLAUDE.md guidance for AI-assisted development

### Changed
- **Documentation Structure**: Streamlined user-guide/datasets.ipynb to focus on data structure overview
- **Version Badges**: Updated all documentation to reflect new version
- **Package Description**: Enhanced with comprehensive business applications and use cases

### Technical Improvements
- **Coverage**: 10,501+ lines of new documentation and analysis content
- **Quality Assurance**: All PRs must pass comprehensive testing pipeline
- **Automation**: Documentation automatically updates with code changes
- **Professional Standards**: GitHub Actions status badges show project health

### Business Value
- **Educational Content**: 7 comprehensive analysis notebooks with real insights
- **Strategic Applications**: Business recommendations for marketing, merchandising, and operations
- **Multiple Approaches**: Different analysis frameworks with experimentation guidance
- **Production Ready**: Complete CI/CD pipeline ensures reliability

## [0.0.3] - 2025-01-10

### Added
- Modern `pyproject.toml` configuration following PEP 621 standards
- Comprehensive type hints throughout the codebase
- Enhanced test suite with 9 test cases covering all functionality
- Professional README with badges and comprehensive documentation
- Comprehensive .gitignore following Python best practices

### Changed
- **BREAKING**: Replaced deprecated `pkg_resources` with modern `importlib.resources`
- Updated Python requirement to >=3.8 (from >=3.6)
- Enhanced function docstring with detailed examples and parameter descriptions
- Modernized development workflow with black, isort, mypy integration

### Removed
- Legacy `setup.py` and `setup.cfg` files
- Deprecated `pytest-runner` dependency

### Fixed
- Eliminated deprecation warnings from `pkg_resources`
- Improved error handling and edge case coverage

## [0.0.2] - Previous Release

### Added
- Fixed missing dependency on pyarrow

### Changed
- Improved package metadata

## [0.0.1] - Initial Release

### Added
- Initial package structure with `get_data()` function
- Support for loading 8 Complete Journey datasets:
  - `transactions`: 1.47M purchase records
  - `demographics`: Household characteristics
  - `products`: Product metadata
  - `campaigns`: Marketing campaigns
  - `campaign_descriptions`: Campaign metadata
  - `promotions`: Product placement data
  - `coupons`: Coupon information
  - `coupon_redemptions`: Coupon usage records
- Parquet format for efficient data storage and loading
- Basic documentation and usage examples

### Technical Details
- Python 3.6+ support
- pandas and pyarrow dependencies
- Package data included in distribution

---

## Contributors

- James Cunningham - Original package author
- Brad Boehmke - Package co-author and contributor
- Claude Code AI - Package modernization and documentation

## Data Attribution

This package provides Python access to the Complete Journey dataset:

**Original Data Source:**

- **Provider**: [84.51°](http://www.8451.com/area51/)
- **Original R Package**: [completejourney](https://github.com/bradleyboehmke/completejourney) by Bradley Boehmke
- **License**: Available for research and educational purposes

**Python Package:**

- **License**: MIT License
- **Repository**: [GitHub](https://github.com/cunningjames/completejourney_py)

---

*For detailed technical changes, see the [Git commit history](https://github.com/cunningjames/completejourney_py/commits/main).*