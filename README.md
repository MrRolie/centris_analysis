# Centris Property Analysis - Educational Repository

A comprehensive analysis toolkit for Quebec real estate properties scraped from Centris.ca, with a focus on identifying discounted plex investment opportunities.

**📚 Educational Purpose Notice**: This repository is shared for educational purposes to demonstrate real estate data analysis methodologies and findings. The underlying data is proprietary and not included. This serves as a showcase of analytical approaches, code structure, and insights derived from Quebec real estate market analysis.

## 🏠 Overview

This project demonstrates how to analyze Quebec plex properties to:
- **Identify discounted investment opportunities** using multi-criteria scoring
- **Build predictive models** for property pricing based on revenue and other factors
- **Visualize market patterns** and geographic distributions
- **Compare investment metrics** across different property types

## 🎯 Repository Purpose

This repository serves to:
- **Share analytical methodologies** used in real estate investment analysis
- **Demonstrate code structure** for scalable data science projects
- **Present findings and insights** from Quebec plex market analysis
- **Provide reusable frameworks** for similar real estate analyses
- **Showcase best practices** in data preparation, modeling, and visualization

**Note**: The actual data files are not included due to proprietary restrictions. The code structure and methodology can be adapted for similar datasets.

## 📊 Key Features

### Discount Identification System
- Multi-factor scoring based on revenue yield, price per unit, and market comparisons
- Benchmarking against property type and location averages
- Identification of potentially undervalued properties

### Predictive Modeling
- **Simple Linear Regression**: Revenue → Price relationship
- **Enhanced Multi-feature Model**: Includes property characteristics, location, age, etc.
- **Random Forest**: Advanced ensemble model for better accuracy
- Model comparison and validation metrics

### Comprehensive Analysis
- Revenue yield calculations and distributions
- Price per unit and price per square foot analysis
- Geographic visualization with interactive maps
- Correlation analysis of key property features

## 🛠️ Technology Stack

- **Python 3.8+**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
- **Mapping**: plotly mapbox

## ⚙️ Setup and Configuration

### 1. Environment Setup
Copy the example environment file and configure your data paths:

```bash
cp .env.example .env
```

Edit `.env` to set your data file paths:

```bash
# For macOS users
PLEX_CSV_PATH_MACOS=/Users/YourUsername/path/to/centris_comprehensive_plex_data.csv

# For Windows users  
PLEX_CSV_PATH_WINDOWS=C:\Users\YourUsername\path\to\centris_comprehensive_plex_data.csv
```

### 2. Configuration Management
The project uses a centralized configuration system that automatically:
- Detects your operating system (macOS/Windows/Linux)
- Loads the appropriate data path from `.env`
- Provides fallback options for common locations
- Manages project settings (target year, growth rates, random seeds)

Test your configuration:
```python
from utils.config import print_config
print_config()
```

## 📁 Project Structure

```
centris_analysis/
├── plex/                        # Plex analysis notebooks
│   ├── plex_modeling_elasticnet_gbt.ipynb  # ElasticNet vs XGBoost models
│   ├── plex_modeling_stepwise.ipynb       # Stepwise regression analysis
│   └── plex_data_exploration.ipynb        # Data exploration
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── data_preparation.py     # Data preparation functions
│   └── stepAIC.py              # Stepwise regression utilities
├── .env                        # Environment configuration (create from .env.example)
├── .env.example               # Example environment file
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore patterns
└── data/                       # Data directory (create locally)
    └── centris_comprehensive_plex_data.csv
```

## 🚀 Getting Started (For Educational Review)

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter step-criterion
```

### Exploring the Methodology

1. **Clone the repository**
   ```bash
   git clone https://github.com/MrRolie/centris_analysis.git
   cd centris_analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Review the analysis structure**
   - Examine the notebooks in `plex/` to understand the analytical approach
   - Review `utils/` modules for data preparation and configuration management
   - Study the methodology documented in each notebook

4. **Adapt for your own data** (if available)
   - Create your own `.env` file based on `.env.example`
   - Place your data file in the expected location
   - Run the notebooks with your dataset

**Note**: Without the proprietary data, the notebooks serve as educational examples of the analytical process and code structure.

## 📈 Key Metrics & Insights

### Discount Scoring Criteria
- **High Revenue Yield**: Properties above 75th percentile
- **Low Price per Unit**: Properties below 25th percentile
- **Market Comparison**: Below-median pricing metrics

### Model Performance
- Simple Revenue Model: Baseline R² score
- Enhanced Multi-feature Model: Improved accuracy with additional factors
- Random Forest: Best performance with ensemble approach

### Investment Recommendations
- Target properties with >7% revenue yield
- Focus on discount scores ≥3
- Consider model-identified undervalued properties

## 🗺️ Geographic Analysis

Interactive maps showing:
- Property distribution across Quebec
- Revenue yield by location
- Discount opportunities by region
- Price trends and market patterns

## 📊 Visualizations

The analysis includes:
- **Distribution plots** for prices, yields, and property characteristics
- **Correlation heatmaps** showing relationships between features
- **Interactive maps** with property locations and metrics
- **Model performance charts** comparing different approaches
- **Scatter plots** with regression lines and residual analysis

## 🔍 Model Features

### Simple Model
- Gross Potential Income → Price

### Enhanced Model
- Gross Potential Income
- Number of Units
- Living Area (sq ft)
- Lot Area (sq ft)
- Property Age
- Property Category
- Location (Municipality)

## 📋 Data Requirements (For Reference)

If adapting this methodology for your own real estate analysis, your CSV file should include these columns:
- `price`: Property sale price
- `gross_potential_income`: Annual rental income
- `units_count`: Number of rental units
- `living_area_sqft`: Property living area
- `lot_area_sqft`: Lot size
- `year_built`: Construction year
- `category`: Property type (Duplex, Triplex, etc.)
- `lat`, `lng`: Geographic coordinates (optional, for mapping)
- `building_style`: Architectural style
- `assessment_total`: Municipal assessment value
- `assessment_year`: Year of assessment

## 📚 Educational Value

This repository demonstrates:
- **Data pipeline architecture** for real estate analysis
- **Feature engineering techniques** for property valuation
- **Multiple modeling approaches** (Linear, Elastic Net, XGBoost, Stepwise)
- **Cross-platform configuration management**
- **Visualization strategies** for real estate data
- **Investment opportunity identification methods**

## 🔬 Analytical Methodologies Showcased

1. **Data Preparation**: Standardized preprocessing pipeline
2. **Feature Engineering**: Revenue yield, price ratios, geographic features
3. **Exploratory Analysis**: Distribution analysis, correlation studies
4. **Predictive Modeling**: Multiple algorithms with performance comparison
5. **Investment Scoring**: Multi-criteria opportunity identification
6. **Validation Techniques**: Cross-validation, residual analysis
7. **Visualization**: Interactive maps, statistical plots, dashboards

## 🤝 Contributing

This repository welcomes contributions that improve the educational value:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/MethodologyImprovement`)
3. Commit your changes (`git commit -m 'Add improved analysis technique'`)
4. Push to the branch (`git push origin feature/MethodologyImprovement`)
5. Open a Pull Request

Contributions could include:
- Enhanced analytical techniques
- Additional visualization methods
- Improved code documentation
- Performance optimizations
- Extended feature engineering approaches

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Educational and Research Purposes Only**: This analysis is shared for educational purposes to demonstrate real estate data analysis methodologies. The underlying proprietary data is not included. Property investment decisions should be based on comprehensive due diligence and professional advice. The methodologies and code provided are not financial advice.

## 📞 Contact

- **Project Repository**: [https://github.com/MrRolie/centris_analysis](https://github.com/MrRolie/centris_analysis)
- **Data Source Reference**: [Centris.ca](https://www.centris.ca/) (data not included)
- **Purpose**: Educational demonstration of real estate analysis methodologies

---

**Made with ❤️ for educational purposes and Quebec real estate analysis methodology sharing**
