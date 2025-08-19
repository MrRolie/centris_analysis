# Centris Property Analysis

A comprehensive analysis toolkit for Quebec real estate properties scraped from Centris.ca, with a focus on identifying discounted plex investment opportunities.

## 🏠 Overview

This project analyzes Quebec plex properties to:
- **Identify discounted investment opportunities** using multi-criteria scoring
- **Build predictive models** for property pricing based on revenue and other factors
- **Visualize market patterns** and geographic distributions
- **Compare investment metrics** across different property types

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

## 📁 Project Structure

```
centris_analysis/
├── plex_analysis.ipynb          # Main analysis notebook
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
└── data/                        # Data directory (create locally)
    └── centris_comprehensive_plex_data.csv
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/centris_analysis.git
   cd centris_analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your data**
   - Place your `centris_comprehensive_plex_data.csv` file in the `data/` directory
   - The data should contain columns: price, gross_potential_income, units_count, etc.

4. **Run the analysis**
   ```bash
   jupyter notebook plex_analysis.ipynb
   ```

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

## 📋 Data Requirements

Your CSV file should include these columns:
- `price`: Property sale price
- `gross_potential_income`: Annual rental income
- `units_count`: Number of rental units
- `living_area_sqft`: Property living area
- `lot_area_sqft`: Lot size
- `year_built`: Construction year
- `category`: Property type (Duplex, Triplex, etc.)
- `lat`, `lng`: Geographic coordinates (optional, for mapping)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This analysis is for educational and research purposes only. Property investment decisions should be based on comprehensive due diligence and professional advice. The data and models provided are not financial advice.

## 📞 Contact

- **Project Link**: [https://github.com/YourUsername/centris_analysis](https://github.com/YourUsername/centris_analysis)
- **Data Source**: [Centris.ca](https://www.centris.ca/)

---

**Made with ❤️ for Quebec real estate investors**
