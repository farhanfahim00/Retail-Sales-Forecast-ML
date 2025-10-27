# Retail Sales Demand Forecasting Dashboard

An end-to-end machine learning project that predicts retail product demand using historical sales data. Built with Python, Scikit-learn, and Streamlit for interactive visualization and prediction.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

---

## Project Overview

This project demonstrates a complete machine learning workflow from data cleaning to model deployment. The goal is to predict the number of units sold (demand) for retail products based on features like product type, pricing, discounts, customer segment, and geographic location.

### Key Features:
- **ETL Pipeline**: Automated data cleaning and preprocessing
- **Exploratory Data Analysis**: Interactive visualizations of sales patterns
- **ML Model**: Linear Regression for demand forecasting
- **Web Dashboard**: Streamlit app for real-time predictions
- **Model Evaluation**: Comprehensive performance metrics

---

## Project Structure
```
Retail-Sales-Forecast-ML/
├── data/
│   ├── raw/                    # Original dataset (not tracked in git)
│   └── processed/              # Cleaned data ready for modeling
│       └── processed.csv
├── models/
│   ├── demand_model.pkl        # Trained Linear Regression model
│   └── feature_names.pkl       # Feature names for prediction
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Initial data investigation
│   ├── 02_eda.ipynb                 # Exploratory Data Analysis
│   └── 03_model_training.ipynb      # Model development and training
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Project dependencies
├── .gitignore                 # Files to exclude from version control
└── README.md                  # Project documentation
```

---

## Tech Stack

### Core Technologies:
- **Python 3.11**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
- **Matplotlib & Seaborn**: Data visualization
- **Streamlit**: Interactive web dashboard

### Development Tools:
- **Jupyter Notebook**: Experimentation and prototyping
- **Git/GitHub**: Version control
- **VS Code**: Code editor

---

## Dataset

**Source**: [Kaggle - Financials Dataset](https://www.kaggle.com/datasets/atharvaarya25/financials/data)

**Description**: 
- 700 retail transaction records
- Time period: 2013-2014
- 6 products across 5 countries
- Features: Units Sold, Sale Price, Discounts, Manufacturing Price, Product Type, Customer Segment, Country, Discount Band

### Data Cleaning Steps:
1. Removed trailing spaces from column names
2. Converted currency strings to numeric values (`$1,234.56` → `1234.56`)
3. Handled 53 missing discount values (filled with 0 for "None" discount band)
4. Calculated 63 missing profit values using formula: `Profit = Sales - COGS`
5. Converted date columns to datetime format
6. Validated business logic (Sales = Gross Sales - Discounts)

---

## Exploratory Data Analysis

### Key Insights:

**1. Sales Distribution**
- Most transactions range from $0-$200K
- Few high-value outliers reaching $1M+
- Right-skewed distribution indicating typical small to medium orders

**2. Temporal Patterns**
- Clear seasonality with peaks in October 2014
- Cyclical sales patterns suggesting quarterly trends
- Overall upward growth trajectory

**3. Product Performance**
- **Paseo** leads with ~$31M total sales (3x competitors)
- Top 6 products account for majority of revenue
- Average demand per product: ~1,500 units

**4. Discount Impact**
- Surprising finding: Higher discounts don't significantly increase demand
- Low/No discount bands show higher average units sold (~1,650 units)
- Suggests pricing strategy may not be primary demand driver

**5. Feature Correlations**
- Sale Price: 0.81 correlation with Sales
- Discounts: 0.25 correlation with Units Sold
- Units Sold: 0.33 correlation with Sales
- Weak correlations indicate demand is influenced by factors beyond the dataset

---

## Machine Learning Model

### Model Selection: Linear Regression

**Why Linear Regression?**
- Interpretable: Clear understanding of feature impact
- Fast training: Suitable for small datasets
- Baseline model: Industry best practice to start simple
- Transparent coefficients: Shows which features drive predictions

### Features Used (20 total after encoding):
**Numeric Features:**
- Sale Price
- Manufacturing Price
- Discounts
- Month (for seasonality)

**Categorical Features (One-Hot Encoded):**
- Product (6 categories → 5 binary features)
- Segment (3 categories → 2 binary features)
- Country (5 categories → 4 binary features)
- Discount Band (4 categories → 3 binary features)

### Training Process:
1. **Train/Test Split**: 80% training (560 samples), 20% testing (140 samples)
2. **One-Hot Encoding**: Converted categorical variables to binary columns
3. **Model Training**: Fitted Linear Regression on training data
4. **Evaluation**: Tested on unseen data to measure generalization

---

## Model Performance

### Evaluation Metrics:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.185 | Model explains 18.5% of variance in demand |
| **MAE** | 610 units | Average prediction error of ~610 units |
| **RMSE** | 764 units | Root Mean Squared Error indicates some large outliers |

### Performance Analysis:

**Why is R² relatively low (0.185)?**

This is actually **realistic and valuable** for several reasons:

1. **Complex Real-World Problem**: Demand forecasting is inherently difficult because it depends on many external factors not captured in the dataset:
   - Marketing campaigns and advertising spend
   - Competitor pricing and promotions
   - Economic conditions and consumer confidence
   - Seasonal trends beyond simple month numbers
   - Supply chain constraints
   - Brand reputation and customer loyalty

2. **Limited Features**: The dataset contains only 8 core features. Professional demand forecasting models typically use 50-100+ features including:
   - Historical demand patterns (lag features)
   - External economic indicators
   - Weather data
   - Social media sentiment
   - Competitor data

3. **Honest Metrics**: Many online ML tutorials show R² > 0.95, but these often use:
   - Synthetic/toy datasets designed to be easy
   - Data leakage (using future information)
   - Overfitting on small test sets

4. **Better Than Baseline**: R² = 0.185 means the model performs **18.5% better** than simply predicting the average demand for every transaction. This is meaningful improvement!


**Key Insight**: The model learned that products with low or no discounts have higher predicted demand, which aligns with the EDA finding that discounts don't necessarily drive more units sold. This could indicate premium positioning or that discounts are applied reactively to slow-moving inventory.

---

## Streamlit Dashboard

### Interactive Features:

**1. Overview Page**
- Dataset statistics (700 records, 6 products, 5 countries)
- Sample data preview
- Summary statistics of key metrics

**2. Visualizations**
- Sales distribution histogram
- Sales trend over time (2013-2014)
- Top products by revenue
- Demand analysis by discount band

**3. Predict Demand**
- Interactive form with dropdowns and inputs
- Real-time prediction of units sold
- Estimated sales, revenue, and profit calculations
- Supports all product/segment/country combinations

**4. Model Performance**
- R², MAE, RMSE metrics display
- Feature importance visualization
- Model interpretation guide

---

## Installation & Usage

### Setup Instructions:

1. **Clone the repository**
```bash
git clone https://github.com/farhanfahim00/Retail-Sales-Forecast-ML.git
cd Retail-Sales-Forecast-ML
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Access the dashboard**
```
Open your browser and go to: http://localhost:8501
```

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest enhancements
- Submit pull requests
- Share feedback

---

## Contact

**Farhan Fahim**
- GitHub: [@farhanfahim00](https://github.com/farhanfahim00)
- LinkedIn: [[LinkedIn URL](https://www.linkedin.com/in/farhan-fahim/)]

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Acknowledgments

- Dataset source: [Kaggle - Atharvaarya25](https://www.kaggle.com/datasets/atharvaarya25/financials)
- Inspiration: Real-world demand forecasting challenges in retail
- Tools: Scikit-learn, Streamlit, and the Python data science community

