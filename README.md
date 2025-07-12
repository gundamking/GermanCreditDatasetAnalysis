# German Credit Card Risk Analysis

## Overview

This project performs comprehensive analysis and machine learning modeling on the German Credit Card dataset to predict credit risk. The analysis includes exploratory data analysis, feature engineering, and multiple machine learning models to classify whether a credit applicant is likely to default.

## Dataset Description

The German Credit Card dataset contains 1000 instances with 20 attributes (7 numerical, 13 categorical) describing various aspects of credit applicants:

- **Account Balance**: Status of existing checking account
- **Duration**: Duration of credit in months
- **Credit History**: Previous credit behavior
- **Purpose**: Purpose of the credit
- **Credit Amount**: Amount of credit requested
- **Savings**: Savings account/bonds status
- **Employment**: Present employment since
- **Installment Rate**: Installment rate in percentage of disposable income
- **Personal Status**: Personal status and sex
- **Other Debtors**: Other debtors/guarantors
- **Residence**: Present residence since
- **Property**: Most valuable property
- **Age**: Age in years
- **Other Plans**: Other installment plans
- **Housing**: Housing situation
- **Existing Credits**: Number of existing credits at this bank
- **Job**: Employment type
- **Dependents**: Number of people liable to provide maintenance
- **Telephone**: Telephone availability
- **Foreign Worker**: Foreign worker status

## Project Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── germancreditml.py                  # Main analysis script
├── German-Credit-Card-Input-Data-Only.ipynb  # Jupyter notebook
├── german.data                        # Original dataset
├── german.data-numeric                # Numeric version of dataset
├── germandata.csv                     # Processed CSV dataset
├── german.doc                         # Dataset documentation
├── GermanAcademicPaper.pdf            # Academic paper reference
├── images/                            # Generated visualizations
│   ├── correlation_heatmap.png
│   ├── credit_amount_duration.png
│   ├── boxplot.png
│   ├── scatterplot.png
│   └── ...
└── results/                           # Model results and metrics
```

## Features

### Data Analysis
- **Exploratory Data Analysis**: Comprehensive analysis of all features
- **Data Visualization**: Multiple plots including correlation heatmaps, scatter plots, and box plots
- **Feature Engineering**: Polynomial features and categorical encoding
- **Data Preprocessing**: Label encoding and one-hot encoding

### Machine Learning Models
- **Multiple Algorithms**: Comparison of various ML models
- **Model Evaluation**: Accuracy, precision, recall, F1-score, and AUC metrics
- **Cross-validation**: Robust model validation
- **Cost Matrix**: Implementation of domain-specific cost matrix for credit risk

### Key Insights
- Analysis of feature correlations with default risk
- Identification of most important predictive features
- Model performance comparison across different algorithms
- Business implications of false positive vs false negative predictions

## Installation and Usage

### Prerequisites
- Python 3.7+
- Required packages (see requirements.txt)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/gundamking/german-credit-analysis.git
cd german-credit-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python germancreditml.py
```

### Jupyter Notebook
For interactive analysis, open the Jupyter notebook:
```bash
jupyter notebook German-Credit-Card-Input-Data-Only.ipynb
```

## Results

The analysis provides:
- **Model Performance**: Comparison of different ML algorithms
- **Feature Importance**: Identification of key predictive variables
- **Risk Assessment**: Probability scores for credit default
- **Business Recommendations**: Insights for credit risk management

## Methodology

1. **Data Loading and Preprocessing**
   - Load German credit dataset
   - Handle missing values and data types
   - Encode categorical variables

2. **Exploratory Data Analysis**
   - Statistical summaries
   - Correlation analysis
   - Data visualization

3. **Feature Engineering**
   - Polynomial feature creation
   - Feature selection based on correlations
   - Dimensionality reduction

4. **Model Development**
   - Train-test split (80-20)
   - Multiple algorithm comparison
   - Hyperparameter tuning

5. **Model Evaluation**
   - Cross-validation
   - Performance metrics
   - Cost matrix analysis

## Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original dataset: Professor Dr. Hans Hofmann, Universität Hamburg
- Academic reference: GermanAcademicPaper.pdf
- UCI Machine Learning Repository for dataset availability

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. The models and analysis should not be used for actual credit decisions without proper validation and regulatory compliance. 