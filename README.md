# Credit Scoring Model Development

## Project Overview
This project focuses on building a robust credit scoring model for Bati Bank as part of a partnership with an eCommerce company to enable a Buy-Now-Pay-Later (BNPL) service. The goal is to categorize customers as high or low risk and predict optimal loan amounts and durations.

### Objectives:
1. **Understand Credit Risk**: Review key concepts and best practices for credit risk analysis.
2. **Exploratory Data Analysis (EDA)**: Analyze and visualize the provided data to extract actionable insights.
3. **Feature Engineering**: Create predictive features from raw data.
4. **Model Development**: Build and evaluate machine learning models to assign risk probabilities and credit scores.
5. **API Deployment**: Serve the trained model via a REST API for real-time predictions.

## Project Structure
```
project-folder/
├── README.md              # Project overview and instructions.
├── notebooks/             # Jupyter notebooks for analysis and modeling.
│   ├── eda.ipynb          # EDA progress.
│   ├── feature_engineering.ipynb # Initial feature engineering.
├── scripts/               # Python scripts for tasks like data processing.
│   ├── data_cleaning.py   # Script for data cleaning.
│   ├── feature_engineering.py  # Script for feature engineering.
├── data/                  # Folder to hold datasets (if allowed).
│   ├── raw/               # Original/raw data files.
│   ├── processed/         # Cleaned or transformed data files.
├── models/                # Trained models.
├── reports/               # Reports and deliverables.
│   ├── interim_report.pdf # Your interim submission.
├── requirements.txt       # Dependencies for the project.
├── .gitignore             # To exclude unnecessary files.
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Virtual environment tool (e.g., `venv` or `conda`)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project-folder
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: `env\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
- **Run EDA**: Open and execute `notebooks/eda.ipynb`.
- **Feature Engineering**: Use `notebooks/feature_engineering.ipynb` or run `scripts/feature_engineering.py`.
- **Data Cleaning**: Run `scripts/data_cleaning.py` for preprocessing raw data.

## Deliverables
- Interim report summarizing progress on:
  - Understanding credit risk.
  - Initial insights from EDA.
  - Early-stage feature engineering.

## Key References
- [Basel II Capital Accord](https://www.bis.org/publ/bcbs107.htm)
- [Weight of Evidence (WoE) and Information Value (IV)](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)

For detailed documentation on deliverables and deadlines, refer to the `reports/interim_report.pdf` file.

