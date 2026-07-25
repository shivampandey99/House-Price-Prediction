# House Price Prediction Using Linear Regression

## Project Overview

This project builds a Machine Learning model to predict house prices using area-level economic and housing features from the USA Housing Dataset.

A Linear Regression model is trained and evaluated using R², MAE, and RMSE. The project also compares prediction performance across Low, Medium, and High population areas.


## Problem Statement

A real-estate platform wants to provide sellers with an instant, data-driven price estimate for a house based on area-level economic and housing indicators.


## Business Objective

The objective of this project is to build a Linear Regression model that can predict house prices accurately and consistently, helping a real-estate platform provide data-driven price estimates instead of relying only on manual valuation.


## Dataset

**Dataset Name:** USA Housing Dataset  
**Source:** Kaggle  
**Dataset Author:** vedavyasv  
**Dataset Link:** https://www.kaggle.com/datasets/vedavyasv/usa-housing
The dataset contains the following main features:

- Avg. Area Income
- Avg. Area House Age
- Avg. Area Number of Rooms
- Avg. Area Number of Bedrooms
- Area Population
- Price
- Address


## Technologies and Libraries Used

- Python
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn


## Project Workflow

1. Data Loading and Understanding
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Train-Test Split
6. Feature Scaling
7. Linear Regression Model Training
8. Model Prediction
9. Model Evaluation
10. Population Category Performance Comparison


## Data Cleaning

The following data cleaning steps were performed:

- Checked for missing values.
- Checked for duplicate records.
- Removed the free-text `Address` column.
- Renamed columns for easier coding.
- Checked for invalid Price and Population values.

No missing values, duplicate records, or invalid Price/Population values were found.


## Exploratory Data Analysis

### Price Distribution

The Price distribution was approximately symmetric, with a skewness value of approximately **-0.003**. Therefore, a log transformation of Price was not required.

![Price Distribution](Images/price_distribution.png)

### Correlation Analysis

A correlation heatmap was used to examine the relationships among the numerical features and house Price.

![Correlation Heatmap](Images/correlation_heatmap.png)


## Feature Engineering

A new feature called `rooms_per_bedroom` was created using the average number of rooms and bedrooms.

Area Population was also divided into three approximately equal categories:

- Low Population
- Medium Population
- High Population

These categories were used to compare model prediction performance across different population-density groups.


## Model Building

The numerical features were separated from the target variable (`Price`).

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

The input features were standardized using `StandardScaler`, and a Linear Regression model was trained on the training data.


## Model Evaluation

The final Linear Regression model achieved the following performance on the test data:

| Metric | Result |
|---|---:|
| R² Score | 0.918 |
| MAE | 80,881.07 |
| RMSE | 100,448.49 |

The R² score indicates that the model explains approximately **91.8% of the variation in house prices** in the test data.

### Actual vs Predicted Prices

The following plot compares the actual house prices with the prices predicted by the model.

![Actual vs Predicted Prices](Images/actual_vs_predicted.png)


## Population Category Performance Comparison

To evaluate whether the model performs consistently across different types of areas, the test predictions were grouped into Low, Medium, and High population categories.

| Population Category | Test Records | MAE |
|---|---:|---:|
| Low | 323 | 80,626.65 |
| Medium | 338 | 80,982.43 |
| High | 339 | 81,022.41 |

![MAE Across Population Categories](Images/population_category_mae.png)

The MAE values are very similar across all three population categories. This indicates that the model performs consistently across Low, Medium, and High population areas.


## Conclusion

The Linear Regression model performed well on the USA Housing Dataset, achieving an R² score of approximately **0.918**.

The Actual vs Predicted analysis showed that most predictions were reasonably close to the actual house prices. The category-wise analysis also showed similar prediction errors across Low, Medium, and High population areas.

Overall, the model provides a useful baseline for generating consistent, data-driven house price estimates using area-level economic and housing indicators.


## Repository Structure

```text
AIML-Project-RollNo-2302221530101/
├── Dataset/
│   └── USA_Housing.csv
├── Notebook/
│   └── House_Price_Prediction.ipynb
├── Images/
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── population_category_mae.png
└── README.md
```