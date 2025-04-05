# Project Report: House Price Prediction Using Machine Learning

### Introduction

The primary objective of this project was to develop an accurate and reliable machine learning model to predict house prices. As a critical task in the real estate and financial sectors, house price prediction facilitates better decision-making for buyers, sellers, and investors. This project focused on designing and evaluating multiple machine learning models to achieve high predictive accuracy.

### Data Preprocessing

To ensure data quality and enhance the models' performance, the following preprocessing steps were undertaken:

- Verified that no missing or duplicate values existed in the dataset

  ```python
  # check the null values in our dataset
    print(num.isnull().sum().sum())
    print(cat.isnull().sum().sum())

  # check the duplicate valuse in our dataset
    duplicate_count = df.duplicated().sum()
    print(duplicate_count)
  ```

- Detected and removed outliers to minimize their impact on model training.

  ````python
  # Detected the outlieers
  df.query('parking == 3.0 & price > 12000000')

  # remove the outlier
  def outlierRemove(data):
    outlier = []
    q1,q3 = np.percentile(df['price'],[25,75])
    iqr = q3-q1
    upperLayer=q3+1.5*(iqr)
    lowerLayer=q1-1.5*(iqr)
    return lowerLayer,upperLayer
  lowerLayer,upperLayer = outlierRemove(df["price"])
  filterData=df[(df["price"]>=lowerLayer ) & (df["price"]<=upperLayer)]
   print("WithoutOutlier :",filterData)```

  ````

- Engineered features to extract useful information for prediction.

```python
    df['price_per_unit_area'] = (df['price'] / df['area']).round(3)
    df['bathroom_per_bedrooms'] = (df['bathrooms'] / df['bedrooms']).round(3)
    df['luxury_index'] = ((df['airconditioning'] + df['hotwaterheating'] + df['guestroom']) / 3 ).round(3)
```

- Encoded categorical variables into numerical formats.
  ```python
  orEnd = OrdinalEncoder()
  df["furnishingstatus"] = orEnd.fit_transform(df[["furnishingstatus"]])
  ```
- Scaled numerical variables for uniformity across feature ranges.
  ```python
  col_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'price_per_unit_area','price']
  scaler = StandardScaler()
  df[col_to_scale] = scaler.fit_transform(df[col_to_scale])
  ```

### Machine Learning Models Training

Foure machine learning models were trained to evaluate their performance on the house price prediction task: 1. Linear Regression 2. Ridge Regression 3. Random Forest Regressor 4. XGBoost Regressor

### Hyperparameter Tuning

GridSearchCV was applied to optimize the hyperparameters of the machine learning models, enabling the fine-tuning of their performance for improved accuracy.

```python
grid_search = GridSearchCV(estimator=lin_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)
```

### Evaluation Metrics

The models were evaluated using the following performance metrics:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Model Performance

The R² scores achieved by the models are as follows:

- Linear Regression: 82%
- Ridge Regression: 81%
- Random Forest Regressor: 95%
- XGBoost Regressor: 97%

The XGBoost Regressor demonstrated exceptional predictive power with an accuracy of 97%, making it the best-performing model. The Random Forest Regressor also achieved a commendable accuracy of 95%.

### **Conclusion**

This project successfully implemented and evaluated machine learning models for house price prediction. The preprocessing steps, feature engineering, and hyperparameter tuning contributed significantly to the models' accuracy. Among the evaluated models, the XGBoost Regressor emerged as the most reliable, achieving the highest accuracy of 97%.
