## Summary
This project is implemented with modification from the `Probability of Default` white paper published by the National University of Singapore. The intention is to build a binary classificaiton model to predict the credit default status given the relevant features identified

## Folder Structure:
Remark: files saved in **mlruns** folder (mlflow artifacts and loggings) are omitted in the following diagram
```
|-- config|
|   |-- catalog.yaml
|-- data
|   |-- processed
|       |-- processed_input.csv
|   |-- raw
|       |-- input.csv
|-- models
|   |-- lgbm_model.pkl
|-- notebooks
|   |-- scratchpad.ipynb
|-- output
|   |-- best_param.json
|   |-- feature_importance.png
|-- screenshots
|   |-- Screenshot 2024-03-01 at 1.27.31 PM.png
|   |-- Screenshot 2024-03-01 at 1.28.00 PM.png
|   |-- Screenshot 2024-03-01 at 11.38.45 PM.png
|-- src
|   |-- utils
|       |-- data_schema.py
|       |-- general_utility_functions.py
|   |-- __init__.py
|   |-- data_processing.py
|   |-- extract_financial_data.py
|   |-- hyperparameter_tuning.py
|   |-- main.py
|   |-- model_pipeline.py
|-- tests
|   |-- __init__.py
|   |-- test_data_processing.py
|   |-- test_extract_financial_data.py
|   |-- test_model_pipeline.py
|-- .gitignore
|-- .pre-commit-config.yaml
|-- poetry.lock
|-- pyproject.toml
|-- README.md
```

## Exploratory Analysis / Variable Profiling:
<img src="./screenshots/Screenshot%202024-03-01%20at%201.27.31 PM.png" alt="Image1 alt text" width="400"/> <img src="./screenshots/Screenshot%202024-03-01%20at%201.28.00 PM.png" alt="Image2 alt text" width="400"/>

The relevant code can be found in the scratchpad.ipynb or profiling_report.py (streamlit)

## Model Inputs:

| Category | Attribute | Description |
| --- | --- | --- |
| Macro-Financial Factors | `Stock Index Return` | Trailing 1-year return of the primary stock market, winsorized and currency adjusted |
| Macro-Financial Factors | `Short-term Risk-Free Rate` | Yield on 3 month government bills |
| Macro-Financial Factors | `Economy-level Distance-To-Default for financial firms` | Median Distance-to-Default of financial firms in each economy inclusive of those foreign firms whose primary stock exchange is in this economy (Not applicable to China) |
| Macro-Financial Factors | `Economy-level Distance-To-Default for non-financial firms` | Median Distance-to-Default of non-financial firms in each economy inclusive of those foreign firms whose primary stock exchange is in this economy (Not applicable to China) |
| Firm-Specific Attributes | `Distance-to-Default (level)` | Volatility-adjusted leverage based on Merton (1974) with special treatments |
| Firm-Specific Attributes | `Distance-to-Default (trend)` | Volatility-adjusted leverage based on Merton (1974) with special treatments |
| Firm-Specific Attributes | `Cash/Total Assets (level)` | For financial firm’s liquidity - Logarithm of the ratio of each firm’s sum of cash and short-term investments to total assets |
| Firm-Specific Attributes | `Cash/Total Assets (trend)` | For financial firm’s liquidity - Logarithm of the ratio of each firm’s sum of cash and short-term investments to total assets |
| Firm-Specific Attributes | `Current Assets/Current Liabilities (level)` | For non-financial firm’s liquidity - Logarithm of the ratio of each firm’s current assets to current liabilities |
| Firm-Specific Attributes | `Current Assets/Current Liabilities (trend)` | For non-financial firm’s liquidity - Logarithm of the ratio of each firm’s current assets to current liabilities |
| Firm-Specific Attributes | `Net Income/Total Assets (level)` | Profitability - Ratio of each firm’s net income to total assets |
| Firm-Specific Attributes | `Net Income/Total Assets (trend)` | Profitability - Ratio of each firm’s net income to total assets |
| Firm-Specific Attributes | `Relative Size (level)` | Logarithm of the ratio of each firm’s market capitalization to the economy’s median market capitalization over the past one year |
| Firm-Specific Attributes | `Relative Size (trend)` | Logarithm of the ratio of each firm’s market capitalization to the economy’s median market capitalization over the past one year |
| Firm-Specific Attributes | `Relative Market-to-Book Ratio` | Individual firm’s market misvaluation/ future growth opportunities relative to the economy’s median level of market-to-book ratio |
| Firm-Specific Attributes | `Idiosyncratic Volatility` | 1-year idiosyncratic volatility of each firm, computed as the standard deviation of its residuals using the market model |


## High Level Workflow Diagram:
**Remark**: 
- Certain trivial functions are omitted to save space
- Hyperparameter tuning through Optuna are done within the `train_model` function
- For more details, please look at the docstring within each class and function

```mermaid
flowchart LR
subgraph Workflow
    direction LR
    subgraph Data_Processing
        direction TB
        filter_data_by_date -->one_hot_encode_categorical_columns -->winsorize_numerical_columns -->min_max_scale_numerical_columns -->add_auxliary_data
    end
    subgraph Modelling_Pipeline
        direction TB
        naive_timeseries_splitting -->remove_unwanted_features -->train_model -->create_model_with_best_params -->eval_model_performance -->get_feature_importance
    end
end

Data_Processing -->Modelling_Pipeline
```

## Processing and Modelling Logic:
1. <span style="background-color: #FFFF00">Schema validation</span> on the input data using **Pandera**
   - Checks if the input columns are of the <span style="background-color: #FFFF00">correct dtypes</span>
   - <span style="background-color: #FFFF00">Coerce</span> if they are not, e.g., Date is stored as string instead of datetime/timestamp 

2. Data Extraction from Yahoo Finance:
   - `get_data` downloads the pricing data into a pandas dataframe
   - `fill_missing_dates` creates a dataframe using the min and max of the input data as the range parameters. Merge with the pricing data fetched to fill dates which no transaction occurred.
   - `calculate_returns` Do a forward for the closing prices of the dates filled and then calculate the daily return
   - `extraction_flow` connects the preceding functions together in a flow

3. Data Processing class function:
   - `filter_data_by_date` filters out data prior to 2000s as <span style="background-color: #FFFF00">recency bias</span> is expected. It can also reduce the risk of data (distribution) drift.
   - `one_hot_encode_categorical_columns` applys encoding to the categorical variable industry classification code using <span style="background-color: #FFFF00">pd.get_dummies</span>
   - `winsorize_numerical_columns` loops through all the column names, if they are of numeric in nature, apply <span style="background-color: #FFFF00">winsorization</span>
   - `min_max_scale_numerical_columns` applys min max scaling by initializing the MinMaxScaler from sklearn.preprocessing. It should not matter for variables that are already encoded.
   - `fetch_auxiliary_data` fetches the <span style="background-color: #FFFF00">Oil (BZ=F)</span> closing price from yfinance API
   - `add_auxiliary_data` merges the processed input dataframe with the Oil pricing data from Yahoo Finance
   - `process_flow` <span style="background-color: #FFFF00">links all the preceding steps</span> together in a flow
  
4. Model Pipeline class function:
  

## Tuning results on validation set:
<img src="./screenshots/Screenshot 2024-03-01 at 11.38.45 PM.png" width="600"/> 

## Model Evaluation:
Best combination of parameters: ```{"lambda_l1": 2.936790772949318, "lambda_l2": 6.397468964536458, "num_leaves": 209, "feature_fraction": 0.680577513115211, "bagging_fraction": 0.407971418466264, "bagging_freq": 5, "min_child_samples": 38, "learning_rate": 0.03502279961404488, "max_depth": 10, "min_split_gain": 0.33635970049343217, "scale_pos_weight": 44.782368871281435, "n_estimators": 566}```

Validation set:
If the tuning objective is to **maximize pr_auc**: ```{'accuracy': 0.9196385190943543, 'f1': 0.9243469102689122, 'pr_auc': 0.35868479625382493, 'roc_auc': 0.8734528473928967}```
Test set: 
If the tuning objective is to **maximize pr_auc**: ```{'accuracy': 0.9871449305411569, 'f1': 0.9894485295679961, 'pr_auc': 0.04393246207745849, 'roc_auc': 0.8005253623188406}```

## Feature importance ranking:
<img src="./output/feature_importance.png" width="600"/> 