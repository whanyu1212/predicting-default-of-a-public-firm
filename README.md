# end-to-end-credit-default-prediction
Default Prediction for public firms

## Summary
This project is implemented with modification from the `Probability of Default` white paper published by the National University of Singapore. The intention is to build a binary classificaiton model to predict the credit default status given the relevant features identified

## Model Inputs:

#### Macro-Financial Factors:
- `Stock Index Return`: Trailing 1-year return of the primary stock market, winsorized and currency adjusted
- `Short-term Risk-Free Rate`: Yield on 3 month government bills
- `Economy-level Distance-To-Default for financial firms` and `Economy-level Distance-To-Default for non-financial firms`: Median Distance-to-Default of financial/non-financial firms in each economy inclusive of those foreign firms whose primary stock exchange is in this economy (Not applicable to China) 

#### Firm-Specific Attributes:
