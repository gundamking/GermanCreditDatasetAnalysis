# Dataset Documentation

## German Credit Card Dataset

### Overview
The German Credit Card dataset is a well-known dataset used for credit risk assessment. It contains 1000 instances with 20 attributes describing various aspects of credit applicants.

### Dataset Files
- `german.data`: Original dataset with categorical variables
- `german.data-numeric`: Numeric version of the dataset
- `germandata.csv`: Processed CSV version
- `german.doc`: Original dataset documentation

### Variable Descriptions

#### Target Variable
- **target**: Credit risk (1 = Good, 2 = Bad)

#### Categorical Variables

1. **account_bal**: Status of existing checking account
   - A11: < 0 DM (negative balance)
   - A12: 0 <= ... < 200 DM
   - A13: >= 200 DM / salary assignments for at least 1 year
   - A14: no checking account

2. **payment_status**: Credit history
   - A30: no credits taken / all credits paid back duly
   - A31: all credits at this bank paid back duly
   - A32: existing credits paid back duly till now
   - A33: delay in paying off in the past
   - A34: critical account / other credits existing

3. **purpose**: Purpose of the credit
   - A40: car (new)
   - A41: car (used)
   - A42: furniture/equipment
   - A43: radio/television
   - A44: domestic appliances
   - A45: repairs
   - A46: education
   - A47: vacation
   - A48: retraining
   - A49: business
   - A410: others

4. **savings_bond_value**: Savings account/bonds
   - A61: < 100 DM
   - A62: 100 <= ... < 500 DM
   - A63: 500 <= ... < 1000 DM
   - A64: >= 1000 DM
   - A65: unknown / no savings account

5. **employed_since**: Present employment since
   - A71: unemployed
   - A72: < 1 year
   - A73: 1 <= ... < 4 years
   - A74: 4 <= ... < 7 years
   - A75: >= 7 years

6. **sex_marital**: Personal status and sex
   - A91: male : divorced/separated
   - A92: female : divorced/separated/married
   - A93: male : single
   - A94: male : married/widowed
   - A95: female : single

7. **guarantor**: Other debtors / guarantors
   - A101: none
   - A102: co-applicant
   - A103: guarantor

8. **most_valuable_asset**: Property
   - A121: real estate
   - A122: if not A121 : building society savings agreement / life insurance
   - A123: if not A121/A122 : car or other, not in attribute 6
   - A124: unknown / no property

9. **type_of_housing**: Other installment plans
   - A141: bank
   - A142: stores
   - A143: none

10. **housing**: Housing
    - A151: rent
    - A152: own
    - A153: for free

11. **job**: Job
    - A171: unemployed / unskilled - non-resident
    - A172: unskilled - resident
    - A173: skilled employee / official
    - A174: management / self-employed / highly qualified employee / officer

12. **telephone**: Telephone
    - A191: none
    - A192: yes, registered under the customers name

13. **foreign**: Foreign worker
    - A201: yes
    - A202: no

#### Numerical Variables

1. **duration**: Duration in months
2. **credit_amount**: Credit amount in DM
3. **installment_rate**: Installment rate in percentage of disposable income
4. **residence_since**: Present residence since
5. **age**: Age in years
6. **concurrent_credits**: Number of existing credits at this bank
7. **number_of_dependents**: Number of people being liable to provide maintenance for

### Cost Matrix
The dataset uses a specific cost matrix for evaluation:

```
      1        2
----------------------------
  1   0        1
-----------------------
  2   5        0
```

Where:
- 1 = Good credit
- 2 = Bad credit
- Rows represent actual classification
- Columns represent predicted classification

It is worse to classify a customer as good when they are bad (cost = 5) than to classify a customer as bad when they are good (cost = 1).

### Data Quality
- No missing values
- 1000 instances
- 20 features (7 numerical, 13 categorical)
- Balanced dataset with slight class imbalance

### Usage Notes
- The dataset is commonly used for credit risk assessment
- Feature engineering is important due to categorical variables
- Cost-sensitive learning should be considered due to the cost matrix
- Cross-validation is recommended for robust evaluation 