# CS7050 Proposal Compliance Review

Source: `CS7050_Project_Proposal.pdf`

## Confirmed requirements

| Proposal requirement | Implementation status |
|---|---|
| 2015–2018 Lending Club records | Preparation filters `issue_date` to 2015–2018 |
| Approximately 100K records | Deterministic 100,000-row limit with seed 42 |
| Annual income, DTI, revolver utilization, FICO | Preserved in `dim_borrower` |
| Loan amount, interest rate, term, installment, grade | Preserved in `dim_loan`; grade was added to the baseline |
| Loan status and principal received | Preserved in `dim_outcome` and `fact_loan` |
| Star/snowflake data mart | SQLite star schema with four dimensions and one fact |
| OLAP slicing and dicing | Allowlisted grouping SQL by year, grade, and status |
| Decision-tree default classification | Implemented with `DecisionTreeClassifier` |
| Linear recovery / LGD analysis | Implemented for charged-off `total_rec_prncp` where sufficient variation exists |
| FastAPI RAG backend | Implemented in `lending_club/api.py` |
| RAG SQL benchmark | Documented sample set; offline tests execute every sample |
| Classification metrics | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and majority-class baseline |
| Recovery metrics | MAE, RMSE, R-squared, and training-mean baseline |

## Deviations and assumptions

- PostgreSQL is specified in the proposal, but SQLite is used as the local
  implementation and PostgreSQL is a supported architectural adaptation.
- NRP GPU and a customized LLM are optional/recommended proposal components;
  the implementation remains functional without GPU or downloaded models.
- The proposal mentions a linear regression model for expected recovery and
  LGD. This implementation estimates principal recovery from charged-off
  records using `total_rec_prncp`; it does not claim a complete LGD estimate
  because the Kaggle subset fields and proposal do not define a separate loss
  basis.
- The proposal's 40-query LLM benchmark is not available as an external model
  benchmark in this offline implementation. The equivalent submitted sample
  set is deterministic and tested end-to-end.
- No Kaggle data or credentials are present or downloaded by this project.
