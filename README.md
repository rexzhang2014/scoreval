# scoreval
An visulized evaluation toolkit for model score. 
The model score will be evaluated static, on the full dataset. And further evaluate along date. 
This aims to help machine learning modelers to evaluate the result effictively and efficiently. 
## Effectiveness
1. static method: on the full eval data set.
    1. PR chart : average performance of the score.
    2. WOE/IV : indicative power of score binning. 
2. time method: metrics along date. 
    1. QTL daily : show stability of the score
    2. Precision daily: show target precision above a score cut-off.
    3. Recall daily: show target coverage above a score cut-off.

## Efficiency
1. Predefined process & metrics: No need to design and analysis on metrics.
2. Deal with multiple model scores: The tool will show charts for multiple model scores for comparison. 
# Input
The input of this package include the below parts:
- Model: a 'model' that implements the 'predict' method, it comes from most common used modeling tools like sklearn, tensorflow etc. The tool will definitely call model.predict() to get the score on the data set. 
- Data set: a data set containing at least 3 columns (row_id, date, label), for calculating the metrics and by dates. If date does not appear, the time method will throw error during run. 

# Output
Pack of charts to evaluate and understand your model. See the link for more details:https://github.com/rexzhang2014/scoreval/blob/main/tests/score-eval-example.ipynb


