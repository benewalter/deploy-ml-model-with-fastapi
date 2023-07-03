# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Benedikt Walter created the model. It is a logistic regression using the default hyperparameters in scikit-learn 1.2.2.

## Intended Use
This model should be used to predict salaries (salary classes) on publicly available Census Bureau data.

## Training Data
Publicly available census data on salaries was used for the training. The data can be found under https://archive.ics.uci.edu/dataset/20/census+income.

## Evaluation Data
Publicly available census data on salaries was used for the testing (same source as above). To do this testing, 20% of the data was not used for the model training. The evaluation was done on this subset of data.

## Metrics
Overall performance metrics:
- Precision: 0.72
- Recall: 0.27
- F1: 0.40
Information on the model performance on different slices of data can be found under https://github.com/benewalter/deploy-ml-model-with-fastapi/blob/master/starter/starter/slice_output.txt.


## Ethical Considerations
The model might potentially be biased if certain groups are under- or overrepresented. Before applying the model, make sure to conduct some additional analyses to check for potential ethical issues.

## Caveats and Recommendations
A very simple logistic regression model has been used. The performance can most certainly be improved by testing different algorithms and by using some additional features.
