#Udacity Project 5 - Intro to Machine Learning Project
##Identify Fraud from Enron Email

1. This project is based on using supervised machine learning to predict if certain employees at Enron are fraudulent or not. The Enron dataset provided is a Python dictionary, which top-level keys are employee’s names. The value for that key is another dictionary, which contains several different features about the employee, such as their salary, total stock value, number of emails to and from them, and many others. Many of the feature fields are incomplete in this dataset and include values such as ‘NaN’ or NoneType Python objects. I found 2 outliers while doing analysis. One is ‘TOTAL’, which appears to be all of the datasets features combined together into one Python dictionary. The other is something that appears to be an organization called ‘THE TRAVEL AGENCY IN THE PARK’. I removed both of these from the dataset to avoid skewing my classification system’s results.

2. I used all of the features in the dataset except for each person’s email address.. I also created the following new features for the dataset based on existing features.

**Email Based Features**

| Feature Name | Description |
| ------------- | ------------- |
| to_poi_percentage  | The percentage of emails the person sent to a POI  |
| from_poi_percentage | The percentage of emails from a POI to a particular person |
| total_poi_percentage | The percentage of total emails from a POI and to a POI from a particular person |
| sqrt_total_poi_percentage | The square root value of the total_poi_percentage value |

I created the variables above because I thought that most fraudulent employees would be communicating between one another more frequently than to non-fraudulent employees.

**Financial Based Features**

| Feature Name | Description |
| ------------- | ------------- |
| total_employee_worth | The sum of a person’s salary, bonus, and total stock value |
| log_total_employee_worth | The log based transformation of the total_employee_worth variable |

I created these variables based on assumptions known about why Enron eventually went bankrupt and had so many fraudulent employees. I thought that fraudulent employees would most likely have higher sums of financial value than non-fraudulent employees, so I created the total_employee worth variable. I also transformed it using a natural log to see how much it would help the overall success of my classification system. Turns out it helped quite a bit along with the square root of the total_poi_percentage feature above. After running PCA, found that 2 of the features explained approximately 21% of the variance in the dataset, so I decided to limit the number of features to 5 initially. However, to be sure, I included a few other number of components in my final model to make sure I didn’t hinder its performance. Finally I decided to include feature scaling (using MinMaxScaler) in my classification Pipeline after PCA alone wasn’t yielding very high results. This helped improve precision and recall scores significantly.

3. My classification system is built with a sklearn pipeline, starting with feature scaling, then PCA, then a classifier. Grid search cross validation was used to tune parameters for each step of the pipeline. Below are the classification algorithms I tried:

- Random Forest
- Decision Tree
- Gaussian Naive-Bayes
- K Nearest Neighbors
- AdaBoost
- Linear SVM

When I first started testing, I used Grid Search cross-validation with the highest resulting F1 score for selecting the best algorithm. Precision score was quite high for each algorithm, especially with Gaussian Naive Bayes and Random Forest classification, however recall was never above approximately 28%.  Because of this, I chose to choose the best algorithm based on recall as the scoring metric. As a result, K Nearest Neighbors ended up being the most optimal algorithm. Below is the final pipeline that was chosen:

```python
Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('feat', PCA(copy=True, n_components=2, whiten=False)), ('classifier', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform'))])
```

4. Many machine learning algorithms have parameters that can be tuned to optimize the classification success of the algorithm. These parameters tend to be used to find a middle-ground between under and overfitting a classification model to a dataset. If tuning of these parameters is not done well, a model can be very variated when trying to predict new data, or highly biased, meaning that it basically ignores new data features being used for prediction. For my classification system, I chose to use grid search cross-validation to exhaustively search a list of particular parameters for each pipeline I created.

5. Validation is method of evaluating algorithms by splitting the dataset into training and testing sets. A classic mistake that can be made in validation is training and testing on the entire dataset. This can lead to algorithms to showing very high accuracy results when predicting values in an dataset, but being highly variated in prediction results of a new dataset. For my model, I chose to use stratified shuffle split cross-validation. This type of validation creates a random number of folds to split the training and testing set into, but keeps class distributions equal across each fold.

6. Below are the metrics of my classification model using tester.py.

| Evaluation Metric (rounded) | Score |
| ------------- | ------------- |
| Accuracy | .89 |
| Precision | .61 |
| Recall | .46 |
| F1 | .52 |

The two evaluation metrics I was most concerned with are precision and recall. The precision score shows how many predicted fraudulent employees are actually fraudulent. Alternatively, recall shows how many positive classifications the model returned. Based on the scores above, my algorithm classified actual fraudulent employees pretty well. However it also predicted many people that are actually innocent, as fraudulent.
