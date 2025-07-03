Our data is imbalanced, has missing values, and invalid bonce rates.

During our EDA we have so far created dataframes that we will use for training and evaluate which one creates the best performance.
We will use the dataframes in combinations with various classifer models that have been used during our Machine Learning course of 2025 and other techniques for dealing with missing values and imbalanced data.

The five models we will use are all from SKLearn:
1. LogisticRegression
2. DecisionTreeClassifier
3. RandomForestClassifier
4. AdaBoostClassifier
5. KNeighborsClassifier

Our evaluation metrics will be:
1. Accuracy
2. Precision
3. Recall
4. Model prediction speed. The final model will be used in real time in a ecommerce production environment in order to predict if users will convert or not. So the model has to be rapid!
IMPORTANT: We will in no part of the evaluation process consider F1. We want to capture as many converting users as possible, as fast as possible. So we will optimise for recall and latency. As a back up, in case this becomes relevant after talking to domain experts, we will also track Precision. In order to achieve this we will calcute a *F1-style* score that weights both Recall/Precision with latency.

Regarding speed: For each candidate model we record median prediction latency (milliseconds per sample).

Our visualision techniques for each model will be:
1. Printed summaries of each evaluation metric
2. Confusion matrices
3. A HTML Display of a dataframe with all the relevant results from the training.

We need do deal with the missing values and bounce rates by:
1. Using the dataframes where we replaced all missing values with 0.
2. Using the dataframes where we dropped all rows with missing values.
3. Use KNNImputer to replace missing values in the feauture 'SpecialDay'
These will be our baselines.

We will perform a train/test split on each of the dataframes and then perform 
1. One-Hot Encoding on categorical features; VisitorType, Weekend and, where applicable, Bounce Rate type, and drop the first of each in order to keep the number of columns as low as possible. Let's use ColumnTransformer for this.
2. Feature Scaling on numerical features; Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues. Lets use StandardScaler.

Train a RandomForestClassifier on each of our different train/test sets.
For each of the four sets we will use the evaluation metrics outlined above and visualise them according to the visualisation techniques mentiond above. As we have an imbalanced dataset, lets use StratifiedKFold.
We will choose the two best performers (one focusing on Recall-Latency and one on Precision-Latency) and then use these two sets for the rest of our training and evaluation.

After each of the preprocessing pipelines we log a dataset profile consisting of
• row count (n_samples) and
• minority‑class share (positive_ratio = y.sum()/n_samples).
These numbers are derived with Counter(y) immediately after the train/test split and are persisted to our results DataFrame.
The snapshot tells us how severe class imbalance remains in each variant and guides the choice of resampling: pipelines whose positive ratio ≥ 0.35 may skip heavy oversampling, while those below that threshold proceed to the under/over/SMOTE experiment.

For each of the two chosen sets we need to deal with the imbalance by testing these three approaches from IMBLearn for each set. 
1. Subsampling
2. Oversamplig
3. SMOTE, Synthetic Minority Oversampling Technique
As we have an imbalanced dataset, lets use StratifiedKFold for each fold.

Train a RandomForestClassifier on each of our six sets of preprocessing strategy/imbalance strategy combinations.
For each of the sets we will use the evaluation metrics outlined above and visualise them according to the visualisation techniques mentiond above.
We will choose the two best performers (one focusing on Recall-Latency and one on Precision-Latency) and then use these two sets for the rest of our training and evaluation.

When we train on data that has already been balanced through undersampling, oversampling or SMOTE, we fix class_weight=None in every estimator grid.
This avoids double‑counting the minority class: the resampling step has equalised the class distribution, so letting the model apply an additional weighting would skew the decision boundary.

For each of the two best combinations of preprocessing + imbalance strategy we will:
1. Use GridSearchCV with different params for each of the different Classifier models we will use. So 2 x 5 (15) models. As we have an imbalanced dataset, lets use StratifiedKFold.
2. In each GridsearchCV the parameters will be:
    1. estimator will be one of the five classifier models
    2. param_grid will be a variety of params suitable for each model
    3. cv will be 3 - for this type of prediction 3 folds will be plenty
    4. n_jobs will be -1 - the dataset is so small we dont need to be conservative with resources
    5. verbose will be 2
    6. scoring will be Accuracy, Recall and Precision
3. When each of the grid searches have been run the results of each shall be visualised according to the visualisation instructions above. We should store all the results in a dict which we turn into a dataframe for easy visualisation and analysis.

We will for each of the models do the following error analysis:
1. Identify and analyze misclassified samples from best-performing model
2. Group errors by type (false positives vs. false negatives)
3. Recall-Latency Trade-off in the form of a graph for all models.
4. Precision-Latency Trade-off in the form of a graph for all models.