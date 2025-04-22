Our data is imbalanced, has missing values, and invalid bonce rates.

During our EDA we have so far created 4 dataframes that we will use for training and evaluate which one creates the best performance.
We will use the dataframes in combinations with various classifer models that have been used during our Machine Learning course of 2025 and other techniques for dealing with missing values and imbalanced data.

These are the four dataframes:

1. session_data_bounce_rate_feature_nan_rows_dropped_df,
2. session_data_bounce_rate_feature_nans_replaced_with_zero_and_unknown_df,
3. session_data_nan_rows_dropped_df,
4. session_data_nans_replaced_with_zero_and_unknown_df

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
IMPORTANT: We will in no part of the evaluation process consider F1. We want to capture as many converting users as possible, as fast as possible. So we will optimise for recall and latency.

Regarding speed: For each candidate model we record median prediction latency (milliseconds per sample) on a fixed 1 000‑row validation batch.
Concretely, inside every cross‑validation fold we time one call to clf.predict(X_val[:1000]) with time.perf_counter(), divide by 1 000 and store the result.
The median across folds is reported as p50 latency, a stable proxy for typical real‑time service overhead on production hardware.

Our visualision techniques for each model will be:
1. Printed summaries of each evaluation metric
2. Confusion matrices
3. Feature importance graphs
4. A HTML Display of a dataframe with all the relevant results from the training.
5. Basic info like: which model was used, which params were used, how many samples etc.

We need do deal with the missing values and bounce rates by:
1. Using the dataframes where we replaced all missing values with 'Unknown'; 'session_data_nans_replaced_with_zero_and_unknown_df' and 'session_data_bounce_rate_feature_nans_replaced_with_zero_and_unknown_df'
2. Using the dataframes where we dropped all rows with missing values; 'session_data_nan_rows_dropped_df' and 'session_data_bounce_rate_feature_nan_rows_dropped_df'
This means we will have four different baselines to train on.

**Before building the pipeline we cast `OperatingSystems`, `Browser`, `Region`, and `TrafficType` to `category` dtype so they are routed to the categorical one‑hot branch.**

Lets use this Canonical preprocessing pipeline
from sklearn.compose   import ColumnTransformer
from sklearn.pipeline  import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute    import KNNImputer             # numeric only
from sklearn.ensemble  import RandomForestClassifier # first‑pass model

numeric_cols = [
    "Administrative", "Administrative_Duration",
    "Informational",  "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates",    "ExitRates",
    "PageValues",     "SpecialDay"
]

categorical_cols = [
    "Month", "Weekend",
    "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType"
]

preproc = ColumnTransformer(
    transformers=[
        ("num_impute_scale", Pipeline([
            ("impute", KNNImputer()),          # only cheatsheet tools
            ("scale",  StandardScaler())
        ]), numeric_cols),
        ("cat_onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

baseline_model = Pipeline([
    ("prep", preproc),
    ("clf",  RandomForestClassifier(random_state=42))
])


We will perform a train/test split on each of the four dataframes and then perform 
1. One-Hot Encoding on features; VisitorType, Weekend and, where applicable, Bounce Rate type, and drop the first of each in order to keep the number of columns as low as possible. Let's use ColumnTransformer for this.
2. Feature Scaling on features; Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues. Lets use StandardScaler.

Train a RandomForestClassifier on each of our four different train/test sets.
For each of the four sets we will use the evaluation metrics outlined above and visualise them according to the visualisation techniques mentiond above. As we have an imbalanced dataset, lets use StratifiedKFold.
We will choose the two best performers and then use these two sets for the rest of our training and evaluation.

After each of the four preprocessing pipelines we log a dataset profile consisting of
• row count (n_samples) and
• minority‑class share (positive_ratio = y.sum()/n_samples).
These numbers are derived with Counter(y) immediately after the train/test split and are persisted to our results DataFrame.
The snapshot tells us how severe class imbalance remains in each variant and guides the choice of resampling: pipelines whose positive ratio ≥ 0.35 may skip heavy oversampling, while those below that threshold proceed to the under/over/SMOTE experiment.

For each of the two chosen sets we need to deal with the imbalanace by testing these three approaches from IMBLearn for each set. 
1. Subsampling
2. Oversamplig
3. SMOTE, Synthetic Minority Oversampling Technique
As we have an imbalanced dataset, lets use StratifiedKFold for each fold.
This means we will have 2 x 3 (6) ways of dealing with the imbalance.

Train a RandomForestClassifier on each of our six sets of preprocessing strategy/imbalance strategy combinations.
For each of the 6 sets we will use the evaluation metrics outlined above and visualise them according to the visualisation techniques mentiond above.
We will choose the two best performers and then use these two sets for the rest of our training and evaluation.

When we train on data that has already been balanced through undersampling, oversampling or SMOTE, we fix class_weight=None in every estimator grid.
This avoids double‑counting the minority class: the resampling step has equalised the class distribution, so letting the model apply an additional weighting would skew the decision boundary.

For each of the two best combinations of preprocessing + imbalance strategy we will:
1. Use GridSearchCV with different params for each of the different Classifier models we will use. So 2 x 5 (15) models. As we have an imbalanced dataset, lets use StratifiedKFold.
2. In each GridsearchCV the parameters will be:
    1. estimator will be one of the five classifier models
    2. param_grid will be a variety of params suitable for each model (see param_grids below)
    3. cv will be 3
    4. n_jobs will be -1
    5. verbose will be 2
    6. scoring will be Accuracy, Recall and Precision
3. When each of the 15 grid searches have been run the results of each shall be visualised according to the visualisation instructions above. We should store all the results in a dict which we turn into a dataframe for easy visualisation and analysis.

We will for each of the 15 models do the following error analysis:
1. Identify and analyze misclassified samples from best-performing model
2. Group errors by type (false positives vs. false negatives)
3. Calculate per-feature error rates to pinpoint which features contribute most to misclassifications
4. Precision-Recall Trade-off in the form of a graph for all models.
5. Recall-Latency Trade-off in the form of a graph for all models.
5. Precision-Latency Trade-off in the form of a graph for all models.

# Param_grids per model:
### LogisticRegression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga'],  # Solver algorithm
    'class_weight': [None]  # Weight adjustment for classes
}

### DecisionTreeClassifier
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],  # Maximum tree depth
    'min_samples_split': [2, 5, 10, 20],  # Min samples to split node
    'min_samples_leaf': [1, 2, 4, 8],  # Min samples in leaf node
    'criterion': ['gini', 'entropy'],  # Split criterion
    'class_weight': [None]  # Weight adjustment
}

### RandomForestClassifier
param_grid = {
    'n_estimators': [5, 10, 20],  # Number of trees
    'max_depth': [None, 5, 10],  # Maximum tree depth
    'min_samples_split': [2, 5, 10],  # Min samples to split node
    'min_samples_leaf': [1, 2, 4],  # Min samples in leaf node
    'bootstrap': [True, False],  # Bootstrap samples
    'class_weight': [None,]  # Weight adjustment
}

### AdaBoostClassifier
param_grid = {
    'n_estimators': [10, 50, 100],  # Number of estimators
    'learning_rate': [0.01, 0.1, 1.0],  # Learning rate
    'algorithm': ['SAMME', 'SAMME.R'],  # Boosting algorithm
    'estimator': [
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=2)
    ]  # Base estimator type
}

### KNeighborsClassifier
param_grid = {
    "n_neighbors": [3, 5, 7, 11],          # neighbourhood size
    "weights": ["uniform", "distance"],    # vote weighting
    "p": [1, 2]                            # 1 = Manhattan, 2 = Euclidean
}