# ML-KNN-Stroke-Prediction
Exploring KNN algorithm for stroke prediction dataset. The dataset comes from kaggle [https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset]

## Data Loading and Initial Cleaning and Exploration
```bash
$ Loading, Initial Exploration and Cleaning...
Information about dataset:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5110 entries, 0 to 5109
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   id                 5110 non-null   int64
 1   gender             5110 non-null   object
 2   age                5110 non-null   float64
 3   hypertension       5110 non-null   int64
 4   heart_disease      5110 non-null   int64
 5   ever_married       5110 non-null   object
 6   work_type          5110 non-null   object
 7   Residence_type     5110 non-null   object
 8   avg_glucose_level  5110 non-null   float64
 9   bmi                4909 non-null   float64
 10  smoking_status     5110 non-null   object
 11  stroke             5110 non-null   int64
dtypes: float64(3), int64(4), object(5)
memory usage: 479.2+ KB
None
Null values in the dataset:
id                     0
gender                 0
age                    0
hypertension           0
heart_disease          0
ever_married           0
work_type              0
Residence_type         0
avg_glucose_level      0
bmi                  201
smoking_status         0
stroke                 0
dtype: int64
Number of instances with missing BMI values and a stroke: 40
Dropping Non Relevant Columns and Rows...
Dropped column: id
Dropped row index(es) where gender == 'Other': [3116]
```
We can see that 201 observations has missing bmi values, Then we count the number of instances where there is a stroke and has missing bmi values.
We will not be dropping any missing values columns since we will be losing 40 instances of stroke. We will later impute the values with median but we can also test out different methods and pick the one that performs best.

## Exploratory Data Analysis
We start by looking at the correlation plot of numerical features.
```bash
$Correlation Plot...
```
![correlation_plot](https://github.com/kkharel/ML-KNN-Stroke-Prediction/assets/59852121/5c5206c4-7813-4968-aff3-aaf7a096127b)

The correlation plot is suggesting that all numerical features are positively correlated with each other but the strength of the correlation is less than 50%. Age and bmi seems to have highest correlation.

```bash
$Count of Stroke Cases by Features...
```
![stroke_analysis](https://github.com/kkharel/ML-KNN-Stroke-Prediction/assets/59852121/ef60e862-9269-4aba-ac2f-0c388464c85e)

Bar Plot of Stroke Occurences by each feature is suggesting we have higher instances of stroke for females gender, never smoked individuals, married individuals, individuals who work in private sector and individuals who lives in urban areas. Just by looking at the count, we have some idea on number of stroke occurences by each features but we have no idea on what %age of individuals have the stroke. To analyze the percentage we create another bar plot below.

```bash
$Percentage of Stroke Cases by Features...
```
![%age analysis](https://github.com/kkharel/ML-KNN-Stroke-Prediction/assets/59852121/06bc23ea-5e05-4130-9b68-895651ccc082)

From the percentage bar plot, we can see that male has the higher percentage of stroke out of all the observations that are collected for male. People that have heart disease have higher percentage of stroke and people with hypertension also have higher pecentage of stroke.

Let's further look into the relationship between numerical variables using bubble plots.

```bash
$Bubble Plot of Age, BMI, Average Glucose Level...
```
![bubble_plots](https://github.com/kkharel/ML-KNN-Stroke-Prediction/assets/59852121/c1378c16-b977-42ed-a268-66a7eaf5adef)

From the bubble plot, we can see age and bmi has somewhat linearly increasing relationship. Higher bmi and higher age can be a potential reason for having a stroke. We can also see that higher glucose and older age can also be a potential reason for having a stroke. If we look at age and bmi plot, we can see two clusters suggesting that glucose may be bimodal. To further explore the data from above findings, we look at the distribution of numerical variables as well as counts of categorial variables below.

```bash
$Distribution Plot of Data...
```
![data_distribution](https://github.com/kkharel/ML-KNN-Stroke-Prediction/assets/59852121/f9d9f4a0-9e2a-46a9-bfe7-843558b9cc67)


From the data distribution, we further see that our classes of stroke are imbalanced and the glucose level may have bimodal distribution. BMI looks normally distributed. For better visualization of relationship between numerical features (can also be categorical as well), I'm also plotting the pairplot below with kernel density estimation on the diagonal.

```bash
$Pairplot of Numerical Features and Stroke...
```
![pairplot](https://github.com/kkharel/ML-KNN-Stroke-Prediction/assets/59852121/e3c9b37c-c876-41ca-8c01-d9defa311a8c)

The pairplot further strengthen out findings from bubble plots above. Now, we handle the outliers on bmi column and look at the boxplot before and after handling
```bash
$Outlier Handling...
$Boxplot Before Outliers...
```
![boxplot_before](https://github.com/kkharel/ML-KNN-Stroke-Prediction/assets/59852121/e9510e50-c930-4256-8140-a12b50b1df51)

The boxplot further strengthens our findings of outliers on bmi observations. We will only handle extreme points above 60 that was shown on bubble plot above and keep the rest of the observations. We do not want to lose any stroke observations since the class stroke has little instances compared to no stroke in the dataset

```bash
$Boxplot After Outlier handling...
```
![boxplot_after](https://github.com/kkharel/ML-KNN-Stroke-Prediction/assets/59852121/dc6887b7-dce8-4fcf-8e97-4fab6725d40c)

As we previously saw that glucose may be bimodal, let's verify this using  hypothesis testing  with Hartigan's Dip Test.

```bash
$Test for Bimodality...
Reject the null hypothesis for avg_glucose_level: Data is not unimodal (potentially bimodal or multimodal
).
Dip Statistic for avg_glucose_level: 0.010850592149298637
P-value for avg_glucose_level: 0.00039398915610933116
Fail to reject the null hypothesis for bmi: Data appears unimodal.
Dip Statistic for bmi: 0.0
P-value for bmi: 1.0
```
We confirmed that the data is bimodal. To handle for this situation, we can create a binary variables with bins based on average glucose levels. We also create binary variable to bin different age groups.

We have explored our data and found some interesting insights already. We found we had missing values, avg_glucose_level is bimodal, numerical features are in different scales, dataset is imbalanced among other interesting insights. Now we will split the dataset and prepare the data for machine learning algorithm.

```bash
$Train Test Split...
```
This project is mainly focused on understanding and diving deep into the KNN algorithm only and anyone who is interested to explore other machine learning algorithms can do so as well

## Data preprocessing and feature engineering and Selection

We impute the missing values on BMI column using median value

```bash
$Imputing Missing Values...
```
After imputing, we will reset the index of all the splits.

```bash
$ Resetting index from all splits...
```
Now we perform feature engineering to categorize average glucose level and age groups
```bash
$Engineering features...
```
After featuring engineering is done, we will perform encoding. We encoded the features based on the data type of each feature. We have explored, encoded and scaled the dataset for ML algorithm to the best of our knowledge. Now, we need to know which features to select for our KNN algorithm. SInce it is a distance based algorithm, having a lot of dimensions would distort the results since it will say that two points are close together when in reality in higher dimension, they can be very far apart.

There is also k-means feature engineering/selection technnique based on cluster analysis. We can use this method to discover groups of observations representing stroke that share similar patterns. We won't go over this algorithm in this project, since this is only focused on nearest neighbors. We will stick with MI technique.

We will also skip Principal Component Analysis for another time. PCA helps in dimensionality reduction, Anomaly detection, Noise Reduction and Decorrelation. It does so by partitioning the variation in the data.

Another common technique that is utilized for feature engineering is target encoding. This is mainly used for categorical variables. It is similar to one hot encoding, feature labeling with the difference that it also uses the label for encoding. Target encoding is used when we have high cardinality features. A target encoding
will derive the number of categories using the features relationship with the target variable. Another use case can be when the feature scores poorly with feature metric
like mutual information. A target encoding can help reveal feature's true informativeness. Target encoding should be carried out with caution because it risk overfitting data among other problems.

```bash
$One-Hot Encoding, Label Encoding and Min-Max Scaling features...
```

After enncoding variables, we need to choose which features to include for the prediction model initially. The metric we choose for our analysis is mutual information since it captures all types of relationship between variables and provides the score. We will look at the relationship between different variables and manually select the features based on mutual information score. Mutual Information: - helps us locate features with most potential It dscribes relationships between two variables in terms of uncertainty. If we know the value of a feature, how much more confident would we be about the target.

When MI is zero, the quantities are independent, neither can tell us anything about the other. In theory, there is no upperbound to what MI can be but values above 2 are uncommon. MI is a logarithmic quantity so it increases slowly. Note that we need to tell mutual information classifier explicitly which features are discrete.

From the MI score, we can see how much does each feature contributes to the prediction of the stroke variable. We will select the features that contributes more than 0.001 towards the prediction. We can also plot and see if low scores variables are indeed important or not. We have identified a set of potential features and we will select these features to develop our model. We can iterate over with different threshold selection to choose the features and evaluate the model performance. For our model, we include all the features to make the prediction

```bash
$Calculating MI Scores for Features...
$Selecting Features based on MI Score...
MI Score for each feature:
age                               0.038336
age_category_retired              0.024256
bmi                               0.012844
age_category_adult                0.008743
ever_married                      0.006712
avg_glucose_level                 0.006138
hypertension                      0.005797
heart_disease                     0.005460
work_type_children                0.005250
glucose_category_high             0.005033
age_category_teen                 0.003898
age_category_underage             0.003782
smoking_status_formerly smoked    0.001988
work_type_Self-employed           0.001761
glucose_category_dangerous        0.001188
glucose_category_normal           0.001176
smoking_status_Unknown            0.001075
age_category_old adult            0.000806
glucose_category_low              0.000590
smoking_status_never smoked       0.000230
Residence_type                    0.000213
work_type_Never_worked            0.000210
glucose_category_borderline       0.000206
smoking_status_smokes             0.000102
work_type_Private                 0.000043
work_type_Govt_job                0.000009
gender                            0.000008
Name: MI Scores, dtype: float64
Selected Features with MI Score >= 0.0:
['age', 'age_category_retired', 'bmi', 'age_category_adult', 'ever_married', 'avg_glucose_level', 'hypert
ension', 'heart_disease', 'work_type_children', 'glucose_category_high', 'age_category_teen', 'age_catego
ry_underage', 'smoking_status_formerly smoked', 'work_type_Self-employed', 'glucose_category_dangerous',
'glucose_category_normal', 'smoking_status_Unknown', 'age_category_old adult', 'glucose_category_low', 's
moking_status_never smoked', 'Residence_type', 'work_type_Never_worked', 'glucose_category_borderline', '
smoking_status_smokes', 'work_type_Private', 'work_type_Govt_job', 'gender']
```

As mentioned above, we found that our data is imbalanced. We will use random under sampler to balance the dataset. We can also use over sampler, but using this technique downgraded the performance of ML model. We can also consider another sampling techniques like stratified sampling and see if the performance of our algorithm increases. I already tried separating the dataset based on engineered feature glucose category and sample from each category but that seems to downgrade the performance of the model. 

```bash
$Balancing Imbalanced Class...
Shape of data after resampling:
(348, 27) (348,)
```

The purpose of this project was to explictly understand how KNN algorithm works but not only limited to its parameters, drawbacks and benefits.

Now we have balanced our classes and our data is ready for KNN.

We separate out the categorical variables and numerical variables to apply different distance metrics to them. Note: Apply Manhattan(cityblock) distance when you have a lot of numerical features, we have only 3 numerical features hence we will use euclidean distance here. For categorical variables, we can choose jaccard, hamming among other options based on use cases and data and domain knowledge.We choose hamming distance for our use case here. If we have ordinal variables, then we can implement gower distance for those features. WE do not have any ordinal features in our dataset hence we skip gower distance.

```bash
$Extracting column index of Features...
```
Now we try to find the best k value and best threshold using K-fold Cross Validation for our model by trying different k_values. We iterate through k values and a range of thresholds within each fold of cross-validation. We keep track of the best_k_value, best_threshold, and best_f1_score. The threshold that yields the highest F1-score is selected as the best threshold. 

```bash
$Evaluating and Finding the best k value and threshold using cross-validation for binary classification...
Best K Value: 25
Best Threshold: 0.48
Best F1 Score: 0.7724867724867724
```

We will show the plot generated through applying above method below.

```bash
$Thresholds and K value Plot...
```
![threshold_k](https://github.com/kkharel/ML-KNN-Stroke-Prediction/assets/59852121/308b7b6e-962a-4ce8-a871-b62a657426af)

From the plot above, we can visualize the best threshold and best k value for different thresholds and k values trials. Finally we will train and fit the final model with the best K value and threshold and generate predictions.
```bash
$Building final KNN classifier based on the best hyperparameters found in a previous step and making predictions on a test dataset...
```
Now we will assess the performance of the KNN classifier through various reports, plots and metrics.

```bash
$Classification and Confusion matrix from final classifier...
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.71      0.83      1458
           1       0.13      0.83      0.22        75

    accuracy                           0.72      1533
   macro avg       0.56      0.77      0.53      1533
weighted avg       0.95      0.72      0.80      1533

Confusion Matrix:
[[1040  418]
 [  13   62]]

```
The prevalence of stroke cases in the dataset is 5%, while the model's precision in identifying stroke cases is 13%."

From the classification report, the model's recall for stroke cases (83%) is a crucial metric because it indicates that the model is effective at capturing a significant portion of observations that may have a stroke. The lower precision for stroke cases (13%) means that there will be some false positives, but this might be an acceptable trade-off, depending on the context, since we want to ensure that we don't want to miss any potential stroke cases. The overall accuracy (72%) reflects that the model is fairly performing in identifying observations that may have a stroke within the dataset but. 

In disease prevention and identification cases like stroke prediction, the model is not acceptable. We can try other models like Logistic Regression, Random Forests etc... to see if they perform better than the KNN model. I would like to reiterate again that the focus for this project was on KNN algorithm but upcoming projects will implement other models as well. If anyone is interested to try out other models, feel free to clone the repository and pick up from here.

The weighted average precision (95%) suggests that the model is highly precise for non-stroke cases, which can be important to avoid unnecessary concern for those without strokes.

From the consfusion metrics, 
True Positives (TP): 62, the model correctly identified 60 instances as potential stroke cases
True Negatives (TN): 1040, the model correctly identified 1161 instances as non stroke cases
False Positives (FP): 418, the model incorrectly predicted a stroke when there was no stroke
False Negatives (FN): 13, the model failed to identify 15 instances of potential stroke cases

Now we plot the ROC curve.

```bash
$ROC plot...
```
![ROC_curve](https://github.com/kkharel/ML-KNN-Stroke-Prediction/assets/59852121/45a07a9f-c081-497d-b7a8-8838708a500d)

ROC curve above plots the True Positive Rate (sensitivity) against the False Positive Rate (1 - specificity) at different decision thresholds. A perfect model hugs the upper-left corner of the plot, while random guessing follows the diagonal line. The Area Under the Curve (AUC) quantifies overall model performance, with a higher AUC indicating better discrimination ability. ROC curves aid in threshold selection and provide insights into a model's ability to distinguish between positive and negative cases. Our model is good through the lens of ROC curve.

We have insights into the model predictions. Now, we should be able to explain our model, There are various ways to explain machine learning models but we will be focusing on couple metrics to explain our model

Permutation importance is one way to explain our model by shuffling the values of a column concept. Permutation importance is used only after the model is fit. It helps us answer " What features does the model thinks are important?"

```bash
$Explaining the KNN Model with Permutation Importance...
Explained as: feature importances

Feature importances, computed as a decrease in score when feature
values are permuted (i.e. become noise). This is also known as
permutation importance.

If feature importances are computed on the same data as used for training,
they don't reflect importance of features for generalization. Use a held-out
dataset if you want generalization feature importances.

0.0660 ± 0.0119  age
0.0061 ± 0.0013  age_category_old adult
0.0039 ± 0.0111  avg_glucose_level
0.0025 ± 0.0044  smoking_status_Unknown
0.0025 ± 0.0028  Residence_type
0.0018 ± 0.0059  bmi
0.0017 ± 0.0016  heart_disease
0.0017 ± 0.0018  age_category_retired
0.0014 ± 0.0021  glucose_category_low
0.0014 ± 0.0024  ever_married
0.0014 ± 0.0022  hypertension
0.0009 ± 0.0018  work_type_Govt_job
0.0005 ± 0.0015  age_category_adult
     0 ± 0.0000  age_category_underage
     0 ± 0.0000  age_category_teen
     0 ± 0.0000  glucose_category_borderline
     0 ± 0.0000  work_type_children
     0 ± 0.0000  work_type_Never_worked
     0 ± 0.0000  glucose_category_dangerous
-0.0003 ± 0.0037  work_type_Private
                 … 7 more …
```

Features with positive permutation importance indicates that shuffling the values for these features resulted in decrease in model performance. Hence, these are considered important features for models prediction. Features with neutral or close to zero importance indicates that Shuffling their values did not significantly affect the model performance. Hence, they are neither important nor unimportant. Features with negative permutation importance indicates that shuffling the values for these features resulted in increase in model performace compared to baseline. This suggests that model might have been relying on incorrect or misleading information from these features, and shuffling them helped the model make better predictions. We should consider re-evaluating the relevance of features with negative importance and investigate the reasons behind negative impact on the model. The standard deviation measures the uncertainty or variablility in the importance scores across multiple iterations of permutation process. It tells us that there is some uncertainty about the exact magnitude of its importance.

Another way to explain the model is to use Partial Dependence Plots. It is also calculated after model is being fit. It is used for intrepretable models like decision trees and linear regression as they provide insights into how each feature addects the model prediction. KNN is a non-parametric approach and hence PDP's are not applicable. "How does each feature affects the predictions?". While feature importance shows what variables most affect predictions, partial dependence plots show how a feature affects predictions.

PDPs helps us answer the questions like below:
Controlling for all other features, what impact do glucose_category have on stroke? 
Are predicted stroke differences between class 1 and class 0 due to difference in their glucose level or by some other factors?

Since we cannot use PDP to explain our model we will use LimeTabularExplainer which visualizes and presents the importance of each feature in contributing to the model's decision for a specific instance from the test data. This can help us understand the model's behavior and the factors influencing its predictions for that instance.

```bash
$Explaining the KNN Model with LIME using instance from dataset...
Explanation for row index: 1
<lime.explanation.Explanation object at 0x000001B30173E3E0>
Row data:
gender                            0.000000
hypertension                      0.000000
heart_disease                     0.000000
ever_married                      1.000000
Residence_type                    1.000000
work_type_Govt_job                0.000000
work_type_Never_worked            0.000000
work_type_Private                 1.000000
work_type_Self-employed           0.000000
work_type_children                0.000000
smoking_status_Unknown            1.000000
smoking_status_formerly smoked    0.000000
smoking_status_never smoked       0.000000
smoking_status_smokes             0.000000
glucose_category_borderline       0.000000
glucose_category_dangerous        0.000000
glucose_category_high             0.000000
glucose_category_low              0.000000
glucose_category_normal           1.000000
age_category_adult                0.000000
age_category_old adult            0.000000
age_category_retired              1.000000
age_category_teen                 0.000000
age_category_underage             0.000000
age                               0.743652
avg_glucose_level                 0.326009
bmi                               0.528436
Name: 1, dtype: float64
```
Note that LIME creates a locally faithful model around a specific prediction by sampling and perturbing the input data. It offers insights into how different features
affects the prediction for a single instance.

We cannot forget SHAP values for model interpretability. Shapley values is a model-agnostic way and are used to explain the prediction of a model by quantifying how much each feature contributes to that prediction. They provide a global explanation for a specific prediction. These are particularlyimportant for complex models like gradient boosting and deep learning.

To conclude, the KNN model is not performing well for this specific use case. We can try other algorithms like Logistic Regression, Decision Trees, Random Forests etc... and see if those models perform better than the current model. There is a lot of room for improvement. One can tune hyperparameters of the model like k-value and threshold and see if there is positive change in performance. One can always iterate and improve the model.

