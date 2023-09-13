# ML-KNN-Stroke-Prediction
Exploring KNN algorithm for stroke prediction dataset

# We can see that 201 observations has missing bmi values,
# Lets count the number of instances where there is a stroke and has missing bmi values

# We will not be dropping any missing values columns since we will be losing
# 40 instances of stroke. We will impute the values with median but we can also
# test out different methods and pick the one that performs best.

# Removing non-relevant columns


# We will not be removing any observations as outliers since these
# observations are meaningful for stroke prediction


# The correlation plot is suggesting that all numerical features are positively
# correlated with each other but the strength of the correlation is less than
# 50%. Age and bmi seems to have highest correlation.


# Bar Plot of Stroke Occurences by each feature



# In the dataset, we have higher instances of stroke for females gender, never
# smoked individuals, married individuals, individuals who work in private sector
# and individuals who lives in urban areas. Just by looking at the count, we have
# some idea on number of stroke occurences by each features but we have no idea
# on what %age of individuals have the stroke. To analyze the percentage we create
# another bar plot below.




# From the percentage bar plot, we can see that male has the higher percentage
# of stroke out of all the observations that are collected for male. People that
# have heart disease have higher percentage of stroke and people with hypertension
# also have higher pecentage of stroke.

# Let's further look into the relationship between numerical variables using
# bubble plots

# Bubble plots 



# From the bubble plot, we can see age and bmi has somewhat linearly increasing
# relationship. Higher bmi and higher age can be a potential reason for having
# a stroke. We can also see that higher glucose and older age can also be a 
# potential reason for having a stroke. If we look at age and bmi plot, we can
# see two clusters suggesting that glucose may be bimodal.

# To further explore the data from above findings, we look at the distribution
# of numerical variables as well as counts of categorial variables below.


# From the data distribution, we further see that our classes of stroke are imbalanced
# and the glucose level may have bimodal distribution. BMI looks normally distributed.
# For better visualization of relationship between numerical features (can also
# be categorical as well), I'm also plotting the pairplot below with kernel
# density estimation on the diagonal


# The pairplot further strengthen out findings from bubble plots above.
# As we previously saw that glucose may be bimodal, let's verify this using 
# hypothesis testing  with Hartigan's Dip Test.


# We confirmed that the data is bimodal. To handle for this situation, 
# we can create a binary variables with bins based on average glucose levels


# We have explored our data and found some interesting insights already. We found
# we had missing values, avg_glucose_level is bimodal, numerical features are in
# different scales, dataset is imbalanced among other interesting insights. 
# Now we will split the dataset and prepare the data for machine learning algorithm.

# This project is mainly focused on understanding and diving deep into
# the KNN algorithm only and anyone who is interested to explore other 
# machine learning algorithms can do so as well



# Feature engineering to handle for bimodal column

# We will encoding the features based on the data type of each feature.



# We have explored, encoded and scaled the dataset for ML algorithm to the 
# best of our knowledge. Now, we need to know which features to select for our
# KNN algorithm. SInce it is a distance based algorithm, having a lot of dimensions
# would distort the results since it will say that two points are close together
# when in reality in higher dimension, they can be very far apart.

# There is also k-means feature engineering/selection technnique based on cluster 
# analysis. We can use this method to discover groups of observations 
# representing stroke that share similar patterns. We won't go over
# this algorithm in this project, since this is only focused on
# nearest neighbors. We will stick with MI technique.

# We will also skip Principal Component Analysis for another time. 
# PCA helps in dimensionality reduction, Anomaly detection, Noise Reduction 
# and Decorrelation. It does so by partitioning the variation in the data.

# Another common technique that is utilized for feature engineering is target
# encoding. This is mainly used for categorical variables. It is similar to
# one hot encoding, feature labeling with the difference that it also uses the 
# the label for encoding

# Target encoding is used when we have high cardinality features. A target encoding
# will derive the number of categories using the features relationship with the 
# target variable. Another use case can be when the feature scores poorly with feature metric
# like mutual information. A target encoding can help reveal feature's true 
# informativeness. Target encoding should be carried out with caution because
# it risk overfitting data among other problems.


# The metric we choose for our analysis is mutual information since it captures
# all types of relationship between variables and provides 
# the score. We will look at the relationship between different variables and
# manually select the features based on mutual information score.
# Mutual Information: - helps us locate features with most potential
# It dscribes relationships between two variables in terms of uncertainty.
# If we know the value of a feature, how much more confident would we be
# about the target.

# When MI is zero, the quantities are independent, neither can tell us anything 
# about the other. In theory, there is no upperbound to what MI can be 
# but values above 2 are uncommon. MI is a logarithmic quantity so it increases slowly.

# Note that we need to tell mutual information classifier explicitly which
# features are discrete.


# From the MI score, we can see how much does each feature contributes 
# to the prediction of the stroke variable. We will select the features
# that contributes more than 0.001 towards the prediction. We can also
# plot and see if low scores variables are indeed important or not

# We have identified a set of potential features and we will select these
# features to develop our model




# As mentioned above, we found that our data is imbalanced. We will
# use random under sampler to balance the dataset. We can also use over sampler,
# but using this technique downgraded the performance of ML model. We can also
# consider another sampling techniques like stratified sampling and see if the performance
# of our algorithm increases. I already tried separating the dataset
# based on engineered feature glucose category and sample from each
# category but that seems to downgrade the performance of the model. 





# The purpose of this project was to explictly understand how KNN algorithm works but not 
# only limited to its parameters, drawbacks and benefits, 

# Now we have balanced our classes and our data is ready for KNN

# We will separate out the categorical variables and numerical variables
# to apply different distance metrics to them.
# Note: Apply Manhattan(cityblock) distance when you have a lot of numerical features, 
# we have only 3 numerical features hence we will use euclidean distance here.
# For categorical variables, we can choose jaccard, hamming among other options
# based on use cases and data and domain knowledge.We choose hamming distance
# for our use case here. If we have ordinal variables,
# then we can implement gower distance for those features. WE do not have any
# ordinal features in our dataset hence we skip gower distance.





  
  # The below function finds the distance based on features data types. We
  # can also give weights to the distances but for our model we give equal weights
  # to both hamming and euclidean distance measures.
  --custom distance


# Now we try to find the best k value and best threshold for our model by trying
# different k_values. Then, we select the optimal threshold and k value for our
# final model.


# We can see from the plot above that the model is able to distinguish between
# positive and negative classes when the K-value is 21. The area under the curve is
# 86% which is not bad for this model performance.

# Finally we will train and fit the final model with the best K value and threshold
# and generate predictions.



# Now we will assess the performance of the KNN classifier through various reports,
# plots and metrics.

# The prevalence of stroke cases in the dataset is 5%, while the model's 
# precision in identifying stroke cases is 17%."

# From the classification report,

# The model's recall for stroke cases (80%) is a crucial metric because it 
# indicates that the model is effective at capturing a significant portion 
# of observations that may have a stroke/
  
# The lower precision for stroke cases (17%) means that there will be 
# some false positives, but this might be an acceptable trade-off, 
# depending on the context, since we want to ensure that we don't miss 
# any potential stroke cases.

# The overall accuracy (80%) reflects that the model is performing 
# in identifying observations that may have a stroke within the dataset.

# The weighted average precision (95%) suggests that the model is 
# highly precise for non-stroke cases, which can be important to 
# avoid unnecessary concern for those without strokes.

# Confusion Matrix
# True Positives (TP): 60, the model correctly identified 60 instances as potential stroke cases
# True Negatives (TN): 1161, the model correctly identified 1161 instances as non stroke cases
# False Positives (FP): 297, the model incorrectly predicted a stroke when there was no stroke
# False Negatives (FN): 15, the model failed to identify 15 instances of potential stroke cases

# ROC Curve



# ROC curve above plots the True Positive Rate (sensitivity) against
# the False Positive Rate (1 - specificity) at different decision thresholds. 
# A perfect model hugs the upper-left corner of the plot, while random guessing 
# follows the diagonal line. 
# The Area Under the Curve (AUC) quantifies overall model performance,
# with a higher AUC indicating better discrimination ability. 
# ROC curves aid in threshold selection and provide insights into a model's 
# ability to distinguish between positive and negative cases.
# Our model is performing really well through the lens of ROC curve.

# We have insights into the model predictions. Now, we should be able to
# explain our model, There are various ways to explain machine learning models
# but we will be focusing on couple metrics to explain our model

# Explaining the model
#Permutation importance is one way to explain our model by shuffling the values of 
# a column concept. Permutation importance is used only after the model is fit. It helps
# us answer " What features does the model thinks are important?"



# Features with positive permutation importance indicates that shuffling
# the values for these features resulted in decrease in model performance. Hence,
# these are considered important features for models prediction

# Features with neutral or close to zero importance indicates that Shuffling their values
# did not significantly affect the model performance. Hence, they are neither
# important nor unimportant

# Features with negative permutation importance indicates that shuffling
# the values for these features resulted in increase in model performace compared
# to baseline. This suggests that model might have been relying on incorrect or
# misleading information from these features, and shuffling them helped the model
# make better predictions

# We should consider re-evaluating the relevance of features with negative importance
# and investigate the reasons behind negative impact on the model

# The standard deviation measures the uncertainty or variablility in the importance
# scores across multiple iterations of permutation process. It tells us that
# there is some uncertainty about the exact magnitude of its importance


# Another way to explain the model is to use Partial Dependence Plots. It is 
# also calculated after model is being fit. It 
# is used for intrepretable models like decision trees and linear regression
# as they provide insights into how each feature addects the model prediction.
# KNN is a non-parametric approach and hence PDP's are not applicable.

# How does each feature affects the predictions?

# While feature importance shows what variables most affect predictions,
# partial dependence plots show how a feature affects predictions.

# PDPs helps us answer the questions like below:

# Controlling for all other features, what impact do glucose_category
# have on stroke? 
# Are predicted stroke differences between class 1 and class 0 due to
# difference in their glucose level or by some other factors?

# Since we cannot use PDP to explain our model we will use LimeTabularExplainer
# which visualizes and presents the importance of each feature in 
# contributing to the model's decision for a specific instance 
# from the test data. This can help us understand the model's behavior 
# and the factors influencing its predictions for that instance.





# Note that LIME creates a locally faithful model around a specific prediction by sampling
# and perturbing the input data. It offers insights into how different features 
# affects the prediction for a single instance.

# We cannot forget SHAP values for model interpretability. 
# Shapley values is a model-agnostic way and are used to explain the prediction of a model 
# by quantifying how much each feature contributes to that prediction. 
# They provide a global explanation for a specific prediction. These are particularly
# important for complex models like gradient boosting and deep learning.


