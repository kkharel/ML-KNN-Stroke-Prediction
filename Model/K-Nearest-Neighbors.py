import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve,  f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import  RandomOverSampler
from sklearn.feature_selection import mutual_info_classif
from eli5.sklearn import PermutationImportance
from lime.lime_tabular import LimeTabularExplainer
import eli5
import diptest
import warnings

def load_describe_data():
  data = pd.read_csv("stroke.csv")
  print('Information about dataset: ')
  print(data.info())
  null_values = data.isnull().sum()
  print(f'Null values in the dataset:\n{null_values}')
  missing_bmi_stroke_count = data[data['bmi'].isnull() & (data['stroke'] == 1)].shape[0]
  print("Number of instances with missing BMI values and a stroke:", missing_bmi_stroke_count)
  return data  

def drop_non_relevant_cols_rows(data):
  dropped_column = 'id'
  data.drop(columns=[dropped_column], inplace=True)
  dropped_rows = data[data['gender'] == 'Other'].index.tolist()
  data = data[data['gender'] != 'Other']
  print("Dropped column:", dropped_column)
  print("Dropped row index(es) where gender == 'Other':", dropped_rows)
  return data

def correlation_plot(data):
  features = data.select_dtypes(include = ['float']).columns.tolist()
  cor_df = data[features]
  correlation_matrix = cor_df.corr()
  plt.figure(figsize = (10,6))
  heatmap = sns.heatmap(correlation_matrix, annot = True, fmt = "0.2f", cmap = "coolwarm_r", annot_kws={"size": 12})
  heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, horizontalalignment="right", fontsize = 14)
  heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0, horizontalalignment = "right", fontsize = 14)
  plt.title("Correlation Heatmap of Numerical Features", fontsize = 16)
  plt.savefig('correlation_plot.jpg', format = 'jpg', dpi = 300, bbox_inches = 'tight')
  plt.show()
  
def stroke_barplots(data):
  gender_data = data.groupby(['gender'])['stroke'].sum().reset_index()
  hypertension_data = data.groupby(['hypertension'])['stroke'].sum().reset_index()
  heart_disease_data = data.groupby(['heart_disease'])['stroke'].sum().reset_index()
  marriage_data = data.groupby(['ever_married'])['stroke'].sum().reset_index()
  work_type_data = data.groupby(['work_type'])['stroke'].sum().reset_index()
  residence_type_data = data.groupby(['Residence_type'])['stroke'].sum().reset_index()
  smoking_status_data = data.groupby(['smoking_status'])['stroke'].sum().reset_index()

  plt.figure(figsize=(20, 12))
  def annotate_bars(ax, data):
    for index, row in data.iterrows():
      ax.text(index, row['stroke'], str(row['stroke']), ha='center', va='bottom')
  # Gender vs. Stroke
  plt.subplot(2, 4, 1)
  bars = plt.bar(gender_data['gender'], gender_data['stroke'], color = 'b')
  plt.title('Gender vs. Stroke')
  annotate_bars(plt.gca(), gender_data)
  # Hypertension vs. Stroke
  plt.subplot(2, 4, 2)
  bars = plt.bar(hypertension_data['hypertension'], hypertension_data['stroke'], color = 'g')
  plt.xticks([0, 1], ['No Hypertension', 'Hypertension'])
  plt.title('Hypertension vs. Stroke')
  annotate_bars(plt.gca(), hypertension_data)
  # Heart Disease vs. Stroke
  plt.subplot(2, 4, 3)
  bars = plt.bar(heart_disease_data['heart_disease'], heart_disease_data['stroke'], color = 'r')
  plt.xticks([0, 1], ['No Heart Disease', 'Heart Disease'])
  plt.title('Heart Disease vs. Stroke')
  annotate_bars(plt.gca(), heart_disease_data)
  # Ever Married vs. Stroke
  plt.subplot(2, 4, 4)
  bars = plt.bar(marriage_data['ever_married'], marriage_data['stroke'], color = 'c')
  plt.xticks([0, 1], ['Unmarried', 'Married'])
  plt.title('Ever Married vs. Stroke')
  annotate_bars(plt.gca(), marriage_data)
  # Work Type vs. Stroke
  plt.subplot(2, 4, 5)
  bars = plt.bar(work_type_data['work_type'], work_type_data['stroke'], color = 'm')
  plt.title('Work Type vs. Stroke')
  plt.xticks(rotation=45)
  annotate_bars(plt.gca(), work_type_data)
  # Residence Type vs. Stroke
  plt.subplot(2, 4, 6)
  bars = plt.bar(residence_type_data['Residence_type'], residence_type_data['stroke'], color = 'y')
  plt.xticks([0, 1], ['Rural', 'Urban'])
  plt.title('Residence Type vs. Stroke')
  annotate_bars(plt.gca(), residence_type_data)
  # Smoking Status vs. Stroke
  plt.subplot(2, 4, 7)
  bars = plt.bar(smoking_status_data['smoking_status'], smoking_status_data['stroke'], color = 'k')
  plt.title('Smoking Status vs. Stroke')
  plt.xticks(rotation=45)
  annotate_bars(plt.gca(), smoking_status_data)
  plt.tight_layout()
  plt.savefig("stroke_analysis.jpg", format="jpg", dpi=300, bbox_inches="tight")
  plt.show()

def percentage_barplots(data):
  counts1 = data.groupby(['gender', 'stroke'])['stroke'].value_counts().unstack(fill_value = 0)
  counts1['%_stroke_1'] = counts1[1] / (counts1[0] + counts1[1]) * 100
  counts1['%_no_stroke'] = 100 - counts1['%_stroke_1']
  counts2 = data.groupby(['hypertension', 'stroke'])['stroke'].value_counts().unstack(fill_value = 0)
  counts2['%_stroke_1'] = (counts2[1] / (counts2[0] + counts2[1])) * 100
  counts2['%_no_stroke'] = 100 - counts2['%_stroke_1']
  counts3 = data.groupby(['heart_disease', 'stroke'])['stroke'].value_counts().unstack(fill_value = 0)
  counts3['%_stroke_1'] = counts3[1] / (counts3[0] + counts3[1]) * 100
  counts3['%_no_stroke'] = 100 - counts3['%_stroke_1']
  counts4 = data.groupby(['ever_married', 'stroke'])['stroke'].value_counts().unstack(fill_value = 0)
  counts4['%_stroke_1'] = counts4[1] / (counts4[0] + counts4[1]) * 100
  counts4['%_no_stroke'] = 100 - counts4['%_stroke_1']
  counts5 = data.groupby(['work_type', 'stroke'])['stroke'].value_counts().unstack(fill_value = 0)
  counts5['%_stroke_1'] = counts5[1] / (counts5[0] + counts5[1]) * 100
  counts5['%_no_stroke'] = 100 - counts5['%_stroke_1']
  counts6 = data.groupby(['Residence_type', 'stroke'])['stroke'].value_counts().unstack(fill_value = 0)
  counts6['%_stroke_1'] = counts6[1] / (counts6[0] + counts6[1]) * 100
  counts6['%_no_stroke'] = 100 - counts6['%_stroke_1']
  counts7 = data.groupby(['smoking_status', 'stroke'])['stroke'].value_counts().unstack(fill_value = 0)
  counts7['%_stroke_1'] = counts7[1] / (counts7[0] + counts7[1]) * 100
  counts7['%_no_stroke'] = 100 - counts7['%_stroke_1']
  
  fig, axs = plt.subplots(2, 4, figsize=(20, 12))
  ax = axs[0, 0]
  counts1[['%_no_stroke', '%_stroke_1']].plot(kind='bar', stacked=True, ax=ax)
  for i, (no_stroke, stroke, percentage_no_stroke, percentage_stroke) in enumerate(zip(counts1['%_no_stroke'], counts1['%_stroke_1'], counts1['%_no_stroke'], counts1['%_stroke_1'])):
    y_no_stroke = no_stroke / 2
    y_stroke = no_stroke + stroke / 2
    if percentage_no_stroke > 10:
      y_no_stroke += 1  
    if percentage_stroke > 10:
      y_stroke += 1  
    ax.annotate(f'{percentage_no_stroke:.1f}%', (i, y_no_stroke), ha='center', va='center', fontsize=10, color='black')
    ax.annotate(f'{percentage_stroke:.1f}%', (i, y_stroke), ha='center', va='center', fontsize=10, color='black')
  plt.xlabel('Gender')
  plt.ylabel('%age')
  plt.title('%age of Gender vs Stroke')
  
  ax = axs[0, 1]
  counts2[['%_no_stroke', '%_stroke_1']].plot(kind='bar', stacked=True, ax=ax)
  for i, (no_stroke, stroke, percentage_no_stroke, percentage_stroke) in enumerate(zip(counts2['%_no_stroke'], counts2['%_stroke_1'], counts2['%_no_stroke'], counts2['%_stroke_1'])):
    y_no_stroke = no_stroke / 2
    y_stroke = no_stroke + stroke / 2
    if percentage_no_stroke > 10:
      y_no_stroke += 1 
    if percentage_stroke > 10:
      y_stroke += 1  
    ax.annotate(f'{percentage_no_stroke:.1f}%', (i, y_no_stroke), ha='center', va='center', fontsize=10, color='black')
    ax.annotate(f'{percentage_stroke:.1f}%', (i, y_stroke), ha='center', va='center', fontsize=10, color='black')
  plt.xlabel('Hypertension')
  plt.ylabel('%age')
  plt.title('%age of hypertension vs Stroke')
  
  ax = axs[0, 2]
  counts3[['%_no_stroke', '%_stroke_1']].plot(kind='bar', stacked=True, ax=ax)
  for i, (no_stroke, stroke, percentage_no_stroke, percentage_stroke) in enumerate(zip(counts3['%_no_stroke'], counts3['%_stroke_1'], counts3['%_no_stroke'], counts3['%_stroke_1'])):
    y_no_stroke = no_stroke / 2
    y_stroke = no_stroke + stroke / 2
    if percentage_no_stroke > 10:
      y_no_stroke += 1  
    if percentage_stroke > 10:
      y_stroke += 1  
    ax.annotate(f'{percentage_no_stroke:.1f}%', (i, y_no_stroke), ha='center', va='center', fontsize=10, color='black')
    ax.annotate(f'{percentage_stroke:.1f}%', (i, y_stroke), ha='center', va='center', fontsize=10, color='black')
  plt.xlabel('Heart-Disease')
  plt.ylabel('%age')
  plt.title('%age of Heart Disease vs Stroke')
  
  ax = axs[0, 3]
  counts4[['%_no_stroke', '%_stroke_1']].plot(kind='bar', stacked=True, ax=ax)
  for i, (no_stroke, stroke, percentage_no_stroke, percentage_stroke) in enumerate(zip(counts4['%_no_stroke'], counts4['%_stroke_1'], counts4['%_no_stroke'], counts4['%_stroke_1'])):
    y_no_stroke = no_stroke / 2
    y_stroke = no_stroke + stroke / 2
    if percentage_no_stroke > 10:
      y_no_stroke += 1 
    if percentage_stroke > 10:
      y_stroke += 1  
    ax.annotate(f'{percentage_no_stroke:.1f}%', (i, y_no_stroke), ha='center', va='center', fontsize=10, color='black')
    ax.annotate(f'{percentage_stroke:.1f}%', (i, y_stroke), ha='center', va='center', fontsize=10, color='black')
  plt.xlabel('Ever Married')
  plt.ylabel('%age')
  plt.title('%age of Ever Married vs Stroke')
  
  ax = axs[1, 0]
  counts5[['%_no_stroke', '%_stroke_1']].plot(kind='bar', stacked=True, ax=ax)
  for i, (no_stroke, stroke, percentage_no_stroke, percentage_stroke) in enumerate(zip(counts5['%_no_stroke'], counts5['%_stroke_1'], counts5['%_no_stroke'], counts5['%_stroke_1'])):
    y_no_stroke = no_stroke / 2
    y_stroke = no_stroke + stroke / 2
    if percentage_no_stroke > 10:
      y_no_stroke += 1  
    if percentage_stroke > 10:
      y_stroke += 1 
    ax.annotate(f'{percentage_no_stroke:.1f}%', (i, y_no_stroke), ha='center', va='center', fontsize=10, color='black')
    ax.annotate(f'{percentage_stroke:.1f}%', (i, y_stroke), ha='center', va='center', fontsize=10, color='black')
  plt.xlabel('Job Type')
  plt.ylabel('%age')
  plt.title('%age of Job Type vs Stroke')
  
  ax = axs[1, 1]
  counts6[['%_no_stroke', '%_stroke_1']].plot(kind='bar', stacked=True, ax=ax)
  for i, (no_stroke, stroke, percentage_no_stroke, percentage_stroke) in enumerate(zip(counts6['%_no_stroke'], counts6['%_stroke_1'], counts6['%_no_stroke'], counts6['%_stroke_1'])):
    y_no_stroke = no_stroke / 2
    y_stroke = no_stroke + stroke / 2
    if percentage_no_stroke > 10:
      y_no_stroke += 1 
    if percentage_stroke > 10:
      y_stroke += 1 
    ax.annotate(f'{percentage_no_stroke:.1f}%', (i, y_no_stroke), ha='center', va='center', fontsize=10, color='black')
    ax.annotate(f'{percentage_stroke:.1f}%', (i, y_stroke), ha='center', va='center', fontsize=10, color='black')
  plt.xlabel('Residence Type')
  plt.ylabel('%age')
  plt.title('%age of Residence Type vs Stroke')
  
  ax = axs[1, 2]
  counts7[['%_no_stroke', '%_stroke_1']].plot(kind='bar', stacked=True, ax=ax)
  for i, (no_stroke, stroke, percentage_no_stroke, percentage_stroke) in enumerate(zip(counts7['%_no_stroke'], counts7['%_stroke_1'], counts7['%_no_stroke'], counts7['%_stroke_1'])):
    y_no_stroke = no_stroke / 2
    y_stroke = no_stroke + stroke / 2
    if percentage_no_stroke > 10:
      y_no_stroke += 1  #
    if percentage_stroke > 10:
      y_stroke += 1 
    ax.annotate(f'{percentage_no_stroke:.1f}%', (i, y_no_stroke), ha='center', va='center', fontsize=10, color='black')
    ax.annotate(f'{percentage_stroke:.1f}%', (i, y_stroke), ha='center', va='center', fontsize=10, color='black')
  plt.xlabel('Smoking Status')
  plt.ylabel('%age')
  plt.title('%age of Smoking Status vs Stroke')
  for ax in axs[:, 0]:
    ax.set_ylabel('%age')
  for ax in axs[0, :]:
    ax.set_title(ax.get_title(), fontsize=14)
  for ax in axs[1, 3:]:
    fig.delaxes(ax)
  for ax in axs.ravel():
    if ax.get_legend() is not None:
      ax.get_legend().remove()
  legend = fig.legend(labels=['No Stroke', 'Stroke'], loc='lower right', bbox_to_anchor=(1.0, 0.1), fontsize=20)
  plt.subplots_adjust(right=0.85)
  plt.tight_layout()
  plt.savefig("%age analysis.jpg", format="jpg", dpi=300, bbox_inches="tight")
  plt.show()

def bubble_plots(data):
  type_colors = {1: 'red', 0: 'green'}
  stroke_colors = data['stroke'].map(type_colors)
  sorted_indices = stroke_colors.argsort()
  stroke_colors_sorted = stroke_colors.iloc[sorted_indices]
  fig, axes = plt.subplots(1, 3, figsize=(18, 8))
  
  scatter1 = axes[0].scatter(data['age'], data['bmi'], c=stroke_colors_sorted, s=data['avg_glucose_level'], alpha=0.7)
  legend_labels = list(type_colors.keys())
  legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=type_colors[type], markersize=10, label=type) for type in legend_labels]
  axes[0].legend(handles=legend_handles, labels=legend_labels, loc='best')
  axes[0].set_ylabel('bmi')
  axes[0].set_xlabel('Age')
  axes[0].set_title('Stroke - age vs bmi sized by avg_glucose_level colored by stroke')
  
  scatter2 = axes[1].scatter(data['age'], data['avg_glucose_level'], c=stroke_colors_sorted, s=data['bmi'], alpha=0.7)
  axes[1].legend(handles=legend_handles, labels=legend_labels, loc='best')
  axes[1].set_ylabel('Avg_glucose_level')
  axes[1].set_xlabel('Age')
  axes[1].set_title('Stroke - avg_glucose_level vs age sized by bmi colored by stroke')
  
  scatter3 = axes[2].scatter(data['avg_glucose_level'], data['bmi'], c=stroke_colors_sorted, s=data['age'], alpha=0.7)
  axes[2].legend(handles=legend_handles, labels=legend_labels, loc='best')
  axes[2].set_ylabel('bmi')
  axes[2].set_xlabel('avg_glucose_level')
  axes[2].set_title('Stroke - avg_glucose_level vs bmi sized by age colored by stroke')
  plt.tight_layout()
  plt.savefig("bubble_plots.jpg", format="jpg", dpi=300, bbox_inches="tight")
  plt.show()


def plot_distribution(data):
  column_names = data.columns.tolist()
  num_plots = len(column_names)
  figsize = (30,20)
  ncols = 3
  row_spacing = 0.5
  nrows = (num_plots - 1) // ncols + 1
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, gridspec_kw={'hspace': row_spacing})
  axes = axes.flatten()
  for idx, column in enumerate(column_names):
    ax = axes[idx]
    if column in data.select_dtypes(include=['float']).columns:
      sns.histplot(data[column], ax=ax, kde=True)
    else:
      sns.countplot(data=data, x=column, ax=ax)
    ax.set_title(column)
    ax.tick_params(axis='x', rotation=45)
  for ax in axes[num_plots:]:
    ax.axis('off')
  plt.savefig("data_distribution.jpg", format="jpg", dpi=300, bbox_inches="tight")
  plt.show()

def pairplot(data):
  warnings.filterwarnings("ignore", category=UserWarning)
  plt.figure(figsize = (10,6))
  num_cols = ['age', 'avg_glucose_level', 'bmi', 'stroke']
  sns.set(style="ticks")
  sns.pairplot(data[num_cols], hue="stroke", kind='kde')
  plt.tight_layout()
  plt.savefig("pairplot.jpg", format="jpg", dpi=300, bbox_inches="tight")
  plt.show()

def boxplot_before_outliers(data, feature):
  plt.figure(figsize=(8, 6))
  sns.boxplot(data=data, y=feature)
  plt.title(f'Box Plot of {feature} (Before Outlier Handling)')
  plt.savefig("boxplot_before.jpg", format = 'jpg', dpi = 300, bbox_inches = 'tight')
  plt.show()
    
def outlier_handling(data, feature):
  Q1=data[feature].quantile(0.20)
  Q3=data[feature].quantile(0.80)
  IQR=Q3-Q1
  lower_bound=Q1-1.5*IQR
  upper_bound=Q3+1.5*IQR
  data.loc[data[feature] > upper_bound, feature] = upper_bound
  data.loc[data[feature] < lower_bound, feature] = lower_bound
  return data

def boxplot_after_outliers(data, feature):
  plt.figure(figsize=(8, 6))
  sns.boxplot(data=data, y=feature)
  plt.title(f'Box Plot of {feature} (After Outlier Handling)')
  plt.savefig("boxplot_after.jpg", format = 'jpg', dpi = 300, bbox_inches = 'tight')
  plt.show()

def bimodal_test(feature):
  dip_statistic, p_value = diptest.diptest(data[feature])
  alpha = 0.05
  if p_value < alpha:
    print(f"Reject the null hypothesis for {feature}: Data is not unimodal (potentially bimodal or multimodal).")
  else:
    print(f"Fail to reject the null hypothesis for {feature}: Data appears unimodal.")
  print(f"Dip Statistic for {feature}: {dip_statistic}")
  print(f"P-value for {feature}: {p_value}")

def split_data(data):
  X = data.drop(columns = ['stroke'])
  y = data[['stroke']]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 10, stratify = y)
  return X_train, X_test, y_train, y_test

def impute_missing_values(X_train, X_test):
  imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
  missing_col = ['bmi']
  X_train[missing_col] = imputer.fit_transform(X_train[missing_col])
  X_test[missing_col] = imputer.transform(X_test[missing_col])
  return X_train, X_test

def reset_index(X_train, X_test, y_train, y_test):
  X_train = X_train.reset_index(drop = True)
  X_test = X_test.reset_index(drop = True)
  y_train = y_train.reset_index(drop = True)
  y_test = y_test.reset_index(drop = True)
  return X_train, X_test, y_train, y_test

def categorize_glucose_level(value):
  if value < 70:
    return "low"
  elif 70 <= value < 126:
    return "normal"
  elif 126 <= value < 182:
    return "borderline"
  elif 182 <= value < 250:
    return "high"
  else:
    return "dangerous"

def categorize_age_level(value):
  if value < 10:
    return "underage"
  elif 10 <= value < 20:
    return "teen"
  elif 20 <= value < 40:
    return "adult"
  elif 40 <= value < 60:
    return "old adult"
  else:
    return "retired"

def map_categories(X_train, X_test):
  X_train['glucose_category'] = X_train['avg_glucose_level'].transform(categorize_glucose_level)
  X_test['glucose_category'] = X_test['avg_glucose_level'].transform(categorize_glucose_level)
  X_train['age_category'] = X_train['age'].transform(categorize_age_level)
  X_test['age_category'] = X_test['age'].transform(categorize_age_level)
  return X_train, X_test

def preprocess_data(X_train, X_test):
  onehot_features = ['work_type', 'smoking_status', 'glucose_category', 'age_category']
  onehot_encoder = OneHotEncoder(sparse_output=False)
  X_train_encoded = onehot_encoder.fit_transform(X_train[onehot_features])
  X_test_encoded = onehot_encoder.transform(X_test[onehot_features])
  feature_names = onehot_encoder.get_feature_names_out()
  X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=feature_names)
  X_train_encoded_df = X_train_encoded_df.astype(int)
  X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=feature_names)
  X_test_encoded_df = X_test_encoded_df.astype(int)
  X_train = X_train.drop(columns=onehot_features)
  X_test = X_test.drop(columns=onehot_features)
  X_train = pd.concat([X_train, X_train_encoded_df], axis=1)
  X_test = pd.concat([X_test, X_test_encoded_df], axis=1)
  
  numerical_features = ['age', 'avg_glucose_level', 'bmi']
  scaler = MinMaxScaler()
  X_train_num_encoded = scaler.fit_transform(X_train[numerical_features])
  X_test_num_encoded = scaler.transform(X_test[numerical_features])
  feature_names = scaler.get_feature_names_out()
  X_train_num_encoded_df = pd.DataFrame(X_train_num_encoded, columns = feature_names)
  X_test_num_encoded_df = pd.DataFrame(X_test_num_encoded, columns = feature_names)
  X_train = X_train.drop(columns=numerical_features)
  X_test = X_test.drop(columns=numerical_features)
  X_train = pd.concat([X_train, X_train_num_encoded_df], axis=1)
  X_test = pd.concat([X_test, X_test_num_encoded_df], axis=1)
  
  categorical_features = ['gender', 'ever_married', 'Residence_type', 'hypertension', 'heart_disease']
  label_encoders = {}
  for feature in categorical_features:
    label_encoder = LabelEncoder()
    X_train[feature] = label_encoder.fit_transform(X_train[feature])
    X_train[feature] = X_train[feature] .astype(int)
    X_test[feature] = label_encoder.transform(X_test[feature])
    X_train[feature] = X_train[feature].astype(int)
    label_encoders[feature] = label_encoder
  return X_train, X_test

def mi_scores(X_train, y_train, discrete_features):
  mi_score = mutual_info_classif(X_train, y_train, discrete_features = X_train.dtypes == int)
  mi_score = pd.Series(mi_score, name = "MI Scores", index = X_train.columns)
  mi_score = mi_score.sort_values(ascending = False)
  return mi_score
  
def select_features(mi_score, threshold=0.0):
  print("MI Score for each feature:")
  print(mi_score)
  selected_features = mi_score[mi_score >= threshold].index.tolist()
  print(f"Selected Features with MI Score >= {threshold}:")
  print(selected_features)
  return selected_features

def balance_class(X_train, y_train, selected_features):
  undersample = RandomUnderSampler(sampling_strategy='majority', random_state=0)
  X_balanced, y_balanced = undersample.fit_resample(X_train[selected_features], y_train['stroke'])
  X_train_resampled = X_balanced.reset_index(drop=True).copy()
  y_train_resampled = y_balanced.reset_index(drop=True).copy()
  print("Shape of data after resampling:")
  print(X_train_resampled.shape, y_train_resampled.shape)
  return X_train_resampled, y_train_resampled

def get_indices(X_train_resampled):
  categories = X_train_resampled.select_dtypes(include = ['int']).columns.tolist()
  numerical = X_train_resampled.select_dtypes(include = ['float']).columns.tolist()
  categorical_indices = [X_train_resampled.columns.get_loc(col) for col in categories]
  numerical_indices = [X_train_resampled.columns.get_loc(col) for col in numerical]
  return categorical_indices, numerical_indices

def custom_distance(X, y, categorical_indices, numerical_indices):
  hamming_dist = distance.hamming(X[categorical_indices], y[categorical_indices])
  euclidean_dist = distance.euclidean(X[numerical_indices], y[numerical_indices])
  weighted_distance = 1 * hamming_dist + 1 * euclidean_dist
  return weighted_distance

def find_best_k_and_threshold(X_train_resampled, y_train_resampled):
  features = X_train_resampled.columns.tolist()
  k_values = list(range(3, 40, 2))
  thresholds = np.linspace(0, 1, 101) 
  best_k_value = None
  best_threshold = None
  best_f1_score = 0.0  
  best_threshold_f1_scores = []  
  threshold_scores = {}  
  for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=custom_distance, metric_params={
      'categorical_indices': categorical_indices,
      'numerical_indices': numerical_indices
    })
    predicted_probabilities = cross_val_predict(knn_classifier, X_train_resampled[features].values, y_train_resampled.values.ravel(), cv=5, method='predict_proba')
    threshold_f1_scores = []  
    for threshold in thresholds:
      binary_predictions = (predicted_probabilities[:, 1] > threshold).astype(int)
      f1 = f1_score(y_train_resampled, binary_predictions)
      if f1 > best_f1_score:
        best_f1_score = f1
        best_k_value = k
        best_threshold = threshold
        best_threshold_f1_scores = [f1]
      elif f1 == best_f1_score:
        best_threshold_f1_scores.append(f1)
      threshold_f1_scores.append(f1)
    threshold_scores[k] = threshold_f1_scores
  print(f"Best K Value: {best_k_value}")
  print(f"Best Threshold: {best_threshold}")
  print(f"Best F1 Score: {best_f1_score}")
  return best_k_value, best_threshold, best_f1_score, best_threshold_f1_scores, thresholds, threshold_scores

def score_threshold_plot(thresholds, threshold_scores, best_k_value, best_threshold):
  plt.figure(figsize=(12, 8))
  for k, f1_scores in threshold_scores.items():
    plt.plot(thresholds, f1_scores, label=f'k={k}')
  plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold (k={best_k_value})')
  plt.xlabel('Threshold')
  plt.ylabel('F1 Score')
  plt.title(f'F1 Scores for Different k Values at Various Thresholds')
  plt.legend()
  plt.savefig("threshold_k.jpg", format = 'jpg', dpi = 300, bbox_inches = 'tight')
  plt.show()

def final_model(X_train_resampled, y_train_resampled, X_test, best_k_value, best_threshold, categorical_indices, numerical_indices):
  features = X_train_resampled.columns.tolist()
  label = ['stroke']
  final_knn_classifier = KNeighborsClassifier(n_neighbors=best_k_value, metric=custom_distance, metric_params={
    'categorical_indices': categorical_indices,
    'numerical_indices': numerical_indices
  }, n_jobs=-1)
  final_knn_classifier.fit(X_train_resampled[features].values, y_train_resampled.values.ravel())
  # Prediction
  y_pred_prob_final = final_knn_classifier.predict_proba(X_test[features].values)[:, 1]
  # Convert predicted probabilities to binary predictions using the best threshold
  combined_predictions_final = (y_pred_prob_final > best_threshold).astype(int)
  return final_knn_classifier, y_pred_prob_final, combined_predictions_final

def model_report(y_test, combined_predictions_final):
  label = ['stroke']
  classification_rep = classification_report(y_test[label].values.ravel(), combined_predictions_final)
  conf_matrix = confusion_matrix(y_test[label].values.ravel(), combined_predictions_final)
  return classification_rep, conf_matrix

def ROC_curve(y_test, y_pred_prob_final):
  fpr, tpr, thresholds = roc_curve(y_test.values.ravel(), y_pred_prob_final)
  roc_auc = roc_auc_score(y_test.values.ravel(), y_pred_prob_final)
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc='lower right')
  plt.savefig("ROC_curve.jpg", format = 'jpg', dpi = 300, bbox_inches = 'tight')
  plt.show()

def permutation_importance():
  features = X_train_resampled.columns.tolist()
  label = ['stroke']
  perm = PermutationImportance(final_knn_classifier, random_state=42).fit(X_test[features].values, y_test[label].values.ravel())
  print(eli5.format_as_text(eli5.explain_weights(perm, feature_names= features)))

def LIME_explainer(classifier, X_train_resampled, X_test, row_index):
  features = X_train_resampled.columns.tolist()
  label = ['stroke']
  explainer = LimeTabularExplainer(X_train_resampled[features].values, mode="classification")
  instance_to_explain = X_test[features].iloc[row_index]  
  explanation = explainer.explain_instance(instance_to_explain, final_knn_classifier.predict_proba)
  print("Explanation for row index:", row_index)
  print(explanation)
  print("Row data:")
  print(X_test.iloc[row_index])


if __name__ == "__main__":
  
  # Load and preprocess the data
  print("Loading, Initial Exploration and Cleaning...")
  data = load_describe_data()
  
  print("Dropping Non Relevant Columns and Rows...")
  data = drop_non_relevant_cols_rows(data)

  # Create plots and visualizations for EDA
  print("Correlation Plot...")
  correlation_plot(data)
  print("Count of Stroke Cases by Features...")
  stroke_barplots(data)
  print("Percentage of Stroke Cases by Features...")
  percentage_barplots(data)
  print("Bubble Plot of Age, BMI, Average Glucose Level...")
  bubble_plots(data)
  print("Distribution Plot of Data...")
  plot_distribution(data)
  print("Pairplot of Numerical Features and Stroke...")
  pairplot(data)
  
  # Capping Outliers
  print("Outlier Handling...")
  print("Boxplot Before Outliers...")
  boxplot_before_outliers(data, 'bmi')
  outlier_handling(data, 'bmi')
  print("Boxplot After Outlier handling...")
  boxplot_after_outliers(data, 'bmi')
  
  # Testing for bimodality
  print("Test for Bimodality...")
  bimodal_test('avg_glucose_level')
  bimodal_test('bmi')

  # Train test split the data
  print("Train Test Split...")
  X_train, X_test, y_train, y_test = split_data(data)
  
  # Impute missing data
  print("Imputing Missing Values...")
  X_train, X_test = impute_missing_values(X_train, X_test)

  # Reset the index of dataset
  print("Resetting index from all splits...")
  X_train, X_test, y_train, y_test = reset_index(X_train, X_test, y_train, y_test)

  # Feature engineer and map categories for average glucose level column
  print("Engineering features...")
  X_train, X_test = map_categories(X_train, X_test)
  
  # Encode the dataset
  
  print("One-Hot Encoding, Label Encoding and Min-Max Scaling features...")
  X_train, X_test = preprocess_data(X_train, X_test)
  
  # MI Scores
  print("Calculating MI Scores for Features...")
  mi_score = mi_scores(X_train, y_train['stroke'], discrete_features = X_train.dtypes == int)

  # Select features
  print("Selecting Features based on MI Score...")
  selected_features = select_features(mi_score, threshold=0.00)

  # Balance class
  print("Balancing Imbalanced Class...")
  X_train_resampled, y_train_resampled = balance_class(X_train, y_train, selected_features)

  # Get categorical and numerical feature indices
  print("Extracting column index of Features...")
  categorical_indices, numerical_indices = get_indices(X_train_resampled)

  # Find the best k value
  print("Evaluating and Finding the best k value and threshold using cross-validation for binary classification....")
  best_k_value, best_threshold, best_f1_score, best_threshold_f1_scores, thresholds, threshold_scores = find_best_k_and_threshold(X_train_resampled, y_train_resampled)

  # Create and display thresholds and f_scores
  print("Thresholds and K value Plot...")
  score_threshold_plot(thresholds, threshold_scores, best_k_value, best_threshold)
  
  # Build the final KNN classifier
  print("Building final KNN classifier based on the best hyperparameters found in a previous step and making predictions on a test dataset...")
  final_knn_classifier, y_pred_prob_final, combined_predictions_final = final_model(X_train_resampled, y_train_resampled, X_test, best_k_value, best_threshold, categorical_indices, numerical_indices)

  # Generate and print the model report
  print("Classification and Confusion matrix from final classifier...")
  classification_rep, conf_matrix = model_report(y_test, combined_predictions_final)
  print("Classification Report:")
  print(classification_rep)
  print("Confusion Matrix:")
  print(conf_matrix)

  # Create and display ROC curve
  print("ROC plot...")
  ROC_curve(y_test, y_pred_prob_final)

  # Perform permutation importance
  print("Explaining the KNN Model with Permutation Importance...")
  permutation_importance()

  # Explain using LIME
  print("Explaining the KNN Model with LIME using instance from dataset...")
  LIME_explainer(final_knn_classifier, X_train_resampled, X_test, 1)

