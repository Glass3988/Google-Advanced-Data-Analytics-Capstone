import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.tree import plot_tree

import pickle


df0 = pd.read_csv("HR_capstone_dataset.csv")



df0.columns
df0 = df0.rename(columns={'Work_accident': 'work_accident',
                   'average_montly_hours': 'average_monthly_hours',
                   'time_spend_company': 'company_years',
                   'Department': 'department'})


df0.columns
df0.isna().sum()
df0.duplicated().sum()
df1 = df0.drop_duplicates(keep='first')

percentile25 = df1['company_years'].quantile(0.25)

percentile75 = df1['company_years'].quantile(0.75)

iqr = percentile75 - percentile25


upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print('Lower limit:', lower_limit)
print('Upper limit:', upper_limit)

outliers = df1[(df1['company_years'] > upper_limit) | (df1['company_years'] < lower_limit)]

print("Number of rows in 'company_years' with outliers:", len(outliers))



print(df1['left'].value_counts())
print()
print(df1['left'].value_counts(normalize=True))


df1[df1['number_project']==7]['left'].value_counts()

df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])


df_tree = df1.copy()

df_tree['salary'] = (df_tree['salary'].astype('category').cat
                    .set_categories(['low', 'medium', 'high']).cat.codes)


df_tree = pd.get_dummies(df_tree, drop_first=False)

y = df_tree['left']
X = df_tree.drop('left', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

TreeOG = DecisionTreeClassifier(random_state=0)

cv_params = {'max_depth': [4, 6, 8, None],
             'min_samples_leaf': [1, 2, 5],
             'min_samples_split': [2, 4, 6]
            }


scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

Tree1 = GridSearchCV(TreeOG, cv_params, scoring=scoring, cv=4, refit = 'roc_auc')

%%time
Tree1.fit(X_train, y_train)

Tree1.best_params_

Tree1.best_score_

def make_results(model_name:str, model_object, metric:str):
    
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }
    
    cv_results = pd.DataFrame(model_object.cv_results_)
    
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]]
                                             .idxmax(), :]
    
    auc = best_estimator_results.mean_test_roc_auc
    precision = best_estimator_results.mean_test_precision
    recall = best_estimator_results.mean_test_recall
    f1 = best_estimator_results.mean_test_f1
    accuracy = best_estimator_results.mean_test_accuracy
    
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'auc': [auc],
                          'precision': [precision],
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy]
                         })
    return table


Tree1_results = make_results('Decision Tree 1', Tree1, 'auc')
Tree1_results

ForestOG = RandomForestClassifier(random_state=0)

cv_params = {'max_depth': [3, 5, None],
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1, 2, 3],
             'min_samples_split': [2, 3, 4],
             'n_estimators': [300, 500]
            }

scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

Forest1 = GridSearchCV(ForestOG, cv_params, scoring=scoring, cv = 4, refit='roc_auc')

%%time

Forest1.fit(X_train, y_train)

path = 'data/HR_capstone_dataset.csv'

def write_pickle(path, model_object, save_as:str):
    
    with open(path + save_as + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)


def read_pickle(path, saved_model_name:str):
    
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model


write_pickle(path, Forest1, 'Forest1')

Forest1 = read_pickle(path, 'Forest1')

Forest1.best_score_

Forest1.best_params_

Forest1_results = make_results('Random Forest 1', Forest1, 'auc')
print(Tree1_results)
print(Forest1_results)


def get_scores(model_name: str, model, X_test_data, y_test_data):

    preds = model.best_estimator_.predict(X_test_data)
    probs = model.best_estimator_.predict_proba(X_test_data)[:, 1]

    auc = roc_auc_score(y_test_data, probs)
    pr_auc = average_precision_score(y_test_data, probs)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({
        'model': [model_name],
        'precision': [precision],
        'recall': [recall],
        'f1': [f1],
        'accuracy': [accuracy],
        'ROC_AUC': [auc],
        'PR_AUC': [pr_auc]
    })

    return table


Forest1_test = get_scores('Forest1 test', Forest1, X_test, y_test)
Forest1_test

y_probs = Forest1.best_estimator_.predict_proba(X_test)[:, 1]

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_probs)

pr_auc = average_precision_score(y_test, y_probs)

print(f"PR-AUC: {pr_auc:.4f}")


thresholds = np.linspace(0, 1, 101)

precision_scores = []
recall_scores = []

for threshold in thresholds:
    threshold_preds = (y_probs >= threshold).astype(int)
    
    precision_scores.append(
        precision_score(y_test, threshold_preds, zero_division=0)
    )
    recall_scores.append(
        recall_score(y_test, threshold_preds, zero_division=0)
    )


cost_of_losing_employee = 50000
cost_of_intervention = 5000


thresholds = np.linspace(0, 1, 101)

results = []

for threshold in thresholds:
    preds = (y_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    benefit = tp * cost_of_losing_employee
    intervention_cost = (tp + fp) * cost_of_intervention
    missed_cost = fn * cost_of_losing_employee

    net_value = benefit - intervention_cost - missed_cost

    results.append({
        "threshold": threshold,
        "net_value": net_value
    })

cost_df = pd.DataFrame(results)


best_row = cost_df.loc[cost_df["net_value"].idxmax()]

best_threshold = best_row["threshold"]
best_value = best_row["net_value"]

print(f"Best threshold: {best_threshold:.2f}")
print(f"Estimated net value: ${best_value:,.0f}")

df2 = df_tree.drop('satisfaction_level', axis=1)


df2['overworked'] = df2['average_monthly_hours']

print('Max Hours:', df2['overworked'].max())
print('Min Hours:', df2['overworked'].min())

df2['overworked'] = (df2['overworked'] > 175).astype(int)

df2 = df2.drop('average_monthly_hours', axis=1)


y = df2['left']

X = df2.drop('left', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


TreeOG2 = DecisionTreeClassifier(random_state=0)

cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [1, 2, 5],
             'min_samples_split': [2, 4, 6]
             }

scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

Tree2 = GridSearchCV(TreeOG2, cv_params, scoring=scoring, cv=4, refit='roc_auc')


%%time

Tree2.fit(X_train, y_train)

Tree2.best_params_

Tree2.best_score_

Tree2_results = make_results('Decision Tree 2', Tree2, 'auc')
print(Tree1_results)
print(Tree2_results)


ForestOG2 = RandomForestClassifier(random_state=0)

cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }

scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

Forest2 = GridSearchCV(ForestOG2, cv_params, scoring=scoring, cv = 4, refit='roc_auc')


%%time

Forest2.fit(X_train, y_train)


write_pickle(path, Forest2, 'Forest2')

Forest2 = read_pickle(path, 'Forest2')

Forest2.best_params_

Forest2.best_score_

Forest2_results = make_results('Random Forest 2', Forest2, 'auc')
print(Tree2_results)
print(Forest2_results)


Forest2_test = get_scores('Forest2 test', Forest2, X_test, y_test)

preds = Forest2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=Forest2.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=Forest2.classes_)
disp.plot(values_format='')


Tree2_importances = pd.DataFrame(Tree2.best_estimator_.feature_importances_, 
                                 columns=['gini_importance'], 
                                 index=X.columns
                                )
Tree2_importances = Tree2_importances.sort_values(by='gini_importance', ascending=False)

Tree2_importances = Tree2_importances[Tree2_importances['gini_importance'] != 0]
Tree2_importances
