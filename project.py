#---------------- IMPORT LIBRARIES ----------------#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

#---------------- LOAD DATA ----------------#
data = pd.read_csv("wisc_bc_data.csv")   # <-- change file path

# Convert diagnosis M = 1, B = 0
data["diagnosis"] = data["diagnosis"].map({'M': 1, 'B': 0})

# Drop ID column if present
data = data.drop(columns=["id"], errors='ignore')

X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

#---------------- TRAIN & TEST SPLIT ----------------#
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=123, stratify=y)

#---------------- STANDARDIZE FEATURES ----------------#
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#---------------- METRIC FUNCTION ----------------#
def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred),
        "Error_Rate": 1 - accuracy_score(y_true, y_pred)
    }

results = {}

#---------------- 1. KNN ----------------#
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
pred_knn = knn.predict(X_test_scaled)
results["KNN"] = calculate_metrics(y_test, pred_knn)

#---------------- 2. Naive Bayes ----------------#
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
pred_nb = nb.predict(X_test_scaled)
results["Naive_Bayes"] = calculate_metrics(y_test, pred_nb)

#---------------- 3. Decision Tree ----------------#
dt = DecisionTreeClassifier(random_state=123)
dt.fit(X_train_scaled, y_train)
pred_dt = dt.predict(X_test_scaled)
results["Decision_Tree"] = calculate_metrics(y_test, pred_dt)

#---------------- 4. Logistic Regression ----------------#
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
pred_log = logreg.predict(X_test_scaled)
results["Logistic_Regression"] = calculate_metrics(y_test, pred_log)

#---------------- 5. K-Means (Unsupervised) ----------------#
kmeans = KMeans(n_clusters=2, random_state=123)
clusters = kmeans.fit_predict(X_train_scaled)

# Map cluster label to 0/1 based on majority class
cluster_map = {0: int(y_train[clusters==0].mean() > 0.5),
               1: int(y_train[clusters==1].mean() > 0.5)}

clusters_mapped = np.vectorize(cluster_map.get)(clusters)
results["KMeans"] = calculate_metrics(y_train, clusters_mapped)

#---------------- RESULTS DATAFRAME ----------------#
results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index':'Model'})
print(results_df)

#---------------- PLOT ----------------#
results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Value')

plt.figure(figsize=(10,5))
sns.barplot(x="Metric", y="Value", hue="Model", data=results_melted)
plt.xticks(rotation=45)
plt.title("Model Comparison Across Evaluation Metrics")
plt.show()
