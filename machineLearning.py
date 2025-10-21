# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Step 2: Read CSV files and combine
legitimate_df = pd.read_csv("datasets/structured_legitimate_data.csv")
phishing_df = pd.read_csv("datasets/structured_phishing_data.csv")
df = pd.concat([legitimate_df, phishing_df], axis=0).sample(frac=1).drop_duplicates()

# Step 3: Prepare features and labels
X = df.drop(['URL', 'label'], axis=1)
y = df['label']

# Step 4: Split train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Step 5: Define models
models = {
    'NB': GaussianNB(),
    'SVM': svm.LinearSVC(max_iter=10000),
    'DT': tree.DecisionTreeClassifier(),
    'RF': RandomForestClassifier(n_estimators=60),
    'AB': AdaBoostClassifier(),
    'NN': MLPClassifier(alpha=1, max_iter=1000),
    'KN': KNeighborsClassifier(),
}

# Step 6: Train SVM on train/test split (optional)
models['SVM'].fit(x_train, y_train)
predictions = models['SVM'].predict(x_test)
print("SVM accuracy:", accuracy_score(y_test, predictions))
print("SVM precision:", precision_score(y_test, predictions))
print("SVM recall:", recall_score(y_test, predictions))

# Step 7: K-Fold Cross Validation
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=10)

results = {name: {'accuracy': [], 'precision': [], 'recall': []} for name in models.keys()}

for train_index, test_index in kf.split(X):
    X_train_k, X_test_k = X.iloc[train_index], X.iloc[test_index]
    y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]
    
    for name, model in models.items():
        model.fit(X_train_k, y_train_k)
        y_pred = model.predict(X_test_k)
        
        results[name]['accuracy'].append(accuracy_score(y_test_k, y_pred))
        results[name]['precision'].append(precision_score(y_test_k, y_pred))
        results[name]['recall'].append(recall_score(y_test_k, y_pred))

# Step 8: Calculate average metrics
avg_results = {name: [np.mean(metrics['accuracy']),
                      np.mean(metrics['precision']),
                      np.mean(metrics['recall'])]
               for name, metrics in results.items()}

df_results = pd.DataFrame(avg_results, index=['accuracy', 'precision', 'recall']).T

# Step 9: Visualize results
ax = df_results.plot.bar(rot=0)
plt.ylabel("Score")
plt.title("ML Models Performance")
plt.show()