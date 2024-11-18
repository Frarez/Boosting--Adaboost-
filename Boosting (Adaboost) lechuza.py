import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'lechuzasdataset.csv'
lechuzas_data = pd.read_csv(file_path)

lechuzas_data = lechuzas_data.drop(columns=['id'])

lechuzas_data['Potencia_Class'] = pd.qcut(lechuzas_data['Potencia'], q=3, labels=['Baja', 'Media', 'Alta'])

y_lechuzas_class = lechuzas_data['Potencia_Class']

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_lechuzas_class, y_lechuzas_class)

X_train_lechuzas_class, X_test_lechuzas_class, y_train_lechuzas_class, y_test_lechuzas_class = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_lechuzas_class, y_train_lechuzas_class)

best_rf = grid_search.best_estimator_
print("Mejores hiperparámetros:", grid_search.best_params_)

best_rf.fit(X_train_lechuzas_class, y_train_lechuzas_class)

y_pred_lechuzas_class = best_rf.predict(X_test_lechuzas_class)

accuracy_class = accuracy_score(y_test_lechuzas_class, y_pred_lechuzas_class)
report_class = classification_report(y_test_lechuzas_class, y_pred_lechuzas_class, zero_division=1)

print("Precisión del modelo después de las mejoras:", accuracy_class)
print("\nReporte de clasificación:\n", report_class)

conf_matrix = confusion_matrix(y_test_lechuzas_class, y_pred_lechuzas_class)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Baja', 'Media', 'Alta'], yticklabels=['Baja', 'Media', 'Alta'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()
