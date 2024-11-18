import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'lechuzasdataset.csv'  
lechuzas_data = pd.read_csv(file_path)

lechuzas_data = lechuzas_data.drop(columns=['id'])

lechuzas_data['Potencia_Class'] = pd.qcut(lechuzas_data['Potencia'], q=3, labels=['Baja', 'Media', 'Alta'])

X_lechuzas_class = lechuzas_data.drop(columns=['Potencia', 'Potencia_Class'])
y_lechuzas_class = lechuzas_data['Potencia_Class']

X_train_lechuzas_class, X_test_lechuzas_class, y_train_lechuzas_class, y_test_lechuzas_class = train_test_split(
    X_lechuzas_class, y_lechuzas_class, test_size=0.2, random_state=42
)

ada_boost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)

ada_boost_classifier.fit(X_train_lechuzas_class, y_train_lechuzas_class)

y_pred_lechuzas_class = ada_boost_classifier.predict(X_test_lechuzas_class)

accuracy_class = accuracy_score(y_test_lechuzas_class, y_pred_lechuzas_class)
report_class = classification_report(y_test_lechuzas_class, y_pred_lechuzas_class)

print("Precisión del modelo:", accuracy_class)
print("\nReporte de clasificación:\n", report_class)

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test_lechuzas_class, y_pred_lechuzas_class)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Baja', 'Media', 'Alta'], yticklabels=['Baja', 'Media', 'Alta'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for AdaBoost Classifier")
plt.show()
