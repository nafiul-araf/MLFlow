import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://127.0.0.1:5000')
# load dataset
wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# parameters for random forest
max_depth = 10
n_estimators = 50

with mlflow.start_run():
  rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
  rf.fit(X_train, y_train)
  train_score = rf.score(X_train, y_train)
  y_pred = rf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  mlflow.log_metrics({'training_score': train_score, 'accuracy_score': accuracy})
  mlflow.log_params({'max_depth': max_depth, 'n_estimators': n_estimators})

  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(10, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.savefig('confusion_matrix.png')

  mlflow.log_artifact('confusion_matrix.png')
  mlflow.log_artifact(__file__)

  print(f"Training score {train_score} and accuracy {accuracy}")