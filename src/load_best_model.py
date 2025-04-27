import mlflow
import numpy as np

# Set tracking URI
mlflow.set_tracking_uri('https://dagshub.com/nafiul-araf/MLFlow.mlflow')

# Load the model by name and stage (or version)
model_name = "Random Forest Model (diabetes)"  # your model name

# Option 1: Load by stage (if you have "stage" tagging manually, you have to know version manually instead)
model_version = 3  # Put your correct version here (you know it's 2 from your earlier code)

# Load the model
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri=model_uri)

prediction = model.predict(np.array([[1, 3, 12, 11, 12,9, 8, 3, 1, 12]]))
print(f"Prediction: {prediction}")