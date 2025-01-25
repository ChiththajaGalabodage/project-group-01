import joblib
import pandas as pd

# Step 1: Load the model and vectorizer
model = joblib.load("../models/job_role_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

# Step 2: Load the dataset for evaluation
data_path = "../data/large_job_recommendation_dataset.csv"
df = pd.read_csv(data_path)

# Step 3: Preprocess the evaluation data
df["Combined"] = df["Skills"] + " " + df["Interest_Areas"] + " " + df["Languages"] + " " + df["Tools_Techniques"]
X_eval = vectorizer.transform(df["Combined"])
y_eval = df["Job_Role"]

# Step 4: Evaluate the model
y_pred = model.predict(X_eval)
from sklearn.metrics import classification_report
print("Evaluation Results:")
print(classification_report(y_eval, y_pred))
