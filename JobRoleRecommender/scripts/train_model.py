# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Step 1: Load the dataset
data_path = "../data/large_job_recommendation_dataset.csv"  # Path to the dataset
df = pd.read_csv(data_path)

# Step 2: Preprocess the data
df["Combined"] = df["Skills"] + " " + df["Interest_Areas"] + " " + df["Languages"] + " " + df["Tools_Techniques"]

# Step 3: Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Combined"])
y = df["Job_Role"]

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Save the model and vectorizer
joblib.dump(model, "../models/job_role_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")
