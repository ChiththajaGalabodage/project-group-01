import joblib

# Step 1: Load the model and vectorizer
model = joblib.load("../models/job_role_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

# Step 2: Get user input (Example user data)
user_input = ["Python, Data Analysis Data Science Python Pandas, NumPy"]

# Step 3: Preprocess and predict
user_vectorized = vectorizer.transform(user_input)
predicted_role = model.predict(user_vectorized)

print("Predicted Job Role:", predicted_role[0])
