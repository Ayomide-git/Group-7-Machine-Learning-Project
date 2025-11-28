import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap 
import joblib  # <--- NEW: For saving/loading models
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# STEP 10: PREDICTION FUNCTION (EXPORTABLE)
# ==========================================
def predict_student_grade(student_features):
    """
    Loads the trained model artifacts and predicts a grade based on input features.
    This function is safe to call from app.py.
    """
    try:
        # Load artifacts (created during training)
        model = joblib.load('grade_model.pkl')
        scaler = joblib.load('grade_scaler.pkl')
        feature_names = joblib.load('grade_features.pkl')
    except FileNotFoundError:
        return "Error: Model files not found. Please run 'python model.py' first to train the model."

    # 1. Convert input dict to DataFrame
    input_df = pd.DataFrame([student_features])
    
    # 2. Align columns (ensure input matches training data order)
    # We use reindex to handle missing cols (filling with 0) or extra cols (ignoring)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    # 3. Scale the data
    input_scaled = scaler.transform(input_df)
    
    # 4. Predict
    prediction = model.predict(input_scaled)
    return prediction[0]


# ==========================================
# MAIN EXECUTION BLOCK (TRAINING)
# ==========================================
# This block only runs when you execute 'python model.py' directly.
# It will NOT run when you import this file in app.py.
if __name__ == "__main__":

    # ==========================================
    # STEP 1: LOAD DATA
    # ==========================================
    try:
        df = pd.read_csv('studentpredict (1).csv')
        print("Data Loaded Successfully.")
    except FileNotFoundError:
        print("Error: 'studentpredict (1).csv' not found.")
        exit()

    # ==========================================
    # STEP 0: RENAME COLUMNS
    # ==========================================
    rename_map = {
        "Weekly Study Hour's": "Weekly Study Hours",
        "Reading Frequency(Non Scientific)": "Reading Freq (Non-Sci)",
        "Reading Frequency (Scientific Books)": "Reading Freq (Sci)",
        "Attendance to the seminars/conferences related to the department": "Seminar Attendance",
        "Impact of your projects/activities on your success": "Project Impact",
        "Attendance to class": "Class Attendance",
        "Preparation to midterm exams 1": "Midterm 1 Prep",
        "Preparation to midterm exams 2": "Midterm 2 Prep",
        "Taking Notes in Class": "Taking Notes",
        "Listening in Classes": "Listening in Class",
        "Discussion improves my Interest and success in the course": "Discussion Interest",
        "Cumulative grade point in the last Semester": "Last Semester GPA",
        "Expected Cumulative grade point average in the Graduation": "Expected Graduation GPA",
        "Transportation to the university": "Transportation",
        "Accomodation type in Cyprus": "Accommodation"
    }
    df.rename(columns=rename_map, inplace=True)

    # ==========================================
    # STEP 2: EDA & VISUALIZATION
    # ==========================================
    print("--- Generating EDA Diagrams ---")
    
    # (Existing EDA code kept brief for this snippet, but functional)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'STUDENT ID' in df.columns:
        df = df.drop('STUDENT ID', axis=1)

    X = df.drop('GRADE', axis=1)
    y = df['GRADE']

    # ==========================================
    # STEP 3: PREPROCESSING
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # STEP 6: HYPERPARAMETER TUNING (Skipping Base for brevity)
    # ==========================================
    print("\n--- Training Model ---")
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    
    # Reduced CV to 2 for speed in this demo
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # ==========================================
    # STEP 9: SAVE ARTIFACTS (CRITICAL FOR APP)
    # ==========================================
    print("\n--- Saving Model Artifacts ---")
    joblib.dump(best_model, 'grade_model.pkl')
    joblib.dump(scaler, 'grade_scaler.pkl')
    joblib.dump(X.columns, 'grade_features.pkl')
    print("Model, Scaler, and Feature Names saved successfully.")
    
    print("\nTraining Complete. You can now import 'predict_student_grade' in app.py.")