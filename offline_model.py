import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# MAKE SURE TO RUN: pip install shap
import shap 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
try:
    df = pd.read_csv('studentpredict (1).csv')
    print("Data Loaded Successfully.")
except FileNotFoundError:
    print("Error: 'studentpredict (1).csv' not found. Please check the file path.")
    exit()

# ==========================================
# STEP 0: RENAME COLUMNS (Readability)
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

# 1. Histograms
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
n = len(num_cols)

# CHANGE: Set columns to 6
cols = 6 
rows = (n + cols - 1) // cols

# Adjusted figsize width (cols*3.5) so it fits better on screens
fig, axes = plt.subplots(rows, cols, figsize=(22, rows*3))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col].dropna(), bins=15, color='skyblue', edgecolor='black')
    axes[i].set_title(col, fontsize=9) # Made font slightly smaller to fit
    axes[i].set_ylabel("Count")

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.savefig('1_histograms.png')
print("Saved 1_histograms.png (6 columns)")

# 2. Boxplots
target_cols = ['Weekly Study Hours', 'Last Semester GPA', 'Expected Graduation GPA', 'GRADE']
selected_box = [c for c in target_cols if c in df.columns]
if selected_box:
    fig2, axes2 = plt.subplots(1, len(selected_box), figsize=(5*len(selected_box), 6))
    if len(selected_box) == 1: axes2 = [axes2]
    for ax, col in zip(axes2, selected_box):
        ax.boxplot(df[col].dropna(), patch_artist=True, boxprops=dict(facecolor="lightblue"))
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig('2_boxplots.png')
    print("Saved 2_boxplots.png")

# 3. Correlation Heatmap
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(12, 12))
    cax = ax3.imshow(corr, interpolation='nearest', aspect='auto', cmap='coolwarm')
    fig3.colorbar(cax)
    ax3.set_xticks(range(len(num_cols)))
    ax3.set_yticks(range(len(num_cols)))
    ax3.set_xticklabels(num_cols, rotation=90)
    ax3.set_yticklabels(num_cols)
    ax3.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('3_correlation_matrix.png')
    print("Saved 3_correlation_matrix.png")

# ==========================================
# STEP 3: PREPROCESSING
# ==========================================
if 'STUDENT ID' in df.columns:
    df = df.drop('STUDENT ID', axis=1)

X = df.drop('GRADE', axis=1)
y = df['GRADE']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# STEP 4: FEATURE SELECTION
# ==========================================
selector = SelectKBest(score_func=f_regression, k=10)
selector.fit(X_train_scaled, y_train)
selected_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_indices]
print(f"\nTop 10 Statistical Features: {selected_features.tolist()}")

# ==========================================
# STEP 5: BASELINE MODEL (For Comparison)
# ==========================================
print("\n--- Training Baseline Model ---")
rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
rf_base.fit(X_train_scaled, y_train)
y_pred_base = rf_base.predict(X_test_scaled)
r2_base = r2_score(y_test, y_pred_base)
print(f"Baseline Random Forest R2: {r2_base:.4f}")

# ==========================================
# STEP 6: HYPERPARAMETER TUNING
# ==========================================
print("\n--- Starting Grid Search (Tuning) ---")
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

# ==========================================
# STEP 7: EVALUATE & COMPARE
# ==========================================
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test_scaled)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print("\n" + "="*40)
print("FINAL COMPARISON REPORT")
print("="*40)
print(f"Baseline Model R2: {r2_base:.4f}")
print(f"Tuned Model R2:    {r2_tuned:.4f}")
if r2_tuned > r2_base:
    print(f"RESULT: Tuning IMPROVED accuracy by {r2_tuned - r2_base:.4f}")
else:
    print("RESULT: Default model was already optimal.")
print("="*40)

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_tuned, alpha=0.6, color='darkgreen', label='Tuned')
plt.scatter(y_test, y_pred_base, alpha=0.3, color='blue', marker='x', label='Baseline')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Grade')
plt.ylabel('Predicted Grade')
plt.legend()
plt.title('Actual vs Predicted Grades')
plt.savefig('4_actual_vs_predicted.png')
print("Saved 4_actual_vs_predicted.png")

# ==========================================
# STEP 8: DEEP DIVE VISUALIZATIONS
# ==========================================

# Feature Importance Plot
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 7))
plt.barh(feature_importance_df['Feature'].head(10), feature_importance_df['Importance'].head(10), color='teal')
plt.gca().invert_yaxis()
plt.xlabel('Relative Importance')
plt.title('Top 10 Factors Influencing Grades')
plt.savefig('5_feature_importance.png')
print("Saved 5_feature_importance.png")

# Course ID Impact Plot (The #1 Predictor)
plt.figure(figsize=(12, 6))
df.boxplot(column='GRADE', by='COURSE ID', grid=False, patch_artist=True, 
           boxprops=dict(facecolor="salmon"))
plt.title('Grade Distribution by Course ID')
plt.suptitle('')
plt.xlabel('Course ID')
plt.ylabel('Grade Score')
plt.tight_layout()
plt.savefig('6_course_id_impact.png')
print("Saved 6_course_id_impact.png")


# ==========================================
# STEP 9: SHAP ANALYSIS (NEW)
# ==========================================
print("\n--- Generating SHAP Analysis ---")

# 1. Create the Explainer
# We use the best tuned model
explainer = shap.TreeExplainer(best_model)

# 2. Calculate SHAP values for the Test Set
# (This explains why the model predicted what it predicted for the test students)
shap_values = explainer.shap_values(X_test_scaled)

# 3. Create the Summary Plot
# This "Beeswarm" plot is the industry standard for SHAP
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
plt.savefig('7_shap_summary.png', bbox_inches='tight')
print("Saved 7_shap_summary.png")

# 4. Create a Bar Plot (Average impact magnitude)
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type="bar", show=False)
plt.savefig('8_shap_bar.png', bbox_inches='tight')
print("Saved 8_shap_bar.png")

print("\nAll Steps (1-9) Complete.")