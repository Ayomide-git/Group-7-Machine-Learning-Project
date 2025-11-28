import streamlit as st
import pandas as pd
# Import the prediction function from your model.py file
from model import predict_student_grade

# =========================
#   PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide"
)

# =========================
#       HEADER SECTION
# =========================
st.markdown("""
    <h1 style='text-align:center; color:#4CAF50;'>🎓 Student Grade Performance Predictor</h1>
    <p style='text-align:center; font-size:18px;'>
        Enter the student's academic details below to predict their final grade using the AI model.
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
#      INPUT FORM
# =========================
# We organize inputs to match the columns expected by model.py
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📚 Study Habits")
    
    # Study Time
    weekly_study_hours = st.slider("Weekly Study Hours (1-5)", 1, 5, 3)
    
    # Reading Frequencies
    reading_non_sci = st.selectbox(
        "Reading Freq (Non-Sci)", 
        [1, 2, 3], 
        help="1: None, 2: Sometimes, 3: Often"
    )
    reading_sci = st.selectbox(
        "Reading Freq (Scientific)", 
        [1, 2, 3], 
        help="1: None, 2: Sometimes, 3: Often"
    )
    
    # Class Behavior
    taking_notes = st.selectbox(
        "Taking Notes in Class", 
        [1, 2, 3], 
        help="1: Never, 2: Sometimes, 3: Always"
    )
    listening_class = st.selectbox(
        "Listening in Class", 
        [1, 2, 3], 
        help="1: Never, 2: Sometimes, 3: Always"
    )

with col2:
    st.subheader("🏫 Academic Performance")
    
    # Exams
    midterm1 = st.slider("Midterm 1 Preparation (1-5)", 1, 5, 3)
    midterm2 = st.slider("Midterm 2 Preparation (1-5)", 1, 5, 3)
    
    # GPA
    last_semester_gpa = st.number_input(
        "Last Semester GPA", 
        min_value=0.0, 
        max_value=4.0, 
        value=2.5, 
        step=0.1
    )
    expected_grad_gpa = st.number_input(
        "Expected Graduation GPA", 
        min_value=0.0, 
        max_value=4.0, 
        value=3.0, 
        step=0.1
    )
    
    # Attendance
    class_attendance = st.slider("Class Attendance (1-5)", 1, 5, 4)

with col3:
    st.subheader("🌍 Social & Environment")
    
    # Social
    seminar_attendance = st.selectbox(
        "Seminar Attendance", 
        [0, 1], 
        help="0: No, 1: Yes"
    )
    project_impact = st.selectbox(
        "Project Impact on Success", 
        [1, 2, 3], 
        help="1: Low, 2: Medium, 3: High"
    )
    discussion_interest = st.selectbox(
        "Discussion Interest", 
        [1, 2, 3], 
        help="1: Low, 2: Medium, 3: High"
    )
    
    # Logistics
    transportation = st.selectbox(
        "Transportation Type", 
        [1, 2, 3, 4], 
        help="1: Bus, 2: Private, 3: Walk, etc."
    )
    accommodation = st.selectbox(
        "Accommodation Type", 
        [1, 2, 3], 
        help="1: Dorm, 2: Rental, 3: Family"
    )

# =========================
#     PREDICTION LOGIC
# =========================
st.markdown("---")

# Center the button using 3 columns
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    if st.button("🔮 Predict Final Grade", use_container_width=True):
        
        # 1. Gather all inputs into a dictionary
        student_data = {
            "Weekly Study Hours": weekly_study_hours,
            "Reading Freq (Non-Sci)": reading_non_sci,
            "Reading Freq (Sci)": reading_sci,
            "Seminar Attendance": seminar_attendance,
            "Project Impact": project_impact,
            "Class Attendance": class_attendance,
            "Midterm 1 Prep": midterm1,
            "Midterm 2 Prep": midterm2,
            "Taking Notes": taking_notes,
            "Listening in Class": listening_class,
            "Discussion Interest": discussion_interest,
            "Last Semester GPA": last_semester_gpa,
            "Expected Graduation GPA": expected_grad_gpa,
            "Transportation": transportation,
            "Accommodation": accommodation
        }

        # 2. Call the imported function
        with st.spinner("Analyzing student profile..."):
            prediction = predict_student_grade(student_data)

        # 3. Display Result
        if isinstance(prediction, str) and "Error" in prediction:
            st.error(prediction)
        else:
            # We format the prediction to 2 decimal places inside the f-string
            st.markdown(f"""
                <div style='padding:25px; border-radius:15px; background-color:#F0FFF0; text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h3 style='color:#2E7D32; margin:0;'>Predicted Final Grade</h3>
                    <h1 style='color:#1B5E20; font-size: 48px; margin: 10px 0;'>{prediction:.2f}</h1>
                    <p style='color:#555;'>Based on the current academic metrics provided.</p>
                </div>"""
            , unsafe_allow_html=True)
            
            # Contextual message based on grade
            if prediction >= 3.0: 
                st.success("🌟 This student is on track for excellent performance!")
            elif prediction >= 2.0:
                st.info("⚖️ This student is performing steadily.")
            else:
                st.warning("⚠️ This student may require additional support.")

# =========================
#       FOOTER
# =========================
st.markdown("---")
st.caption("Powered by Random Forest Regression | Model trained on 'studentpredict (1).csv'")