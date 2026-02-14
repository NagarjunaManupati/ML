import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from model.logistic_regression import train_and_evaluate as lr
from model.decision_tree import train_and_evaluate as dt
from model.knn import train_and_evaluate as knn
from model.naive_bayes import train_and_evaluate as nb
from model.random_forest import train_and_evaluate as rf
from model.xgboost_model import train_and_evaluate as xgb

st.set_page_config(page_title="CVD Classification")
st.title("🫀 Cardiovascular Disease Classification")

uploaded_file = st.file_uploader(
    "Upload Cardiovascular Disease CSV (Test Data Only)",
    type=["csv"]
)

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

# Target column input
target_column = st.text_input(
    "Target Column Name",
    value="HeartDisease",
    help="Enter the name of your target/label column"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
                
    # Display dataset info
    with st.expander("📊 Dataset Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Features", df.shape[1] - 1)
    
    #df = pd.read_csv(uploaded_file, sep=';')
    # REQUIRED_COLUMNS = {
    #     'age','gender','height','weight','ap_hi','ap_lo',
    #     'cholesterol','gluc','smoke','alco','active','cardio'
    # }

    # if not REQUIRED_COLUMNS.issubset(df.columns):
    #     st.error("❌ Invalid dataset. Upload Cardiovascular Disease CSV only.")
    #     st.stop()

    # df['age'] = df['age'] / 365.25
    # X = df.drop('cardio', axis=1)
    # y = df['cardio']
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if st.button("Run Model"):
        model_map = {
            "Logistic Regression": lr,
            "Decision Tree": dt,
            "KNN": knn,
            "Naive Bayes": nb,
            "Random Forest": rf,
            "XGBoost": xgb
        }

        y_pred, metrics = model_map[model_name](
            X_train, X_test, y_train, y_test
        )

        st.subheader("📊 Evaluation Metrics")
        st.table(pd.DataFrame(metrics, index=["Score"]).T)

        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)




