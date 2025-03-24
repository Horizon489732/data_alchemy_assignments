import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sns.set(style="whitegrid")

# App Title
st.title("Phishing url Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the App mode",
                            ["Dataset Overview", "Interactive EDA", "Model Training"])

@st.cache_data
def get_data():
    df = pd.read_csv("urldata.csv")
    df['Label'] = df['Label'].map({0: 'Real URL', 1: 'Phishing URL'})
    return df

@st.cache_data
def get_scatter_plot(x_axis, color):
    count_data = df.groupby([x_axis, color]).size().reset_index(name='Count')
    fig = px.scatter(count_data, x=x_axis, y='Count', color=color,
                     title=f"Counts of {x_axis} grouped by {color}",
                     template="plotly_dark",
                     text='Count')
    fig.update_traces(textposition='top center')
    fig.update_layout(width=800, autosize=True)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def get_hist(color_col, hist_column):
    count_data = df.groupby([hist_column, color_col]).size().reset_index(name='Count')
    fig = px.bar(count_data, x=hist_column, y='Count', color=color_col,
                 title=f"Count of {hist_column} grouped by {color_col}",
                 template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def get_box_plot(box_column):
    fig_box = px.box(df, x=box_column, title="Box Plot by Category")
    st.plotly_chart(fig_box, use_container_width=True)


df = get_data()

if app_mode == "Dataset Overview":
    
    st.header("Dataset Preview")
    st.write(df.head())

    # Show dataset dimensions
    st.write("Dataset Dimensions:", df.shape)

    # Explaining columns
    column_descriptions = {
    "Domain": "The domain name of the URL.",
    "Have_IP": "Checks for the presence of IP address in the URL, containing an IP address (1) or not (0).",
    "Have_At": "Checks if '@' is present in the URL (phishing indicator).",
    "URL_Length": "The length of the URL.",
    "URL_Depth": "The depth of the URL based on '/' count.",
    "Redirection": "Indicates if redirection '//' occurs (1: yes, 0: no).",
    "https_Domain": " Checks for the presence of http/https in the domain part of the URL (1: yes, 0: no).",
    "TinyURL": "Indicates if a URL shortening service was used (1: yes, 0: no).",
    "Prefix/Suffix": "Checks for prefix or suffix separated by '-' (1: yes, 0: no).",
    "iFrame": "Checks whether a web page response is suspicious based on the presence of the | character.",
    "Mouse_Over": "Detects suspicious mouse-over events (1: if the response is empty or the event is found, 0: otherwise).",
    "Right_Click": "Checks if right-click is disabled (1: if the response is empty or the event is found, 0: otherwise).",
    "Web_Forwards": "Counts the number of forwards (phishing indicator).",
    }
    st.subheader("Column name explain:")

    with st.expander("View Column Descriptions"):
        for col, desc in column_descriptions.items():
            st.markdown(f"**{col}**: {desc}")
    
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Plot numeric column distributions
    st.subheader("Feature Distributions")
    for column in df.drop(columns=['Domain', 'Label']).columns:
        count_data = df[column].value_counts().reset_index()
        count_data.columns = [column, 'Count']
        fig = px.bar(count_data, x=column, y='Count', title=f"Distribution of {column}", template="plotly_dark",
                      category_orders={column: [0, 1]})
        st.plotly_chart(fig, use_container_width=True)

elif app_mode == "Interactive EDA":
    st.header("Exploratory Data Analysis")
    
    # Load dataset (Iris dataset as an example)
    
    #get columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include="object").columns.tolist()
    categorical_columns.reverse()

    # Chart selection
    chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Histogram", "Box Plot", "Correlation Matrix"])

    if chart_type == "Scatter Plot":
        x_axis = st.selectbox("Select X-axis", numeric_columns, index=0)
        color = st.selectbox("Select color grouping", categorical_columns, index=0)
        get_scatter_plot(x_axis, color)

    elif chart_type == "Histogram":
        hist_column = st.selectbox("Select column for histogram", numeric_columns, index=0)
        color_col = st.selectbox("Select hist grouping color", categorical_columns, index=0)
        get_hist(color_col, hist_column)

    elif chart_type == "Box Plot":
        box_column = st.selectbox("Select column for box plot", categorical_columns, index=0)
        get_box_plot(box_column)

    elif chart_type == "Correlation Matrix":
        st.subheader("Interactive Correlation Matrix")
        fig = px.imshow(
            df.select_dtypes(include=['number']).corr(), text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r",
            origin='lower', title="Correlation Matrix", width=600, height=710
        )
        st.plotly_chart(fig, use_container_width=True)

elif app_mode == "Model Training":
            
    if "show_model_details" not in st.session_state:
        st.session_state.show_model_details = True

    st.header("Model Training")

    model_type = st.sidebar.selectbox("Select Model", ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"])

    X = df.drop(columns=['Domain', 'Label'])
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if model_type == "Logistic Regression":
        penalty = st.sidebar.selectbox("Penalty", ["l2", "elasticnet", "none", "l1"])
        C = st.sidebar.slider("C (Regularization)", min_value=0.0001, max_value=10000.0, value=1.0, step=0.1)
        solver = st.sidebar.selectbox("Solver", ["lbfgs", "newton-cg", "liblinear", "sag", "saga"])
        max_iter = st.sidebar.selectbox("Max Iterations", [100, 1000, 2500, 5000])
        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
    
    elif model_type == "SVM":
        C = st.sidebar.selectbox("C", [0.1, 1, 10, 100, 1000])
        gamma = st.sidebar.selectbox("Gamma", [1, 0.1, 0.01, 0.001, 0.0001])
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        degree = st.sidebar.selectbox("Degree (for polynomial kernel)", [2, 3, 4])

        model = SVC(C=C, gamma=gamma, kernel=kernel, degree=degree)

    elif model_type == "Decision Tree":
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        max_depth = st.sidebar.selectbox("Max Depth", [None, 5, 10, 20])
        min_samples_split = st.sidebar.selectbox("Min Samples Split", [2, 5, 10])
        min_samples_leaf = st.sidebar.selectbox("Min Samples Leaf", [1, 2, 4])
        max_features = st.sidebar.selectbox("Max Features", [None, 'sqrt', 'log2'])

        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, 
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                    max_features=max_features)

    elif model_type == "Random Forest":
        n_estimators = st.sidebar.selectbox("Number of Estimators", [10, 50, 100])
        max_depth = st.sidebar.selectbox("Max Depth", [None, 5, 10])
        min_samples_split = st.sidebar.selectbox("Min Samples Split", [2, 5, 10])
        min_samples_leaf = st.sidebar.selectbox("Min Samples Leaf", [1, 2])
        max_features = st.sidebar.selectbox("Max Features", [None, 'sqrt', 'log2'])

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                    max_features=max_features)

    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {acc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        st.write(cm)

        precision = precision_score(y_test, y_pred, pos_label="Phishing URL")
        recall = recall_score(y_test, y_pred, pos_label="Phishing URL")
        f1 = f1_score(y_test, y_pred, pos_label="Phishing URL")

        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")

        st.subheader("Classification Report")

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='magma', cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        st.pyplot(plt)

        st.subheader("Model Details")
            
        if model_type == "Logistic Regression":
            # Logistic Regression log odds
            log_odds = model.coef_  # This returns a 2D array (for binary classification, shape is (1, n_features))
            log_odds_series = pd.Series(log_odds[0], index=X_train.columns)
            st.write("Logistic Regression Log Odds:")
            st.write(log_odds_series.sort_values(ascending=False))

            # Plot Logistic Regression log odds
            plt.figure(figsize=(10, 5))
            sns.barplot(x=log_odds_series.index, y=log_odds_series.values, palette="coolwarm")
            plt.xticks(rotation=45)
            plt.axhline(0, color="black", linestyle="--")
            plt.title("Logistic Regression Log Odds")
            st.pyplot(plt)

        elif model_type == "Decision Tree":
            # Decision Tree visualization
            plt.figure(figsize=(20, 16))
            plot_tree(model, feature_names=X_train.columns.tolist(), filled=True, rounded=True, fontsize=10, max_depth=3)
            plt.title("Decision Tree Structure")
            st.pyplot(plt)

        elif model_type == "Random Forest":
            # Random Forest feature importance
            importance = model.feature_importances_
            plt.figure(figsize=(10, 5))
            sns.barplot(x=X_train.columns, y=importance)
            plt.xticks(rotation=45)
            plt.title("Feature Importance")
            st.pyplot(plt)

        elif model_type == "SVM":
            if model.kernel == "linear":
                # SVM Linear model coefficient (Feature importance)
                coefficients = model.coef_
                coef_series = pd.Series(coefficients[0], index=X_train.columns)
                st.write("SVM Linear Model Feature Coefficients:")
                st.write(coef_series.sort_values(ascending=False))

                # Plot SVM coefficients
                plt.figure(figsize=(10, 5))
                sns.barplot(x=coef_series.index, y=coef_series.values, palette="coolwarm")
                plt.xticks(rotation=45)
                plt.title("SVM Linear Model Feature Coefficients")
                st.pyplot(plt)
            else:
                st.write("For non-linear kernels (e.g., RBF, polynomial), detailed interpretation is not straightforward.")
