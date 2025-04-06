import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

from urllib.parse import urlparse
import requests
import ipaddress
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sns.set_theme(style="whitegrid")

# App Title
st.title("Phishing url Dashboard")
if not os.path.exists("saved_models"):
    os.makedirs("saved_models", exist_ok=True)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the App mode",
                            ["Dataset Overview", "Interactive EDA", "Model Training", "Model Results", "Predict", "Insight"])

@st.cache_data
def get_data():
    df = pd.read_csv("urldata.csv")
    df['Domain'] = df['Domain'].astype(str).apply(lambda x: x.encode('utf-8').decode('utf-8'))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(df.dtypes)
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

def extract_feature(url):
    result = {}

    #Get the Ip
    try:
        ipaddress.ip_address(url)
        result['Have_IP'] = 1
    except:
        result['Have_IP'] = 0

    print(result)

    #Get HaveAt
    result['Have_At'] = 1 if "@" in url else 0

    #Get urlLength
    result['URL_Length'] = 1 if len(url) >= 54 else 0 

    print(result)
    
    #Get URL_Depth
    s = urlparse(url).path.split('/')
    depth = 0
    for j in range(len(s)):
        if len(s[j]) != 0:
            depth = depth+1
    result['URL_Depth'] = depth

    print(result)

    #Get Redirection
    pos = url.rfind('//')
    if pos > 6:
        if pos > 7:
            result['Redirection'] = 1
        else:
            result['Redirection'] = 0
    else:
        result['Redirection'] = 0

    print(result)

    #HTTPSDomain
    domain = urlparse(url).netloc
    if 'https' in domain:
        result['https_Domain'] = 1
    else:
        result['https_Domain'] = 0

    print(result)

    #tinyURL
    shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"
    match=re.search(shortening_services,url)
    if match:
        result['TinyURL'] = 1
    else:
        result['TinyURL'] = 0

    print(result)

    #Pre/Suffix
    if '-' in urlparse(url).netloc:
        result['Prefix/Suffix'] = 1 
    else:
        result['Prefix/Suffix'] = 0

    print(result)

    try:
        response = requests.get(url)
    except:
        response = ""

    #iFrame
    if response == "":
      result['iFrame'] = 1
    else:
        if re.findall(r"[<iframe>|<frameBorder>]", response.text):
            result['iFrame'] = 0
        else:
            result['iFrame'] = 1
    
    print(result)
    
    #mouseOver
    if response == "" :
        result['Mouse_Over'] = 1
    else:
        if re.findall("<script>.+onmouseover.+</script>", response.text):
            result['Mouse_Over'] = 1
        else:
            result['Mouse_Over'] = 0

    print(result)

    #rightClick
    if response == "":
        result['Right_Click'] = 1
    else:
        if re.findall(r"event.button ?== ?2", response.text):
            result['Right_Click'] = 0
        else:
            result['Right_Click'] = 1

    print(result)
    
    #WebForward
    if response == "":
        result['Web_Forwards'] = 1
    else:
        if len(response.history) <= 2:
            result['Web_Forwards'] = 0
        else:
            result['Web_Forwards'] = 1
    
    print(result)

    return pd.DataFrame([result])


df = get_data()

if "model_results" not in st.session_state:
    st.session_state.model_results = {}

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
    "Label": "1 means Phishing URL and vice versa "
    }
    st.subheader("Column name explain:")

    with st.expander("View Column Descriptions"):
        for col, desc in column_descriptions.items():
            st.markdown(f"**{col}**: {desc}")

    st.subheader("Column Data Types")
    st.write(df.dtypes)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Plot numeric column distributions
    st.subheader("Feature Distributions")
    for column in df.drop(columns=['Domain']).columns:
        count_data = df[column].value_counts().reset_index()
        count_data.columns = [column, 'Count']
        fig = px.bar(count_data, x=column, y='Count', title=f"Distribution of {column}", template="plotly_dark",
                      category_orders={column: [0, 1]})
        st.plotly_chart(fig, use_container_width=True)

elif app_mode == "Interactive EDA":
    st.header("Exploratory Data Analysis")

    EDA_df = get_data()
    EDA_df['Label'] = EDA_df['Label'].map({0: 'Real URL', 1: 'Phishing URL'})
    
    #get columns
    numeric_columns = EDA_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = EDA_df.select_dtypes(include="object").columns.tolist()
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
        model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter)
    
    elif model_type == "SVM":
        C = st.sidebar.selectbox("C", [0.1, 1, 10, 100, 1000])
        gamma = st.sidebar.selectbox("Gamma", [1, 0.1, 0.01, 0.001, 0.0001])
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        degree = st.sidebar.selectbox("Degree (for polynomial kernel)", [2, 3, 4])

        model = SVC(C=C, gamma=gamma, kernel=kernel, degree=degree, probability = True)

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

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            model_key = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.write(f"Model Key: {model_key}")

            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc:.4f}")
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            st.write(f"Precision: {precision:.4f}")
            st.write(f"Recall: {recall:.4f}")
            st.write(f"F1 Score: {f1:.4f}")

            st.subheader("Confusion Matrix:")

            cm = confusion_matrix(y_test, y_pred)
            cm_fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Predicted: No Phishing", "Predicted: Phishing"],
                    y=["True: No Phishing", "True: Phishing"],
                    colorscale='Magma',
                    colorbar=dict(title='Count')
            ))
            cm_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Labels", yaxis_title="True Labels")
            st.plotly_chart(cm_fig)
            

            model_filename = f"saved_models/{model_key}.joblib"
            joblib.dump(model, model_filename)

            st.session_state.model_results[model_key] = {
                "Model Path": model_filename,
                "Parameters": {
                    "Penalty": penalty if model_type == "Logistic Regression" else None,
                    "C": C if model_type == "Logistic Regression" else None,
                    "Solver": solver if model_type == "Logistic Regression" else None,
                    "Max Iterations": max_iter if model_type == "Logistic Regression" else None,
                    "Gamma": gamma if model_type == "SVM" else None,
                    "Kernel": kernel if model_type == "SVM" else None,
                    "Degree": degree if model_type == "SVM" else None,
                    "Criterion": criterion if model_type == "Decision Tree" else None,
                    "Max Depth": max_depth if model_type == "Decision Tree" else None,
                    "Min Samples Split": min_samples_split if model_type == "Decision Tree" else None,
                    "Min Samples Leaf": min_samples_leaf if model_type == "Decision Tree" else None,
                    "Max Features": max_features if model_type == "Decision Tree" else None,
                    "Number of Estimators": n_estimators if model_type == "Random Forest" else None
                },
                "Metrics": {
                    "Accuracy": acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Confusion Matrix": cm
                }
            }

            st.subheader("Model Details")
          
            if model_type == "Logistic Regression":
                log_odds = model.coef_  
                log_odds_series = pd.Series(log_odds[0], index=X_train.columns)
                st.write("Logistic Regression Log Odds:")
                st.write(log_odds_series.sort_values(ascending=False))

                log_odds_fig = px.bar(log_odds_series.sort_values(ascending=False), title="Logistic Regression Log Odds",
                                labels={'index': 'Features', 'value': 'Log Odds'})
                st.plotly_chart(log_odds_fig)

            elif model_type == "Decision Tree":
                
                plt.figure(figsize=(20, 16))
                plot_tree(model, feature_names=X_train.columns.tolist(), filled=True, rounded=True, fontsize=10, max_depth=3)
                plt.title("Decision Tree Structure")
                st.pyplot(plt)

            elif model_type == "Random Forest":
                importance = model.feature_importances_

                importance_fig = px.bar(x=X_train.columns, y=importance, title="Feature Importance",
                                    labels={'x': 'Features', 'y': 'Importance'})
                st.plotly_chart(importance_fig)

            elif model_type == "SVM":
                if model.kernel == "linear":
                    
                    coefficients = model.coef_
                    coef_series = pd.Series(coefficients[0], index=X_train.columns)
                    st.write("SVM Linear Model Feature Coefficients:")
                    st.write(coef_series.sort_values(ascending=False))
        
                    svm_coef_fig = px.bar(coef_series.sort_values(ascending=False), title="SVM Linear Model Feature Coefficients",
                                    labels={'index': 'Features', 'value': 'Coefficient'})
                    st.plotly_chart(svm_coef_fig)
                else:
                    st.write("For non-linear kernels (e.g., RBF, polynomial), detailed interpretation is not straightforward.")
        except Exception as e:  # Catch any exception and display it
            st.write(f"Error occurred: {e}")
            st.write("Please check the selected penalty and solver combination for compatibility.")

elif app_mode == "Model Results":
    st.title("Model Results")
    st.write("Here are the results of your trained models:")

    # Check if model results are available
    if st.session_state.model_results:
        # Iterate over the model results
        sorted_models = sorted(st.session_state.model_results.items(), key=lambda x: x[1]['Metrics'].get('Recall', 0), reverse=True)

        for model_name, result in sorted_models:
            st.markdown(f"<h2 style='color: #FF0000; font-weight: bold; text-shadow: 2px 2px 10px #FF0000;'>{model_name} Results</h2>", unsafe_allow_html=True)

            st.write("### Model Parameters:")
            params = result["Parameters"]
            params = {k: v for k, v in params.items() if v is not None} 
            
            if params:
                st.table(params)
            else:
                st.write("No parameters to display.")

            st.write("### Model Metrics:")
            metrics = result["Metrics"]
            
            metrics = {k: v for k, v in metrics.items() if v is not None and k != "Confusion Matrix"}

            if metrics:
                st.table(metrics)
            else:
                st.write("No metrics to display.")

            cm = result["Metrics"].get("Confusion Matrix")
            if cm is not None:
                cm_fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=["Predicted: No Phishing", "Predicted: Phishing"],
                        y=["True: No Phishing", "True: Phishing"],
                        colorscale='Magma',
                        colorbar=dict(title='Count')
                ))
                cm_fig.update_layout(title=f"{model_name} - Confusion Matrix", xaxis_title="Predicted Labels", yaxis_title="True Labels")
                st.plotly_chart(cm_fig)
                
            st.write("---")
    else:
        st.write("No models have been trained yet.")
elif app_mode == "Predict":
    st.title("URL Prediction")
    st.write("Enter a URL to predict if it's a phishing URL using your trained model.")

    if not st.session_state.model_results:
        st.write("No models available. Please train a model first.")
    else:

        model_name = st.selectbox("Select a trained model:", list(st.session_state.model_results.keys()))
        model_info = st.session_state.model_results[model_name]

        model_path = model_info["Model Path"]
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            st.error("Model file not found. Please retrain the model.")
            st.stop()

        input_url = st.text_input("Enter a URL:")
        
        if st.button("Predict"):
            if not input_url:
                st.warning("Please enter a URL.")
            else:
                feature_df = extract_feature(input_url)

                prob = model.predict_proba(feature_df)[:, 1]  # Get probability of class 1 (Phishing)

                # st.write(f"Prediction Probability: {prob[0]:.2f}")
                threshold = st.session_state.get("custom_threshold", 0.9)

                if prob[0] > threshold:
                    print(threshold)
                    # st.success(f"The entered URL is: **Phishing** (Confidence: {prob[0]:.2f})")
                    st.success(f"The entered URL is: **Phishing**")
                else:
                    # st.success(f"The entered URL is: **Not Phishing** (Confidence: {prob[0]:.2f})")
                    st.success(f"The entered URL is: **Not Phishing**")

elif app_mode == "Insight":
    st.title("Insights from EDA")

    st.markdown("""
    ### Observation from EDA
    All features in the dataset are binary, represented as 0s and 1s.
    
    From the **correlation matrix** and feature charts, features like **`Prefix.Suffix`** and **`MouseOver`** appear to have strong relationships with the target label. These features could be key indicators in identifying phishing URLs.

    Since both the input features and the target are binary, models such as **Decision Tree** and **Random Forest** naturally come to mind for their performance and interpretability. Additionally, **SVM** is considered due to its effectiveness in binary classification and separating decision boundaries.

    In the **Model Results** section, models are sorted by **recall**. This is intentional — missing a phishing URL (false negative) could be more harmful than a false alarm.
    """)

    st.markdown("---")

    st.markdown("""
    ###Dashboard Vision

    This dashboard is designed for **user freedom and flexibility**. Users can:
    - Customize training options for different models,
    - Choose which evaluation metrics or visualizations to view,
    - Predict phishing URLs on-demand.

    The goal is to **empower users** to explore, train, and analyze without unnecessary handholding.
    """)

    with st.expander("🤫 psst..."):
        st.markdown("""
        **The models were REALLY BAD at the 'Predict' step at first...**  
        They would literally label **everything** as phishing — yes, even **https://www.google.com/** 😅.

        Why? Because models were ranked by **Recall** — which is great for catching all phishing URLs, but it also means they panic and flag almost everything as suspicious.

        **The Fix:**  
        We changed the strategy! Instead of relying on hard predictions, we now use prediction **probabilities**. The model must be **at least 90% sure** (`threshold = 0.9`) before it screams "Phishing!"

        This little tweak helped calm the paranoia. You're welcome, Google.
        """)

        if st.checkbox("Want to adjust the phishing detection threshold?"):
            new_threshold = st.slider(
                "Choose your threshold (default is 0.9):",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("custom_threshold", 0.9),
                step=0.01
            )
            st.session_state.custom_threshold = new_threshold
            st.write(f"✅ Current threshold set to: **{new_threshold:.2f}**")
