import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as pxr

sns.set(style="whitegrid")

# App Title
st.title("Phishing url Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the App mode",
                            ["Interactive EDA"])

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
                     template="plotly_dark")
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
    

if app_mode == "Interactive EDA":
    st.header("Exploratory Data Analysis")
    
    # Load dataset (Iris dataset as an example)
    
    df = get_data()
    
    st.subheader("Dataset Preview")
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

    
    #get columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include="object").columns.tolist()
    categorical_columns.reverse()

    # scatter plot with Plotly
    x_axis = st.selectbox("Select X-axis", numeric_columns, index=0)
    color = st.selectbox("Select color grouping", categorical_columns, index=0)
    get_scatter_plot(x_axis, color)


    # Interactive histogram
    st.subheader("Interactive Histogram")
    hist_column = st.selectbox("Select column for histogram", numeric_columns, index=0)
    color_col = st.selectbox("Select hist grouping color", categorical_columns, index=0)
    get_hist(color_col,hist_column)


    # Interactive Box Plot
    st.subheader("Interactive Boxplot")
    box_column = st.selectbox("Select column for box plot", categorical_columns, index=0)
    get_box_plot(box_column)

    # Interactive Corr Matrix
    st.subheader("Interactive Correlation Matrix")
    fig = px.imshow(
    df.select_dtypes(include=['number']).corr(),text_auto=True,aspect="auto",
    color_continuous_scale="RdBu_r",  # Red-blue color scale for positive/negative correlations
    origin='lower',title="Correlation Matrix",width = 600, height=710
    )

    st.plotly_chart(fig, use_container_width=True)
