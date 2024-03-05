import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

st.title("Machine Learning Models")

st.sidebar.markdown("### Please input the CSV file for machine learning")

uploaded_files = st.sidebar.text_input("Please enter the URL of the CSV file")
csv = st.sidebar.button("CSV files")
if csv:
      uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files= False)


if uploaded_files:
      df = pd.read_csv(uploaded_files,sep=";")
      df_columns = df.columns
      st.markdown("### Input Data")
      st.dataframe(df.style.highlight_max(axis=0))
      st.markdown("### Visualization Univariate")
      x = st.selectbox("X-axis", df_columns)
      y = st.selectbox("Y-axis", df_columns)
      fig = plt.figure(figsize= (12,8))
      plt.scatter(df[x],df[y])
      plt.xlabel(x,fontsize=18)
      plt.ylabel(y,fontsize=18)
      st.pyplot(fig)


      st.markdown("### Visualization Pair Plot")
      item = st.multiselect("Select columns to visualize", df_columns)
      # Select one criterion for scatter plot color. Assuming categorical variable
      hue = st.selectbox("Color criterion", df_columns)
      
      execute_pairplot = st.button("Draw Pair Plot")
      if execute_pairplot:
            df_sns = df[item]
            df_sns["hue"] = df[hue]
            fig = sns.pairplot(df_sns, hue="hue")
            st.pyplot(fig)
            

      st.markdown("### Modeling")
      ex = st.multiselect("Select explanatory variables (multiple selection possible)", df_columns)
      ob = st.selectbox("Select the target variable", df_columns)
      al = st.multiselect("Select algorithms (multiple selection possible)",["Multiple Regression","Logistic Regression","Support Vector Machine","k-NN","Decision Tree","Random Forest"])
      ak = st.button("Execute")

      if "Multiple Regression" in al:
                  
                  st.markdown("#### Executing Machine Learning (Multiple Regression)")
                  lr_1 = linear_model.LinearRegression()
                  df_ex = df[ex]
                  df_ob = df[ob]
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=0.3)
                  lr_1.fit(X_train, y_train)
                  my_bar = st.progress(0)
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)

                  col1, col2 = st.columns(2)
                  col1.metric(label="Training Score", value=lr_1.score(X_train, y_train))
                  col2.metric(label="Test Score", value=lr_1.score(X_test, y_test))

      if "Logistic Regression" in al:
                  st.markdown("#### Executing Machine Learning (Logistic Regression)")
                  lr_2 = LogisticRegression()
                  df_ex = df[ex]
                  df_ob = df[ob]
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=0.3)
                  lr_2.fit(X_train, y_train)
                  my_bar = st.progress(0)
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)

                  col3, col4 = st.columns(2)
                  col3.metric(label="Training Score", value=lr_2.score(X_train, y_train))
                  col4.metric(label="Test Score", value=lr_2.score(X_test, y_test))
      if "Support Vector Machine" in al:
                  st.markdown("#### Executing Machine Learning (Support Vector Machine)")
                        
                  lr_3 = LinearSVC()
                  df_ex = df[ex]
                  df_ob = df[ob]
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size = 0.3)
                  lr_3.fit(X_train, y_train)
                  my_bar = st.progress(0)
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)

                  col5, col6 = st.columns(2)
                  col5.metric(label="Training Score", value=lr_3.score(X_train, y_train))
                  col6.metric(label="Test Score", value=lr_3.score(X_test, y_test))

      if "Decision Tree" in al:
                  st.markdown("#### Executing Machine Learning (Decision Tree)")
                        
                  lr_4 = DecisionTreeClassifier()
                  df_ex = df[ex]
                  df_ob = df[ob]
                        
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size = 0.3)
                  lr_4.fit(X_train, y_train)
                  my_bar = st.progress(0)
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)

                  col7, col8 = st.columns(2)
                  col7.metric(label="Training Score", value=lr_4.score(X_train, y_train))
                  col8.metric(label="Test Score", value=lr_4.score(X_test, y_test))

      if "k-NN" in al:
                  st.markdown("#### Executing Machine Learning (k-NN)")      
                  lr_5 = KNeighborsClassifier(n_neighbors=10)
                  df_ex = df[ex]
                  df_ob = df[ob]
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size = 0.3)
                  lr_5.fit(X_train, y_train)
                  my_bar = st.progress(0)
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)

                  col9, col10 = st.columns(2)
                  col9.metric(label="Training Score", value=lr_5.score(X_train, y_train))
                  col10.metric(label="Test Score", value=lr_5.score(X_test, y_test))
      if "Random Forest" in al:
                  st.markdown("#### Executing Machine Learning (Random Forest)")
                        
                  lr_6 = RandomForestClassifier()

      
                  df_ex = df[ex]
                  df_ob = df[ob]
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size = 0.3)
                  lr_6.fit(X_train, y_train)
                  my_bar = st.progress(0)
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)

                  col9, col10 = st.columns(2)
                  col9.metric(label="Training Score", value=lr_5.score(X_train, y_train))
                  col10.metric(label="Test Score", value=lr_5.score(X_test, y_test))
