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

st.title("機械学習アプリ")

st.sidebar.markdown("### 機械学習に用いるcsvファイルを入力してください")


uploaded_files = st.sidebar.text_input("Please enter the URL of the CSV file")
csv = st.sidebar.button("CSV files")
if csv:
      uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files= False)


if uploaded_files:
      df = pd.read_csv(uploaded_files,sep=";")
      df_columns = df.columns
      st.markdown("### 入力データ")
      st.dataframe(df.style.highlight_max(axis=0))
      st.markdown("### 可視化 単変量")
      x = st.selectbox("X軸", df_columns)
      y = st.selectbox("Y軸", df_columns)
      fig = plt.figure(figsize= (12,8))
      plt.scatter(df[x],df[y])
      plt.xlabel(x,fontsize=18)
      plt.ylabel(y,fontsize=18)
      st.pyplot(fig)


      st.markdown("### 可視化 ペアプロット")
      item = st.multiselect("可視化するカラム", df_columns)
      #散布図の色分け基準を１つ選択する。カテゴリ変数を想定
      hue = st.selectbox("色の基準", df_columns)
      
      execute_pairplot = st.button("ペアプロット描画")
      if execute_pairplot:
            df_sns = df[item]
            df_sns["hue"] = df[hue]
            fig = sns.pairplot(df_sns, hue="hue")
            st.pyplot(fig)
            

      st.markdown("### モデリング")
      ex = st.multiselect("説明変数を選択してください（複数選択可）", df_columns)
      ob = st.selectbox("目的変数を選択してください", df_columns)
      al = st.multiselect("アルゴリズムを選択してください（複数選択可）",["重回帰分析","ロジスティック回帰分析","サポートベクトルマシーン","k-NN","決定木","ランダムフォレスト"])
      ak = st.button("実行")

      if "重回帰分析" in al:
                  
                  st.markdown("#### 機械学習を実行します（重回帰分析）")
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
                  col1.metric(label="トレーニングスコア", value=lr_1.score(X_train, y_train))
                  col2.metric(label="テストスコア", value=lr_1.score(X_test, y_test))

      if "ロジスティック回帰分析" in al:
                  st.markdown("#### 機械学習を実行します（ロジスティック回帰分析）")
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
                  col3.metric(label="トレーニングスコア", value=lr_2.score(X_train, y_train))
                  col4.metric(label="テストスコア", value=lr_2.score(X_test, y_test))
      if "サポートベクトルマシーン" in al:
                  st.markdown("#### 機械学習を実行します（サポートベクトルマシーン）")

                        
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
                  col5.metric(label="トレーニングスコア", value=lr_3.score(X_train, y_train))
                  col6.metric(label="テストスコア", value=lr_3.score(X_test, y_test))

      if "決定木" in al:
                  st.markdown("#### 機械学習を実行します（決定木）")
                        
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
                  col7.metric(label="トレーニングスコア", value=lr_4.score(X_train, y_train))
                  col8.metric(label="テストスコア", value=lr_4.score(X_test, y_test))

      if "k-NN" in al:
                  st.markdown("#### 機械学習を実行します（k-NN）")      
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
                  col9.metric(label="トレーニングスコア", value=lr_5.score(X_train, y_train))
                  col10.metric(label="テストスコア", value=lr_5.score(X_test, y_test))
      if "ランダムフォレスト" in al:
                  st.markdown("#### 機械学習を実行します（ランダムフォレスト）")
                        
                  lr_6 = RandomForestClassifier()

      
                  df_ex = df[ex]
                  df_ob = df[ob]
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size = 0.3)
                  lr_6.fit(X_train, y_train)
                  my_bar = st.progress(0)
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)

                  col11, col12 = st.columns(2)
                  col11.metric(label="トレーニングスコア", value=lr_6.score(X_train, y_train))
                  col12.metric(label="テストスコア", value=lr_6.score(X_test, y_test))
                              
    

           
