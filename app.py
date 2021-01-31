#Made by Mainak Chaudhuri

import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io


#---------------------------------#




st.set_page_config(page_title='The Machine Learning Premier League',
    layout='wide')


#---------------------------------#
# Model building
def build_model(df):
    df = df.loc[:100] 
    X = df.iloc[:,:-1] 
    Y = df.iloc[:,-1] 

    st.markdown('**Dimension of dataset(Number of rows and columns)**')
    st.write('X (Independent Axis)')
    st.info(X.shape)
    st.write('Y (Dependent Axis)')
    st.info(Y.shape)

    st.markdown('**Details of variables**:')
    st.write('X variable (first 10 are shown)')
    st.info(list(X.columns[:10]))
    st.write('Y variable')
    st.info(Y.name)

   
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
    reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
    models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

    st.subheader('Model Performance Chart')

    st.write('Training set')
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

    st.write('Test set')
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

    st.subheader('Plotting Model Performance (Test set)')


    with st.markdown('**R-squared Score**'):
       
        predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="darkgrid")
        ax1 = sns.barplot(y=predictions_test.index, x="R-Squared score", data=predictions_test)
        ax1.set(xlim=(0, 1))
    st.markdown(imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
        
    plt.figure(figsize=(12, 3))
    sns.set_theme(style="darkgrid")
    ax1 = sns.barplot(x=predictions_test.index, y="R-Squared score", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-r2-wide.pdf'), unsafe_allow_html=True)

    with st.markdown('**RMSE Scores**'):
      
        predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"] ]
        plt.figure(figsize=(12, 9))
        sns.set_theme(style="darkgrid")
        ax2 = sns.barplot(y=predictions_test.index, x="RMSE score", data=predictions_test)
    st.markdown(imagedownload(plt,'plot-rmse-tall.pdf'), unsafe_allow_html=True)
      
    plt.figure(figsize=(12, 3))
    sns.set_theme(style="darkgrid")
    ax2 = sns.barplot(x=predictions_test.index, y="RMSE score", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-rmse-wide.pdf'), unsafe_allow_html=True)

    with st.markdown('**Calculation time**'):
       
        predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="darkgrid")
        ax3 = sns.barplot(y=predictions_test.index, x="Time Taken (in ms)", data=predictions_test)
    st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
       
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="darkgrid")
    ax3 = sns.barplot(x=predictions_test.index, y="Time Taken (in ms)", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)




def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

#---------------------------------#
st.write("""
# ⭐ Machine Learning Premier League. ⭐

## Find the Winning Model for your dataset. 🏆🥇🥈🥉

""")

#---------------------------------#

with st.sidebar.header('File Uploader Section'):
    uploaded_file = st.sidebar.file_uploader("Upload an input as CSV file", type=["csv"])
    


with st.sidebar.header('Set the optimization parameters\n (Grab the slider and set to any suitable point)'):
    
    split_size = st.sidebar.slider('Data split ratio :', 0, 100, 70, 5)
    seed_number = st.sidebar.slider('Set the random-seed-value :', 0, 100, 50, 1)
    
with st.sidebar.header('Project made by:'):
    st.write("Made by: MAINAK CHAUDHURI")
        
#---------------------------------#

st.subheader('Dataset display')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**Snap of the dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Upload a file')
    st.info('OR')
    if st.button('Use preloaded data instead'):
        st.info("Dataset used : Pima diabetes")

       
        diabetes = load_diabetes()
       
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names).loc[:100] 
        Y = pd.Series(diabetes.target, name='response').loc[:100] 
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('Displaying results form a sample preloaded data :')
        st.write(df.head(5))

        build_model(df)
