import pandas as pd
import streamlit as st

df_encoded_=pd.read_csv('df_encoded_.csv')
output_df=pd.read_csv('output_df.csv')

st.write('Predict Skill Category with Bidirectional Long-term Short-term Memory (biLSTM)')

st.subheader('Prediction Result Loss & Accuracy Report')
st.write(output_df)

st.subheader('Results Data')
df_cat_default_type=st.selectbox('Select ID', list(df_encoded_.id.unique()))
                          
df_cat_df=df_encoded_[df_encoded_.id==df_cat_default_type]

st.write(df_cat_df.iloc[:,:2])     
st.write(df_cat_df.iloc[:,3])