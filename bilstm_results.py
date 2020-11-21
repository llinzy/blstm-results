import pandas as pd
import streamlit as st

df_encoded_=pd.read_csv('df_words_matched.csv')
output_df=pd.read_csv('output_df.csv')

st.write('Prediction Results for Binary Bidirectional Long-term Short-term Memory (BiLSTM)')

st.subheader('Prediction Results Loss & Accuracy Report')
st.write(output_df.iloc[:,1:])

st.subheader('Predicted Skill Category for each ID')
df_cat_default_type=st.selectbox('Select ID', list(df_encoded_.id.unique()))
                          
df_cat_df=df_encoded_[df_encoded_.id==df_cat_default_type]

st.write(df_cat_df.iloc[:,1:4])
         
st.subheader('Words Matched to Predicted Skill Category')

words_matched=pd.concat([pd.Series(list(range(1,len(df_cat_df.iloc[:,4])+1)),name='index_'), 
           pd.Series(df_cat_df.iloc[:,4],name='word').explode()],
          axis=1)
words_matched=words_matched.fillna('_')
st.write(''.join(words_matched.word).split(','))
