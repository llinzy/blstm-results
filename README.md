# NLP Deep learning model with interactive visualization 

Deploying deep learning models directly through Heroku can be slow depending on the size of the model.
To overcome this limitation, this demonstration utilizes Streamlit to visualize the the results of a large deep learning model in Heroku rather than running it directly.

Steps:
1. Run the deep learning model locally and save results.
   Model location: blstm_model/blstm_.py
2. Deploy a Streamlit application to visualize model results on Heroku. 

The Streamlit application for this model can be viewed here: https://bilstm.herokuapp.com/
You are welcome to run the deep learning model in heroku directly here: https://nlp-deep-learning.herokuapp.com/

Data: online job postings broken into skill categories as designated by ONET online. Preprocessing: 1. emails removed using regex 2. punctuation removed 3. lowercase transformation 4. stemmed
