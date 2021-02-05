import time
import pandas as pd
import numpy as np
import re
import glob
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from numpy import zeros
import streamlit as st


data3=pd.read_csv('d_group2.csv')
data3=data3.iloc[:,1:]

data3=data3[(data3.occ_group=='technical')|(data3.occ_group=='complex problem solving')]

data3=data3.drop_duplicates(keep='first')

data3['occ_group']=data3['occ_group'].fillna('None')
data3=data3[data3.occ_group!='None']

datalist1=glob.glob('glove_file_*')

combined_datalist=[pd.read_csv(i) for i in datalist1]

words_=list(itertools.chain.from_iterable([list(i.word) for i in combined_datalist]))
values_=list(itertools.chain.from_iterable([list(i.values_) for i in combined_datalist]))

values_=[np.array(re.findall(r'[\d\.]{1,8}',str(i)),dtype='float32') for i in values_]

values_2=[i[:100] for i in values_]

embed_index=dict(zip(words_,values_2))

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(data3.no_stopwords_stemmed)

embedded_skill=word_tokenizer.texts_to_sequences(data3.no_stopwords_stemmed)

vocab_length = len(word_tokenizer.word_index) + 1

embedding_matrix = zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embed_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

len_sent=list([len(i) for i in embedded_skill])
length_long_sentence=max(len_sent)

padded_sentences = pad_sequences(embedded_skill, length_long_sentence, padding='post')

validation_split=.2
indices=np.arange(np.array(padded_sentences,dtype=object).shape[0])
np.random.shuffle(indices)   

data_rand=padded_sentences[indices]

le = preprocessing.LabelEncoder()
labe=[str(i).replace('.xls','') for i in data3.occ_group]

labels_encoded=le.fit(labe)
labels_=le.transform(labe)

labels_rand=np.array(labels_)[indices]

val_sample=int(validation_split * data3.shape[0])
X_train=data_rand[:-val_sample]
y_train=labels_rand[:-val_sample]
X_test=data_rand[-val_sample:]
y_test=labels_rand[-val_sample:]

x_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
x_cv = torch.tensor(X_test, dtype=torch.long)
y_cv = torch.tensor(y_test, dtype=torch.long)

train = torch.utils.data.TensorDataset(x_train, y_train)
valid = torch.utils.data.TensorDataset(x_cv, y_cv)

train_loader = torch.utils.data.DataLoader(train, batch_size=70, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=70, shuffle=False)


class BiLSTM(nn.Module):
  def __init__(self):
    super(BiLSTM, self).__init__()
    self.hidden_size = 164
    drp = 0.4
    n_classes = len(le.classes_)
    self.embedding = nn.Embedding(max_features, embed_size)
    self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
    self.embedding.weight.requires_grad = True
    self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
    self.linear = nn.Linear(self.hidden_size*4 , 164)
    self.relu = nn.ReLU()
    self.out = nn.Linear(164, n_classes)


  def forward(self, x):
    h_embedding = self.embedding(x)
    h_lstm, _ = self.lstm(h_embedding)
    avg_pool = torch.mean(h_lstm, 1)
    max_pool, _ = torch.max(h_lstm, 1)
    conc = torch.cat(( avg_pool, max_pool), 1)
    conc = self.relu(self.linear(conc))
    out = self.out(conc)
    return out
      
embed_size=100
max_features=vocab_length
n_epochs = 7
model = BiLSTM()
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.002, weight_decay=0)
model.cpu()

val_preds1=[]
X1=[]
Y1=[]
ind1=[]
val_loss1=[]
output=[]

for epoch in range(n_epochs):
    start_time = time.time()
    
    model.train()
    avg_loss = 0.  
    for i, (x_batch, y_batch) in enumerate(train_loader):

        
      y_pred = model(x_batch)
      
      loss = loss_fn(y_pred, y_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avg_loss += loss.item() / len(train_loader)
    
    model.eval()        
    avg_val_loss = 0.
    val_preds = np.zeros((len(x_cv),len(le.classes_)))
    X=np.zeros((len(x_cv),length_long_sentence))
    Y=np.zeros((len(x_cv),))
    ind=np.zeros((len(x_cv),))
    val_loss=np.zeros((len(x_cv),))
    
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        
      y_pred = model(x_batch).detach()
      avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
      
      val_preds[i * 70:(i+1) * 70] =F.sigmoid(y_pred).cpu().numpy()
      X[i * 70:(i+1) * 70]=x_batch.cpu().numpy()
      Y[i * 70:(i+1) * 70]=y_batch.cpu().numpy()
      ind[i * 70:(i+1) * 70]=i
      val_loss[i * 70:(i+1) * 70]=avg_val_loss
        
    val_preds1.append(val_preds)
    X1.append(X)
    Y1.append(Y)
    ind1.append(ind)
    val_loss1.append(val_loss)
    
    val_accuracy = sum(val_preds.argmax(axis=1)==y_test)/len(y_test)
    
    
    
    elapsed_time = time.time() - start_time 
    output.append('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy, elapsed_time))
    
    st.write('Epoch {}/{},   loss={:.4f},  val_loss={:.4f},  val_acc={:.4f},  time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy, elapsed_time))
                

output_df=pd.DataFrame([i.split('\t') for i in output])
output_df.columns=['epoch','loss','validation_loss','validation_accuracy','time']
#output_df

Y1_s=[int(i) for i in Y1[3]]
pred=list([i.argmax() for i in val_preds1[3]])

df_encoded=pd.concat([pd.Series(Y1_s, name='actual_class'), 
             pd.Series(pred, name='predicted_class'),
             pd.Series([list(i) for i in X1[3]], name='text')], axis=1)
             
             
word_encoding=word_tokenizer.word_index
word_decoding=dict(zip(list(word_encoding.values()), list(word_encoding.keys())))
word_decoding[0]='None'

skill_decoding=dict(zip(list(labels_), list(labe)))

text2=[]
for i in df_encoded.text:
    text2.append([word_decoding[p] for p in i])
    
text2=[i for i in text2 if i!='None']
df_encoded['text2']=pd.Series(text2)
df_encoded['actual_class_']=[skill_decoding[i] for i in df_encoded['actual_class']]
df_encoded['predicted_class_']=[skill_decoding[i] for i in df_encoded['predicted_class']]
df_encoded_=df_encoded[['actual_class_','predicted_class_','text2']].reset_index()

df_encoded_.columns=['id', 'actual_class_', 'predicted_class_', 'words_']

words3=[]
for i in df_encoded_['words_']:
    words3.append([p for p in re.findall(r'[a-z]{1,20}',str(i))])

words4=[]
for i in words3:
    words4.append([p for p in set(i) if p!='None'])        

df_encoded_['words']= pd.Series(words4)

st.write('Prediction Results for Binary Bidirectional Long-term Short-term Memory (BiLSTM)')

st.subheader('Prediction Results Loss & Accuracy Report')
#st.write(output_df.iloc[:,1:])

st.subheader('Predicted Skill Category for each ID')
df_cat_default_type=st.selectbox('Select ID', list(df_encoded_.id.unique()))
                          
df_cat_df=df_encoded_[df_encoded_.id==df_cat_default_type]

st.write(df_cat_df.iloc[:,:3])
st.write(df_cat_df.iloc[:,3])
         
#st.subheader('Words Matched to Predicted Skill Category')
#st.write(pd.concat([pd.Series(list(range(1,len(df_cat_df.iloc[:,4])+1)),name='index_'), 
           #pd.Series(df_cat_df.iloc[:,3],name='word').explode()],
          #axis=1))
