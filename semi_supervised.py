import pandas as pd
import numpy as np
import pickle
from config import opt

from model import siamese_rnn



############################################################
# read data from pkl file
datapath = opt.processed_data_root
with open(datapath +'train_word.pkl','rb') as f:
    train_q1_word_seq,train_q2_word_seq,y=pickle.load(f)
with open(datapath +'test_word.pkl','rb') as f:
    test_q1_word_seq,test_q2_word_seq = pickle.load(f)
with open(datapath+'word_embedding_matrix.pkl','rb') as f:
    word_embedding_matrix = pickle.load(f)

model = siamese_rnn(word_embedding_matrix,True)

model.load_weights('models/2018-11-20-17-48-12/word_best_model6.h5')

result=model.predict([test_q1_word_seq,test_q2_word_seq])

test = pd.read_csv('./data/raw_data/test.csv')
submit = test
submit['label'] = list(result[:,0])
# submit['label']=submit['label'].apply(lambda x: 1 if x>=0.5 else 0)
# submit['label'] = np.round(submit['label'])
# submit = pd.DataFrame()
# submit['y_pre'] = result[:,0]
submit.to_csv('fake_label.csv',index=False)