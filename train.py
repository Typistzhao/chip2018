import pickle
from config import opt
from utils import *
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

from model import siamese_rnn

from sklearn.metrics import roc_auc_score,f1_score

import time


############################################################
# config of training
is_use_word = False
semi_supervised = False
os.environ["CUDA_VISIBLE_DEVICES"] = opt.NB_CUDA_VISIBLE_DEVICES



############################################################
# read data from pkl file
if is_use_word == True:
    # word
    datapath = opt.processed_data_root
    with open(datapath +'train_word.pkl','rb') as f:
        train_q1_seq,train_q2_seq,y=pickle.load(f)
    with open(datapath +'test_word.pkl','rb') as f:
        test_q1_seq,test_q2_seq = pickle.load(f)
    with open(datapath+'word_embedding_matrix.pkl','rb') as f:
        embedding_matrix = pickle.load(f)
    with open(datapath + 'train_word_features.pkl','rb') as f:
        train_features = pickle.load(f).values
    with open(datapath + 'test_word_features.pkl', 'rb') as f:
        test_features = pickle.load(f).values
else:
    # char
    datapath = opt.processed_data_root
    with open(datapath +'train_char.pkl','rb') as f:
        train_q1_seq,train_q2_seq,y=pickle.load(f)
    with open(datapath +'test_char.pkl','rb') as f:
        test_q1_seq,test_q2_seq = pickle.load(f)
    with open(datapath+'char_embedding_matrix.pkl','rb') as f:
        embedding_matrix = pickle.load(f)
    with open(datapath + 'train_char_features.pkl','rb') as f:
        train_features = pickle.load(f).values
    with open(datapath + 'test_char_features.pkl', 'rb') as f:
        test_features = pickle.load(f).values


############################################################
# train module
model_count = 0
result = np.zeros((len(test_q1_seq),1))
sk_train = np.zeros((train_q1_seq.shape[0], 1))

if semi_supervised == True:
    y_pred  = pd.read_csv('results/ensemble_char_word.csv')['label'].values
    y_pos = y_pred > 0.75
    y_neg = y_pred < 0.25
    y_idx = np.any([y_pos, y_neg], axis=0)
    y_pred = np.round(y_pred[y_idx])
    print(y_idx.shape)

# from features import features_train,features_test


for train_idx, val_idx in StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(y, y):

    val_q1_seq, val_q2_seq  = train_q1_seq[val_idx], train_q2_seq[val_idx]
    train_q1_seq_, train_q2_seq_ = train_q1_seq[train_idx], train_q2_seq[train_idx]
    val_features = train_features[val_idx]
    train_features_ = train_features[train_idx]
    # features_train_, features_val = features_train[train_idx], features_train[val_idx]

    train_y, val_y = y[train_idx], y[val_idx]

    if semi_supervised == True:
        train_q1_seq_ = np.concatenate([train_q1_seq_, test_q1_seq[y_idx]])
        train_q2_seq_ = np.concatenate([train_q2_seq_, test_q2_seq[y_idx]])
        train_y = np.concatenate([train_y,y_pred])
        train_features = np.concatenate([train_features,test_features[y_idx]])

    model = siamese_rnn(embedding_matrix,is_use_word=is_use_word)

    if model_count == 0:
        print(model.summary())
        time_stamp = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        os.mkdir('models/'+time_stamp)


    # callbacks
    RocAuc = RocAucEvaluation(validation_data=([val_q1_seq, val_q2_seq,val_features], val_y), interval=1)
    F1Score = F1ScoreEvaluation(validation_data=([val_q1_seq, val_q2_seq,val_features], val_y), interval=1)
    early_stopping = EarlyStopping(monitor="val_f1", patience=8,mode='max')
    plateau = ReduceLROnPlateau(monitor="val_f1", verbose=1, mode='max', factor=0.8, patience=3)
    best_model_path = "./models/" +time_stamp+ "/word_best_model" + str(model_count) + ".h5"
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True,monitor="val_f1",mode='max')
    csv_logger = CSVLogger('log/log_'+time_stamp+'.csv', append=True, separator=',')

    history_callback = model.fit([train_q1_seq_, train_q2_seq_,train_features_],
                     train_y,
                     validation_data=([val_q1_seq, val_q2_seq,val_features], val_y),
                     epochs=60,
                     batch_size=256,
                     shuffle=True,
                     callbacks=[early_stopping, model_checkpoint,csv_logger,plateau, RocAuc, F1Score], #
                     # sample_weight=sample_weight,
                     verbose=1)

    # loss_history = history_callback.history["loss"]
    # numpy_loss_history = np.array(loss_history)
    # np.savetxt("log/loss_history"+time_stamp+".txt", numpy_loss_history, delimiter=",")

    # model = siamese_rnn(embedding_matrix,is_use_word)
    model.load_weights(best_model_path)
    sk_train[val_idx] = model.predict([val_q1_seq, val_q2_seq, val_features])

    print("************************************")
    print ( "roc_auc_score:\t%.6f \n"%roc_auc_score(val_y, sk_train[val_idx])  )
    print("f1_score:\t%.6f \n" % f1_score(np.round(val_y), np.round(sk_train[val_idx])))
    print("************************************")

    ## predict()
    result += model.predict([test_q1_seq,test_q2_seq,test_features],batch_size=2048)

    model_count += 1


############################################################
# predict module
test = pd.read_csv('./data/raw_data/test.csv')
result/=10
submit = test
submit['label'] = list(result[:,0])
# submit['label']=submit['label'].apply(lambda x: 1 if x>=0.5 else 0)
# submit['label'] = np.round(submit['label'])
if is_use_word == True: flag = 'word'
else:flag = 'char'
submit.to_csv('./results/chip_result_'+flag+'_'+time_stamp+'.csv',index=False)