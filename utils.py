import pandas as pd
import keras.backend.tensorflow_backend as K
from keras.layers import *
from keras.callbacks import *
from tqdm import tqdm
import re
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score




def norm_layer(x, axis=1):
    return (x - K.mean(x, axis=axis, keepdims=True)) / K.std(x, axis=axis, keepdims=True)

def distance(q1,q2,dist,normlize=False):
    if normlize:
        q1 = Lambda(norm_layer)(q1)
        q2 = Lambda(norm_layer)(q2)

    if dist == 'cos':
        return multiply([q1,q2])

    elif dist == 'dice':
        def dice(x):
            return x[0]*x[1]/(K.sum(x[0]**2,axis=1,keepdims=True)+K.sum(x[1]**2,axis=1,keepdims=True))
        return Lambda(dice)([q1,q2])

    elif dist == 'jaccard':
        def jaccard(x):
            return  x[0]*x[1]/(
                    K.sum(x[0]**2,axis=1,keepdims=True)+
                    K.sum(x[1]**2,axis=1,keepdims=True)-
                    K.sum(K.abs(x[0]*x[1]),axis=1,keepdims=True))
        return Lambda(jaccard)([q1,q2])

def pool_corr(q1,q2,pool_way,dist):
    if pool_way == 'max':
        pool = GlobalMaxPooling1D()
    elif pool_way == 'ave':
        pool = GlobalAveragePooling1D()
    else:
        raise RuntimeError("don't have this pool way")

    q1 = pool(q1)
    q2 = pool(q2)

    merged = distance(q1,q2,dist,normlize=True)

    return merged

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=2048)

            y_pred = np.array(y_pred)
            y_pred = y_pred

            real = np.array(self.y_val)
            score = roc_auc_score(real, y_pred)
            print("|-----------------------------------------------")
            print("| ROC-AUC - epoch: %d - score: %.6f " % (epoch + 1, score))

class F1ScoreEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=2048)
            y_pred = np.array(y_pred)
            y_pred = np.round(y_pred)
            real = np.array(self.y_val)
            real = np.round(real)

            _val_f1 = f1_score(real, y_pred)
            _val_recall = recall_score(real, y_pred)
            _val_precision = precision_score(real, y_pred)
            print('| Val_F1: %.4f --Val_Precision: %.4f --Val_Recall: %.4f'%(_val_f1, _val_precision, _val_recall))
            print('|-----------------------------------------------\n')

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    return 2*K.sum(y_true * y_pred) / (K.sum(y_true) + K.sum(y_pred))

def cluster_pos(file='train'):


    tr = pd.read_csv('./data/raw_data/'+file+'.csv')
    # tr = tr.append(pd.read_csv('save_sample.csv')).reset_index(drop=True)
    y_idx = tr['label'] == 1


    # y_pred = pd.read_csv('./149367.csv')['y_pre'].values
    # y_pos = y_pred < 1
    # y_neg = y_pred > 0.5
    # y_idx = np.logical_and(y_pos,y_neg)


    tr = tr.loc[y_idx,['qid1','qid2']]
    print(tr.shape)
    tr = tr.sort_values(by=['qid1', 'qid2']).values
    for i in range(len(tr)):  # 去重：保证 qid1>qid2 即可去重
        if tr[i][0]>tr[i][1]:
            tr[i] = [tr[i][1],tr[i][0]]

    q2group = {}
    num_group = 0
    error = 0
    for q1,q2 in tr:
        assert q1 < q2
        if q1 not in q2group and q2 not in q2group:
            q2group[q1] = num_group
            q2group[q2] = num_group
            num_group += 1
        elif q2 in q2group and q1 not in q2group:
            q2group[q1] = q2group[q2]
        elif q1 in q2group and q2 not in q2group:
            q2group[q2] = q2group[q1]
        else:
            if q2group[q2] != q2group[q1]:
                error+=1
    print(error)
    while error != 0:
        for q1,q2 in tr:
            if q2group[q1] != q2group[q2]:
                group_id = min(q2group[q1],q2group[q2])
                q2group[q1] = group_id
                q2group[q2] = group_id
        error = 0
        for q1,q2 in tr:
            if q2group[q1] != q2group[q2]:
                error+=1
        print(error)


    with open('./info/q2group.json','w') as f:
        f.write(json.dumps(q2group,sort_keys=True,indent=4, separators=(',', ': ')))

    group2q = [{} for i in range(num_group)]
    for q,g_id in q2group.items():
        group2q[g_id][q] = 1

    with open('./info/group2q.json', 'w') as f:
        f.write(json.dumps(group2q, sort_keys=True, indent=4, separators=(',', ': ')))


    group_n = {}
    for i,q in enumerate(group2q):
        group_n[str(i)] = len(q)
    group_n = sorted(group_n.items(),key=lambda x:x[1])
    with open('./info/group_samples_num.json', 'w') as f:
        f.write(json.dumps(group_n, sort_keys=True, indent=4, separators=(',', ': ')))

def cluster_neg():

    tr = pd.read_csv('./data/raw_data/train.csv')
    tr = tr.loc[tr['label'] == 0, ['qid1', 'qid2']].values

    with open('./info/q2group.json','r') as f:
        q2group = json.loads(f.read())

    neg_pair = {}
    for q1,q2 in tr:
        if q1 in q2group and q2 in q2group:
            if q2group[q1]<q2group[q2]:
                neg_pair[str(q2group[q1])+'_'+str(q2group[q2])] = 1
            elif q2group[q1]>q2group[q2]:
                neg_pair[str(q2group[q2])+'_'+str(q2group[q1])] = 1

    with open('./info/neg_rule.json','w') as f:
        f.write(json.dumps(neg_pair,sort_keys=True,indent=4,separators=(',', ': ')))

    te = pd.read_csv('./data/raw_data/test.csv')[['qid1','qid2']].values
    need_rule = {}
    for q1, q2 in te:
        if q1 in q2group and q2 in q2group:
            if q2group[q1] < q2group[q2]:
                pair = str(q2group[q1]) + '_' + str(q2group[q2])
            elif q2group[q1] > q2group[q2]:
                pair = str(q2group[q2]) + '_' + str(q2group[q1])
            else:
                continue
            if pair not in neg_pair:
                if pair not in need_rule:
                    need_rule[pair] = 0
                need_rule[pair]+=1
    need_rule = sorted(need_rule.items(),key=lambda x:x[1])
    with open('./info/need_rule.json','w') as f:
        f.write(json.dumps(need_rule,sort_keys=True,indent=4,separators=(',', ': ')))

def post_process(file,output):

    with open('./info/q2group.json','r') as f:
        q2group = json.loads(f.read())

    with open('./info/group2q.json','r') as f:
        group2q = json.loads(f.read())

    te = pd.read_csv('./data/raw_data/test.csv',usecols=['qid1','qid2']).values
    y_pre = pd.read_csv(file)


    "正例修正"
    n = 0
    loss = 0

    save_samples = []
    s = 0
    for i, (q1, q2) in enumerate(te):
        if q1 in q2group and q2 in q2group:
            if q2group[q1] == q2group[q2]:
                n += 1
                loss = loss - np.log(y_pre.iloc[i,0])
                y_pre.iloc[i, 0] = 1
                save_samples.append([1,q1, q2])

    # save_samples = pd.DataFrame(save_samples,columns=['label','q1','q2'])
    # save_samples.to_csv('./info/save_sample.csv',index=False)
    print('n:',n)

    print(s)
    "负例修正"
    with open('./info/neg_rule.json','r') as f:
        neg_pair = json.loads(f.read())
    n = 0
    for i, (q1, q2) in tqdm(enumerate(te)):
        if q1 in q2group and q2 in q2group:
            if q2group[q1] < q2group[q2]:
                pair = str(q2group[q1]) + '_' + str(q2group[q2])
            elif q2group[q1] > q2group[q2]:
                pair = str(q2group[q2]) + '_' + str(q2group[q1])
            else:
                pair = ''
            if pair in neg_pair:
                loss = loss - np.log(1-y_pre.iloc[i, 0])
                y_pre.iloc[i, 0] = 0
                n += 1
    print('loss:', loss / len(te))
    print(n)

    y_pre.to_csv(output, index=False)

    return y_pre

# 公共个数
def num_of_common(q1,q2):
    t1 = np.asarray(re.split(' ',q1))
    t2 = np.asarray(re.split(' ',q2))
    return len(np.intersect1d(t1,t2))

# 最长公共子序列
def lcs_length(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]

# 编辑距离
def levenshtein(q1, q2):
    len_q1 = len(q1) + 1
    len_q2 = len(q2) + 1
    #create matrix
    matrix = [0 for n in range(len_q1 * len_q2)]
    #init x axis
    for i in range(len_q1):
        matrix[i] = i
    #init y axis
    for j in range(0, len(matrix), len_q1):
        if j % len_q1 == 0:
            matrix[j] = j // len_q1
    for i in range(1, len_q1):
        for j in range(1, len_q2):
            if q1[i-1] == q2[j-1]:
                cost = 0
            else:
                cost = 1
            matrix[j*len_q1+i] = min(matrix[(j-1)*len_q1+i]+1,
                    matrix[j*len_q1+(i-1)]+1,
                    matrix[(j-1)*len_q1+(i-1)] + cost)
    return matrix[-1]

# if __name__=='__main__':
    # cluster_pos()
    # cluster_neg()
    # post_process('fake_label.csv',output='fake_label_after.csv')