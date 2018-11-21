from config import opt

from keras import *
from keras.optimizers import *

from utils import *

def siamese_rnn(embedding_matrix,is_use_word):
    if is_use_word:    text_len = opt.MAX_WORD_SEQUENCE_LENGTH
    else:           text_len = opt.MAX_CHAR_SEQUENCE_LENGTH

    embedding_layer = Embedding( len(embedding_matrix),
                   embedding_matrix.shape[1],
                   weights=[embedding_matrix],
                   input_length=text_len,
                   trainable=False)

    norm = BatchNormalization()

    q1_input = Input(shape=(text_len,), dtype="int32")
    q1 = embedding_layer(q1_input)
    q1 = norm(q1)
    q1_embed = SpatialDropout1D(0.2)(q1)

    q2_input = Input(shape=(text_len,), dtype="int32")
    q2 = embedding_layer(q2_input)
    q2 = norm(q2)
    q2_embed = SpatialDropout1D(0.2)(q2)

    char_bilstm_layer1 = Bidirectional(CuDNNLSTM(300, return_sequences=True),merge_mode='sum')
    char_bilstm_layer2 = Bidirectional(CuDNNLSTM(300, return_sequences=True),merge_mode='sum')

    q1_temp,q2_temp = char_bilstm_layer1(q1_embed),char_bilstm_layer1(q2_embed)
    q1,q2 = char_bilstm_layer2(q1_temp),char_bilstm_layer2(q2_temp)

    merged_max = pool_corr(q1, q2, 'max', 'jaccard')
    merged_ave = pool_corr(q1, q2, 'ave', 'jaccard')

    merged = concatenate([merged_ave,merged_max])
    # merged = Dropout(0.2)(merged)
    # merged = BatchNormalization()(merged)
    merged = Dense(200,activation='relu')(merged)
    merged = Dense(200,activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(merged)

    lr=0.0008

    model = Model(inputs=[q1_input,q2_input], outputs=output)

    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr),metrics=['binary_crossentropy','accuracy',f1])
    # model.load_weights("./data/weights_best_0.0008.hdf5")

    return model