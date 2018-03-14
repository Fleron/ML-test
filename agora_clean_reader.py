import pandas as pd
#import tensorflow as tf
import numpy as np
import re
import win_unicode_console
import matplotlib.pyplot as plt
import itertools
import datetime
from collections import namedtuple as nt


from gensim.models import Word2Vec, Doc2Vec, FastText
from gensim.models import KeyedVectors

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.layers import Dense, Embedding, Flatten, Dropout, Conv1D, LSTM, MaxPooling1D, GRU
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD

##########################################################
# Static variables and set of random seeds.              #
# Win_unicode is to prevent console overflow on windows. #
# Comment out (aswell as import) on other machine        #
##########################################################
win_unicode_console.enable()
np.random.seed(7)


UNIQUE_WORDS = 10866
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 100
NMBR_CATEGORIES = 6
EPOCHS = 6
BATCHSIZE = 128

#unique_words = 15573
#EMBEDDING_DIM = 300
#MAX_SEQUENCE_LENGTH = 50
##################################################################
# Text setup. Clean text and get equal amounts of each category. #
# Also extra functions for sentence checking.                    #
# Finally splitting data in to train,validation and test.        #
##################################################################
def get_equal_for_each_cat(data_set,max_length):
    max_length = int(max_length/3)
    print("Splitting dataset for equal size of each category...")
    cat_list = ['Counterfeits/Watches','Other','Services/Money','Drugs/Opioids/Oxycodone','Data/Accounts','Information/Guides']
    data_set = data_set.loc[data_set[' Category'].isin(cat_list)]
    #randomize order of dataset
    data_set = data_set.sample(frac=1).reset_index(drop=True)

    data_set_1 = data_set.loc[data_set[' Category'] == cat_list[0]]
    data_set_2 = data_set.loc[data_set[' Category'] == cat_list[1]]
    data_set_3 = data_set.loc[data_set[' Category'] == cat_list[2]]
    data_set_4 = data_set.loc[data_set[' Category'] == cat_list[3]]
    data_set_5 = data_set.loc[data_set[' Category'] == cat_list[4]]
    data_set_6 = data_set.loc[data_set[' Category'] == cat_list[5]]
    #if maximum length of each category
    #is needed this is set with a number above 0
    if max_length > 0:
        data_set_1 = data_set_1[:max_length]
        data_set_2 = data_set_2[:max_length]
        data_set_3 = data_set_3[:max_length]
        data_set_4 = data_set_4[:max_length]
        data_set_5 = data_set_5[:max_length]
        data_set_6 = data_set_6[:max_length]

    data_frames = [data_set_1,data_set_2,data_set_3,data_set_4,data_set_5,data_set_6]
    dataset = pd.concat(data_frames)
    dataset = clean_text_language(dataset)
    print('dataset split complete...')
    return dataset
def count_longest_sentence(x):
    lengths = [len(a) for a in x]
    max_length = max(lengths)
    return max_length
def clean_text_language(dataset):
    #removing unnecessary signs in text
    #dataset['text'] = dataset['text'].str.replace('.','')
    #dataset['text'] = dataset['text'].str.replace('!','')
    #dataset['text'] = dataset['text'].str.replace('?','')
    return dataset
def split_data(data,encodes):
    print("random sampling...")
    data = data.sample(frac=1).reset_index(drop=True)
    print("splitting data into x and y...")
    X = data[' Item Description']
    Y = data[' Category']
    x = [text_to_word_sequence(i) for i in X]
    max_len = count_longest_sentence(x)
    print('longest sentence: ',max_len)
    x, word_idx = encode_x(x,encodes)

    print('number of unique words: ', len(word_idx))
    #UNIQUE_WORDS = len(word_idx)

    y = encode_y(Y)
    NMBR_CATEGORIES = len(y[0])
    print("label encoding complete...")
    train_split = int(0.8*len(data))
    val_split = int(0.1*len(data))
    x_train = x[:train_split]
    y_train = y[:train_split]
    cut_point = train_split + val_split
    x_val = x[train_split:cut_point]
    y_val = y[train_split:cut_point]
    x_test = x[cut_point:]
    y_test = y[cut_point:]
    print("splitting data finished.")
    return x_train,y_train,x_val,y_val,x_test,y_test,word_idx

##########################################################
# Create feature representation.                         #
# embedding_case 1 = word2vec, 2 = fasttext, 3 = doc2vec #
# delete model after creation to free up memory.         #
##########################################################
def text_setup_for_feature_representation(dataset,embedding_case):
    dataset = get_equal_for_each_cat(dataset,0)
    dataset = clean_text_language(dataset)
    x = [text_to_word_sequence(k) for k in dataset[' Item Description']]
    #x_1 = [i.split() for i in x]
    if embedding_case == 1:
        #x_1 = [i.split() for i in x]
        print('text prepared for word2vec...')
        model = Word2Vec(x_list, size = 300, window = 5, min_count=3, workers=3)
        print("saving model...")
        model.save("word2vec_agora")
        del model
    elif embedding_case == 2:
        #x_1 = [i.split() for i in x]
        print('text prepared for FastText...')
        model = FastText(x_list, size = 300, window = 5, min_count = 3, workers=3)
        print("saving model...")
        model.save("FastText_agora")
        del model
    elif embedding_case == 3:
        x = [' '.join(a) for a in x]
        taggedDocs = nt('taggedDocs','words tags')
        docs = []
        for i in range(len(x)):
            words = x[i].split()
            tag = [i]
            docs.append(taggedDocs(words,tag))

        print('Text prepared for doc2vec...')
        model = Doc2Vec(x_list, size = 300, window = 8, min_count = 3, workers = 3)
        print("saving model...")
        model.save('doc2vec_agora_all')
        del model

###########################
# Create Embedding Layer. #
###########################
def create_embedding(model, word_idx):
    num_words = min(UNIQUE_WORDS,len(word_idx))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    #vocab = dict([(k,v.index) for k,v in model.wv.vocab.items()])
    #weights = model.wv.syn0
    not_matched_word = []
    for word,i in word_idx.items():
        if i >= UNIQUE_WORDS:
            continue
        try:
            embedding_vector = model[word]
        except Exception as e:
            embedding_vector = None

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            not_matched_word.append(word)
    #print(embedding_matrix)
    embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            input_length = MAX_SEQUENCE_LENGTH,
                            trainable = True)
    #print(not_matched_word[0:10])
    return embedding_layer

#############################
# Encode text BoW or TF-IDF #
#############################
def encode_x(X, encode_tfidf_bool):
    #maxlen = MAX_SEQUENCE_LENGTH
    if encode_tfidf_bool:
        tokenizer = Tokenizer(num_words = UNIQUE_WORDS)
        tokenizer.fit_on_texts(X)
        x = tokenizer.texts_to_matrix(X)
        #print(mat_seq.shape)
        word_idx = tokenizer.word_index
        #MAX_SEQUENCE_LENGTH = len(word_idx)
        #return x, word_idx
    else:
        tokenizer = Tokenizer(num_words = UNIQUE_WORDS)
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        word_idx = tokenizer.word_index
        x = sequence.pad_sequences(sequences,maxlen = MAX_SEQUENCE_LENGTH)
        print("sequence padding complete..")
        #return x,word_idx
    return x,word_idx

def encode_y(Y):
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)
    y = np_utils.to_categorical(encoded_y)
    return y
#####################################################
# Load data and pre-created feature representation. #
#####################################################
def load_data(path = 'Agora.csv'):
    print("loading data...")
    data = pd.read_csv(path,encoding='latin1')
    return data
def load_feature_rep(path, feature_type):
    model = None
    if feature_type.lower() == 'w2v':
        model = Word2Vec.load(path)
    elif feature_type.lower() == 'ft':
        model = FastText.load(path)
    elif feature_type.lower() == 'd2v':
        model = Doc2Vec.load(path)
    elif feature_type.lower() == 'w2v-google':
        print("loading pretrained word2vec on google news...")
        model = KeyedVectors.load_word2vec_format('./word2vec_models/GoogleNews-vectors-negative300.bin', binary=True)
    elif feature_type.lower() == 'ft-fb':
        print("loading pretrained fasttext on facebook...")
        model = KeyedVectors.load_word2vec_format('./FastText_models/crawl-300d-2M.vec', binary = False)
    return model

#####################
# Create the model. #
#####################
def create_model(embedding_layer, dropout_size, input_dim, model_type):
    model = Sequential()
    #Create LSTM model
    if model_type.lower() == 'lstm':
        if embedding_layer is not None:
            model.add(embedding_layer)
        else:
            model.add(Embedding(input_dim,
                                EMBEDDING_DIM,
                                input_length = input_dim,
                                trainable = True))
        model.add(Conv1D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu'))
        model.add(MaxPooling1D(pool_size = 5))
        model.add(LSTM(128,dropout = 0.2, recurrent_dropout = 0.2))
        if dropout_size > 0.0:
            model.add(Dropout(dropout_size))
        model.add(Dense(NMBR_CATEGORIES,activation = 'softmax'))
    #Create MLP model
    elif model_type.lower() == 'mlp':
        if embedding_layer is not None:
            print("adding embedding layer...")
            model.add(embedding_layer)
            model.add(Flatten())
        model.add(Dense(250, input_dim = input_dim, activation = 'relu'))
        if dropout_size > 0.0:
            model.add(Dropout(dropout_size))
        model.add(Dense(125,activation = 'relu'))
        if dropout_size > 0.0:
            model.add(Dropout(dropout_size))
        model.add(Dense(NMBR_CATEGORIES,activation='softmax'))
    elif model_type.lower() == 'gru':
        if embedding_layer is not None:
            model.add(embedding_layer)
        else:
            model.add(Embedding(input_dim,
                                EMBEDDING_DIM,
                                input_length = input_dim,
                                trainable = True))
        #model.add(Conv1D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu'))
        #model.add(MaxPooling1D(pool_size = 5))
        model.add(GRU(128,dropout = 0.2, recurrent_dropout = 0.2))
        if dropout_size > 0.0:
            model.add(Dropout(dropout_size))
        model.add(Dense(NMBR_CATEGORIES,activation = 'softmax'))
    elif model_type.lower() == 'mult-cnn-lstm':
        feature_w2v_google = load_feature_rep('w2v-google')
        feature_ft_facebook = load_feature_rep('ft-fb')
        embedding_w2v_google = create_embedding(feature_w2v_google)
        embedding_ft_facebook = create_embedding(feature_ft_facebook)
        model1 = Sequential()
        model1.add(embedding_w2v_google)
        model2 = Sequential()
        model2.add(embedding_ft_facebook)
        model = Sequential()
        model.add(Merge([model1,model2],mode='concat',concat_axis=1))
        model.add(Reshape((EMBEDDING_DIM,)))
        model.add(Conv1D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu'))
        model.add(MaxPooling1D(pool_size = 5))
        model.add(LSTM(128,dropout = 0.2, recurrent_dropout = 0.2))
        model.add(Dense(NMBR_CATEGORIES,activation = 'softmax'))
    return model

####################
# Train the model. #
####################
def train_model(model, x, y, epochs,batchsize):
    x_train,x_val,x_test = x
    y_train,y_val,y_test = y
    print('compiling model')
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #Earlystopping
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2,
                                  verbose = 0, mode = 'auto')

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    time1 = datetime.datetime.now()
    history = model.fit(x_train, y_train,
                        validation_data = (x_val, y_val),
                        epochs = epochs,
                        callbacks = [earlyStopping],
                        batch_size = batchsize)
    time2 = datetime.datetime.now()
    print('finished training in: ', (time2-time1).total_seconds()/60, " minutes")
    y_true = [np.argmax(i) for i in y_test]
    preds = model.predict(x_test,batch_size = batchsize)
    preds_true = [np.argmax(i) for i in preds]
    cnf_matrix = confusion_matrix(y_true,preds_true)
    print('finished predictions...')
    plot_scores(history)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['1','3','5'],
                        title='Confusion matrix')
    plt.show()
    #save_model('mlp_model_1.h5')
    return model,history
#x = data.loc[data[' Category'].str.contains("Info")]
#
def kfold_cross_run(dataset_size,encode_tfidf_bool,
            model_type,feature_rep_path,
            feature_rep_type,dropout_size,folds):
    data = load_data()
    print(len(data))
    data = get_equal_for_each_cat(data,dataset_size)
    print(len(data))
    data = data.dropna(subset=[' Item Description'])
    print(len(data))
    print("random sampling...")
    data = data.sample(frac=1).reset_index(drop=True)
    print("splitting data into x and y...")
    X = data[' Item Description']
    Y = data[' Category']
    x = [''.join(i) for i in X]
    print("encode x and y...")
    x, word_idx = encode_x(x,encode_tfidf_bool)
    print('number of unique words: ',len(word_idx))
    y = encode_y(Y)
    print("load feature representation...")
    feature_rep = load_feature_rep(feature_rep_path,feature_rep_type)
    print('creating embedding...')
    embedding_layer = create_embedding(feature_rep,word_idx)
    kfold = StratifiedKFold(n_splits = folds, shuffle = True, random_state = 14)
    cvscores = []
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2,
                                  verbose = 0, mode = 'auto')
    #modelCheckpoint = ModelCheckpoint('./modelCheckPoints/modelCheckpoint_{!s}_{!s}'.format(model_type,feature_rep_type),
    #                                  monitor='val_loss',save_best_only = True)
    print("y shape: ",y.shape)
    msk = np.random.rand(len(x)) < 0.9
    x_test = x[~msk]
    x = x[msk]
    y_test = y[~msk]
    y=y[msk]
    Y_index = Y[msk]

    print("x shape: ",x.shape)
    #print("y.shape: ",[y.index])
    i = 1
    for train,test in kfold.split(x,Y_index):
        print("Training on fold number: ",i)
        i += 1
        model = create_model(embedding_layer, dropout_size, MAX_SEQUENCE_LENGTH, model_type)
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])
        history = model.fit(x[train], y[train],
                  validation_data = (x[test], y[test]),
                  epochs = EPOCHS,
#                  callbacks = [earlyStopping,modelCheckpoint],
                  batch_size = BATCHSIZE)
        scores = model.evaluate(x[test],y[test])
        _scores_ = history.history['val_acc']
        preds = model.predict(x[test])
        y_true = [np.argmax(i) for i in y[test]]
        preds = [np.argmax(i) for i in preds]
        cvscores.append(scores[1] * 100)
        print(confusion_matrix(y_true,preds))
        print("val_acc in function: ",_scores_)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#    model = load_model('./modelCheckPoints/modelCheckpoint')
#    preds = model.predict(x_test)
#    y_true = [np.argmax(i) for i in y_test]
#    preds = [np.argmax(i) for i in preds]
    #cvscores.append(scores[1] * 100)
#    print(confusion_matrix(y_true,preds))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
#####################################
# Plot scores and confusion matrix. #
#####################################
def plot_scores(history):
    # "Acc"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.savefig('./auto_figures/acc_%s_%s_%s')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.savefig('./auto_figures/loss_%s_%s_%s')
    plt.show()
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

############################################
# load feature rep, create and train model #
############################################
def one_run(dataset_size, encode_tfidf_bool,
            model_type,feature_rep_path,
            feature_rep_type,dropout_size):

    data = load_data('yelp_simplified_90K.csv')
    data = get_equal_for_each_cat(data,dataset_size)
    x_train,y_train,x_val,y_val,x_test,y_test,word_idx = split_data(data,encode_tfidf_bool)
    print('loading model...')
    feature_rep = load_feature_rep(feature_rep_path,feature_rep_type)
    print('creating embedding...')
    embedding_layer = create_embedding(feature_rep,word_idx)
    print('create model...')
    model = create_model(embedding_layer,dropout_size,MAX_SEQUENCE_LENGTH,model_type)
    x = [x_train,x_val,x_test]
    y = [y_train,y_val,y_test]
    print("training model...")
    model,history = train_model(model,x,y,EPOCHS,BATCHSIZE)

if __name__ == "__main__":
#    one_run(dataset_size = 30000, encode_tfidf_bool = False,
#            model_type = 'mlp', feature_rep_path = 'None',
#            feature_rep_type = 'ft-fb', dropout_size = 0.4)

    kfold_cross_run(dataset_size = 33000, encode_tfidf_bool = False,
            model_type = 'gru', feature_rep_path = 'None',
            feature_rep_type = 'ft-fb', dropout_size = 0.0,folds = 10)
