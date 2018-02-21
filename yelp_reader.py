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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers import Dense, Embedding, Flatten, Dropout, Conv1D, LSTM, MaxPooling1D
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD

#Write save fig function. skip plots. for overnight runs. could be comp spec.
#add dropout. or something. horrible results on 3
#maybe try other Loss function (shouldn't matter)
#add Callback.EarlyStop. For overnight runs.
#on Overnight run, increase epochs, obv.
#enable multi run of different Parameters. for other comps.
#check for another feature sel technique. now TF-IDF, Word2Vec, Doc2Vec, FastText.
#Maybe SVM? TF-IDF + SVM?
win_unicode_console.enable()
np.random.seed(7)
UNIQUE_WORDS = 54616
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 1023
NMBR_CATEGORIES = 3
EPOCHS = 4
BATCHSIZE = 128

def get_equal_for_each_cat(data_set,max_length):
    data_set = data_set.loc[data_set['stars'].isin([5,3,1])]
    data_set_1 = data_set.loc[data_set['stars'] == 1]
    data_set_3 = data_set.loc[data_set['stars'] == 3]
    data_set_5 = data_set.loc[data_set['stars'] == 5]
    #min_length = min([len(data_set_1),len(data_set_3),len(data_set_5)])
    dataset_3 = data_set_3[:max_length]
    dataset_1 = data_set_1[:max_length]
    dataset_5 = data_set_5[:max_length]
    data_frames = [dataset_3,dataset_1,dataset_5]
    dataset = pd.concat(data_frames)
    print('dataset split complete...')
    return dataset
def clean_text_for_word2vec(data_set):
    dataset = get_equal_for_each_cat(data_set,20000)
    x = [k for k in dataset['text']]
    x_1 = [i.split() for i in x]
    print('text prepared for word2vec...')
    create_word2vec(x_1)
def clean_text_for_doc2vec(data_set):
    taggedDocs = nt('taggedDocs','words tags')
    dataset = get_equal_for_each_cat(data_set,30000)
    x = [k for k in dataset['text']]
    docs = []
    for i in range(len(x)):
        words = x[i].split()
        tag = [i]
        docs.append(taggedDocs(words,tag))
    create_doc2vec(docs)
def clean_text_for_fasttext(data_set):
    dataset = get_equal_for_each_cat(data_set,20000)
    x = [k for k in dataset['text']]
    x_1 = [i.split() for i in x]
    create_fasttext(x_1)
def load_word2vec_google():
    print('loading Word2Vec pretrained...')
    model = KeyedVectors.load_word2vec_format('./word2vec_models/GoogleNews-vectors-negative300.bin', binary=True)
    return model
def load_FastText_Facebook():
    print('loading FastText pretrained...')
    model = KeyedVectors.load_word2vec_format('./FastText_models/crawl-300d-2M.vec', binary = False)
    return model
def save_model(model,path):
    model.save(path)
    del model
def load_model(path):
    model = load_model(path)
    return model
def create_word2vec(x_list):
    print('creating word2vec...')
    model = Word2Vec(x_list, size = 300, window = 5, min_count=3, workers=3)
    model.save("word2vec_yelp")
    weights = model.wv.syn0
    np.save(open("word2vec_yelp_weights",'wb'),weights)
def create_doc2vec(x_list):
    print('creating doc2vec...')
    #documents = Doc2Vec.TaggedLineDocument()
    model = Doc2Vec(x_list, size = 300, window = 8, min_count = 5, workers = 4)
    model.save('doc2vec_90k_yelp')
def create_fasttext(x_list):
    print("creating FastText model...")
    model = FastText(x_list, size = 300, window = 5, min_count = 3, workers=3)
    print("saving model...")
    model.save("FastText_yelp")
    del model
def load_word2vec_yelp(path):
    print('loading w2v yelp...')
    w2v_model = Word2Vec.load(path)
    return w2v_model
def load_FastText_yelp(path):
    print('loading FastText yelp...')
    ft_model = FastText.load(path)
    return ft_model
def load_doc2vec(path):
    print("loading doc2vec...")
    d2v_model = Doc2Vec.load(path)
    return d2v_model
def _initial_creation():
    print("Loading dataset")
    data_set = pd.read_csv("yelp_review.csv",usecols = [3,5])
    print(data_set.head())
    data_set.dropna()
    #print(data_set.value_counts())
    #clean_text_for_word2vec(data_set)
    print("splitting dataset...")
    data = get_equal_for_each_cat(data_set,30000)
    print("cleaning data...")
    #data['text'] = data['text'].str.replace('I\'m','i am')
    #data['text'] = data['text'].str.replace('i\'m','i am')
    #data['text'] = data['text'].str.replace('I\'ve','i have')
    #data['text'] = data['text'].str.replace('i\'ve','i have')
    #data['text'] = data['text'].str.replace('5/5','STRONGPOSITIVE')
    #data['text'] = data['text'].str.replace('4/5','POSITIVE')
    #data['text'] = data['text'].str.replace('3/5','AMBIVALENT')
    #data['text'] = data['text'].str.replace('2/5','NEGATIVE')
    #data['text'] = data['text'].str.replace('1/5','STRONGNEGATIVE')


    #data['text'] = data['text'].str.replace(r'\W+',' ')
    data['text'] = data['text'].str.lower()
    print("random sampling...")
    data = data.sample(frac=1).reset_index(drop=True)
    #print(data_set['text'].value_counts())
    print("storing to csv...")
    data.to_csv('yelp_simplified_90K.csv')
    return data
def load_data(path = 'yelp_simplified_90K.csv'):
    data = pd.read_csv(path)
    return data
def train_model(model,x,y,epochs,batchsize):
    x_train,x_val,x_test = x
    y_train,y_val,y_test = y
    print('compiling model')
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'nadam',
                  metrics = ['accuracy'])
    time1 = datetime.datetime.now()
    history = model.fit(x_train,y_train,
                        validation_data = (x_val,y_val),
                        epochs = epochs,
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
    return history
def create_model_MLP(embedding_layer,dropout_bool,input_dim):
    model = Sequential()
    if embedding_layer is not None:
        model.add(embedding_layer)
        model.add(Flatten())
    model.add(Dense(250, input_dim = input_dim, activation = 'relu'))
    if dropout_bool:
        model.add(Dropout(0.4))
    model.add(Dense(125,activation = 'relu'))
    if dropout_bool:
        model.add(Dropout(0.4))
    model.add(Dense(NMBR_CATEGORIES,activation='softmax'))
    return model
def create_model_LSTM(embedding_layer,dropout_bool,input_dim):
    model = Sequential()
    if embedding_layer is not None:
        model.add(embedding_layer)
    else:
        model.add(Embedding(input_dim,
                                EMBEDDING_DIM,
                                input_length = input_dim,
                                trainable = True))
    model.add(Conv1D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_size = 3))
    model.add(LSTM(128,dropout = 0.2, recurrent_dropout = 0.2))
    if dropout_bool:
        model.add(Dropout(0.5))
    model.add(Dense(NMBR_CATEGORIES,activation = 'softmax'))
    return model
def word2vec_embedding(model,word_idx):
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
def fasttext_embedding(model,word_idx):
    num_words = min(UNIQUE_WORDS,len(word_idx))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    #vocab = dict([(k,v.index) for k,v in model.vocab.items()])
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
def count_longest_sentence(x):
    sentences = [a.split() for a in x]
    lengths = []
    for i in sentences:
        lengths.append(len(i))
    max_length = max(lengths)
    return max_length
def tokenizer_encoder(X):
    tokenizer = Tokenizer(num_words = UNIQUE_WORDS)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    word_idx = tokenizer.word_index
    x = sequence.pad_sequences(sequences,maxlen = MAX_SEQUENCE_LENGTH)
    print("sequence padding complete..")
    return x,word_idx
def tokenizer_encoder_tfidf(X):
    tokenizer = Tokenizer(num_words = UNIQUE_WORDS)
    tokenizer.fit_on_texts(X)
    mat_seq = tokenizer.texts_to_matrix(X)
    print(mat_seq.shape)
    word_idx = tokenizer.word_index
    MAX_SEQUENCE_LENGTH = len(word_idx)
    return mat_seq, word_idx
def split_data(data,encodes):
    print("random sampling...")
    data = data.sample(frac=1).reset_index(drop=True)
    print("splitting data into x and y...")
    X = data['text']
    Y = data['stars']
    X = X.tolist()
    #x = [a.split() for a in X]
    x = [''.join(a) for a in X]
    max_len = count_longest_sentence(x)
    print('longest sentence: ',max_len)

    if encodes == 1:
        x, word_idx = tokenizer_encoder(x)
    else:
        x, word_idx = tokenizer_encoder_tfidf(x)

    print('number of unique words: ', len(word_idx))
    #UNIQUE_WORDS = len(word_idx)

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)
    y = np_utils.to_categorical(encoded_y)
    NMBR_CATEGORIES = len(y[0])
    print("label encoding complete...")
    train_split = int(0.7*len(data))
    val_split = int(0.2*len(data))
    x_train = x[:train_split]
    y_train = y[:train_split]
    cut_point = train_split + val_split
    x_val = x[train_split:cut_point]
    y_val = y[train_split:cut_point]
    x_test = x[cut_point:]
    y_test = y[cut_point:]
    print("splitting data finished.")
    return x_train,y_train,x_val,y_val,x_test,y_test,word_idx
def create_feature_representation():
    print("loading data...")
    data = load_data()
    clean_text_for_doc2vec(data)

def one_run_w2v_google():
    #data  = _initial_creation()
    data = load_data()
    x_train,y_train,x_val,y_val,x_test,y_test,word_idx = split_data(data,1)
    print('loading w2v google...')
    w2v_model = load_word2vec_google()
    print('creating embedding...')
    embedding_layer = word2vec_embedding(w2v_model,word_idx)
    print('creating model...')
    #mlp_model = create_model_MLP(embedding_layer, True, MAX_SEQUENCE_LENGTH)
    lstm_model = create_model_LSTM(embedding_layer, False, MAX_SEQUENCE_LENGTH)
    x = [x_train,x_val,x_test]
    y = [y_train, y_val, y_test]
    print('training model...')
    #history = train_model(mlp_model,x,y,EPOCHS, BATCHSIZE)
    history = train_model(lstm_model,x,y,EPOCHS, BATCHSIZE)
def one_run_w2v_yelp():
    #data  = _initial_creation()
    data = load_data()
    x_train,y_train,x_val,y_val,x_test,y_test,word_idx = split_data(data,1)
    print('loading w2v yelp...')
    w2v_model = load_word2vec_yelp('word2vec_yelp')
    print('creating embedding...')
    embedding_layer = word2vec_embedding(w2v_model,word_idx)
    print('creating model...')
    #mlp_model = create_model_MLP(embedding_layer, True, MAX_SEQUENCE_LENGTH)
    lstm_model = create_model_LSTM(embedding_layer, True, MAX_SEQUENCE_LENGTH)
    x = [x_train,x_val,x_test]
    y = [y_train, y_val, y_test]
    print('training model...')
    #history = train_model(mlp_model,x,y,EPOCHS, BATCHSIZE)
    history = train_model(lstm_model,x,y,EPOCHS, BATCHSIZE)
def one_run_tfidf():
    data = load_data()
    x_train,y_train,x_val,y_val,x_test,y_test,word_idx = split_data(data,2)
    #mlp_model = create_model_MLP(None,True,len(word_idx))
    #lstm_model = create_model_LSTM(None,True,len(word_idx))

    x = [x_train,x_val,x_test]
    y = [y_train, y_val, y_test]
    print('training model...')
    #history = train_model(mlp_model,x,y,EPOCHS, BATCHSIZE)
    #history = train_model(lstm_model,x,y,EPOCHS, BATCHSIZE)
def one_run_ft_yelp():
    #data  = _initial_creation()
    data = load_data()
    #clean_text_for_fasttext(data)
    x_train,y_train,x_val,y_val,x_test,y_test,word_idx = split_data(data,1)
    #ft_model = load_FastText_yelp('FastText_yelp')
    ft_model = load_FastText_Facebook()
    print('creating embedding...')
    embedding_layer = fasttext_embedding(ft_model,word_idx)
    print('creating model...')
    #mlp_model = create_model_MLP(embedding_layer, True, MAX_SEQUENCE_LENGTH)
    lstm_model = create_model_LSTM(embedding_layer, True, MAX_SEQUENCE_LENGTH)
    x = [x_train,x_val,x_test]
    y = [y_train, y_val, y_test]
    print('training model...')
    #history = train_model(mlp_model,x,y,EPOCHS, BATCHSIZE)
    history = train_model(lstm_model,x,y,EPOCHS, BATCHSIZE)
def one_run_d2v_yelp():
    data = load_data()
    #clean_text_for_doc2vec(data)
    data = get_equal_for_each_cat(data,10000)
    x_train,y_train,x_val,y_val,x_test,y_test,word_idx = split_data(data,1)
    print('loading w2v yelp...')
    d2v_model = load_doc2vec('doc2vec_90k_yelp')
    print('creating embedding...')
    embedding_layer = word2vec_embedding(d2v_model,word_idx)
    print('creating model...')
    mlp_model = create_model_MLP(embedding_layer, False, MAX_SEQUENCE_LENGTH)
    #lstm_model = create_model_LSTM(embedding_layer, True, MAX_SEQUENCE_LENGTH)
    x = [x_train,x_val,x_test]
    y = [y_train, y_val, y_test]
    print('training model...')
    history = train_model(mlp_model,x,y,EPOCHS, BATCHSIZE)
    #history = train_model(lstm_model,x,y,EPOCHS, BATCHSIZE)
def testing_function():
    data = load_data()
    print(data.head())
    print(len(data))
    data = get_equal_for_each_cat(data,10000)
    print(data.head())
    print(len(data[data['stars'] == 5]))
    print(len(data[data['stars'] == 3]))
    print(len(data[data['stars'] == 1]))
    x_train,y_train,x_val,y_val,x_test,y_test,word_idx = split_data(data,1)
    print(len(word_idx))
    print(y_val.shape)
    print(len(y_val))
    print(y_val[:30])

if __name__ == "__main__":
    #_initial_creation()
    #one_run_w2v_google()
    #create_feature_representation()
    #one_run_w2v_yelp()
    #one_run_tfidf()
    #one_run_ft_yelp()
    one_run_d2v_yelp()
    #testing_function()
