# imports
import os
# The following line suppresses a bunch of BS that prints when you import TF.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import nltk
if not os.stat('corpora/'):
    nltk.download("stopwords")
    nltk.download('wordnet')
from nltk.corpus import stopwords
from collections import Counter
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import re
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import logging


'''neural_net_2.
From a YouTube comment on this NLP video: https://www.youtube.com/watch?v=CMrHM8a3hqw
1. Segmentation <<< break data into sentences
2. Tokenizing <<< break sentence into words
3. Stop Words <<< mark down 'Verb to be., prepositions, ...etc...'
4. Stemming <<< same words with different prefix or suffix
5. Lemmatization <<< learning that multiple words can have the same meaning (is, am, are >>> be)
6. Speech Tagging <<< adding tags to words ( noun, verb, preposition)
7. Name Entity Tagging <<< introduce machine to some group of words that may occur in documents
8. Machine Learning (ex. naive bayes calssification) <<< learning the human sentiment and speech

Step 0 would be data import and trimming. It is unlikely that steps 6 and 7 will be implemented at this time. 
This video also does not describe Encoding, which in our case, should probably be implemented just before ML.
Vectorization was not mentioned either.. Not sure if Lists need to be changed to nparrays or something...
look into TFID Vectorizer
'''


ml_logger = logging.getLogger('ML_event_logger')

def log(string):
    ml_logger.debug(msg=string)
    print(string)

lemmatizer= WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Bro I wrote so much code to do this import, I feel like an idiot
df_train = pd.read_csv('ml_model/kaggle/emotions-dataset-for-NLP/train.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('ml_model/kaggle/emotions-dataset-for-NLP/test.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('ml_model/kaggle/emotions-dataset-for-NLP/val.txt', names=['Text', 'Emotion'], sep=';')

dataframe_list = [df_train, df_test, df_val]
master_df = pd.concat(dataframe_list)
# creating a list allows universal dataset operations. 

df_train.name='Training DataFrame'
df_test.name='Testing DataFrame'
df_val.name='Validation DataFrame'
# Naming allows for-loops to print 'dataframe.name: dataframe.<relevant_info>'

for dataframe in dataframe_list:
    print(dataframe.name+':',dataframe.shape)

# These functions prove the import worked and show us some data and its shape.
# In this case, the shape is "how many sentences, and how many columns"
# But the only columns are the sentences and their emotion tags. 

#After Data Cleaning below, the shape is printed again to reflect the changes.
print("***Data Trimming***")
print ("Duplicate removal:")
# Remove duplicates
for dataframe in dataframe_list:
    index = dataframe[dataframe.duplicated() == True].index
    if len(index) > 0:
        print("Removing Duplicates from",dataframe.name+": ")
        print(index.to_list())
        
    dataframe.drop(index, axis = 0, inplace = True)
    dataframe.reset_index(inplace=True, drop = True)

# Remove clashes (same text, different labels)
for dataframe in dataframe_list:
    index = dataframe[dataframe['Text'].duplicated() == True].index
    if len(index) > 0:
        print("Removing Collisions from",dataframe.name+": ")
        print(index.to_list())
        
    dataframe.drop(index, axis = 0, inplace = True)
    dataframe.reset_index(inplace=True, drop = True)

# now is where stopword stuff begins so let's stop and evaluate.
print("Basic Data Trimming Done.")
print("New Sizes:")

for dataframe in dataframe_list:
    print(dataframe.name+':',dataframe.shape)

#DATA ANALYTICS
negating_words = [ "cant", "couldnt", "doesnt", "didnt", "dont", "hasnt", "hadnt", "havent", "isnt", "no", "not", "never" "shouldnt",  "wasnt", "wont", "wouldnt" ]
negative_emotions = ["sadness", "fear", "anger"]
def get_negation_data(sentence_list, label_list):
    negating_sentences = []
    for sentence in sentence_list:
        found_flag = False
        for word in sentence.split(' '):
            for neg_word in negating_words:
                if neg_word == word:
                    found_flag = True
                    index = sentence_list.index(sentence)
                    negating_sentences.append( (sentence, label_list[index]) )
                    continue
            if found_flag:
                continue
                # need a flag to continue a second time
    return negating_sentences

negated_training_data   = get_negation_data(df_train['Text'].to_list(), df_train['Emotion'].to_list())
negated_test_data       = get_negation_data(df_test['Text'].to_list(), df_test['Emotion'].to_list())
negated_validation_data = get_negation_data(df_val['Text'].to_list(), df_val['Emotion'].to_list())
all_negated_data = negated_training_data + negated_test_data + negated_validation_data

def get_emotion_breakdown():
        emotions = master_df['Emotion'].value_counts().keys().tolist()
        counts = master_df['Emotion'].value_counts().tolist()
        return emotions, counts

def get_sorted_wordlist(direction):
    word_counter = Counter()
    master_df['Text'].str.lower().str.split().apply(word_counter.update)
    sorted_list = { word: number for word, number in sorted(word_counter.items(), key=lambda item: item[1], reverse=True)} 
    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    if direction=="desc":
        return sorted_list
    else:
        keys = list(sorted_list.keys())
        values = list(sorted_list.values())
        return dict( zip(keys, values.__reversed__()) )

print("Performing NLP Processing and Normalizing...")
    
def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]
    
    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    
    text = text.split()

    text=[y.lower() for y in text]
    
    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan
            
def normalize_text(df):
    df.Text=df.Text.apply(lambda text : lower_case(text))
    df.Text=df.Text.apply(lambda text : remove_stop_words(text))
    df.Text=df.Text.apply(lambda text : Removing_numbers(text))
    df.Text=df.Text.apply(lambda text : Removing_punctuations(text))
    df.Text=df.Text.apply(lambda text : Removing_urls(text))
    df.Text=df.Text.apply(lambda text : lemmatization(text))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence

df_train= normalize_text(df_train)
df_test= normalize_text(df_test)
df_val= normalize_text(df_val)

#Preprocess text
X_train = df_train['Text']
y_train = df_train['Emotion']

X_test = df_test['Text']
y_test = df_test['Emotion']

X_val = df_val['Text']
y_val = df_val['Emotion']

# These text values are still not encoded for ML use.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)
emotion_list = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
# the to_categorical() function converts each into a CCEF matrix form...
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# Then, the words need to be encoded as well, with each word getting a unique integer.
# Apparently, this is called Tokenizing. 
tokenizer = Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))

# idk if I should even bother with all this fancy shit. 
# neural_net 1 already did all of this, I just wrote the code by hand...

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

maxlen = max([len(t) for t in df_train['Text']])

X_train = pad_sequences(sequences_train, maxlen=229, truncating='pre')
X_test = pad_sequences(sequences_test, maxlen=229, truncating='pre')
X_val = pad_sequences(sequences_val, maxlen=229, truncating='pre')

vocabSize = len(tokenizer.index_word) + 1
print(f"Vocabulary size = {vocabSize}")

TRAIN = False

# TRAINING
if TRAIN:

    #GloVe Embedding
    path_to_glove_file = 'ml_model/kaggle/glove.6B.200d.txt'
    num_tokens = vocabSize
    embedding_dim = 200 #latent factors or features  
    hits = 0
    misses = 0
    embeddings_index = {}

    print("Searching for matches for GloVe Embedding...")
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))

    print("Encoding an embedding matrix...")
    # Assign word vectors to our dictionary/vocabulary
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    print("Ready to train Neural Net...")

    # ML Model Here

    from keras.callbacks import CSVLogger
    csv_logger = CSVLogger('ml.log', append=True)
    adam = Adam(learning_rate=0.005)

    model = Sequential()
    model.add(Embedding(vocabSize, 200, input_length=X_train.shape[1], weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(LSTM(256, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2)))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        verbose=1,
                        batch_size=256,
                        epochs=2,
                        callbacks=[callback]
                    )
    print("Trained. Saving.")
    model.save('neural_net_2')
else:
    print("Loading Saved Neural Net Model...")
    model = keras.models.load_model('neural_net_2')
    print("Loaded.")

def predict(sentence):
    confidence_threshold= 0.27
    print(sentence)
    sentence = normalized_sentence(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
    result_set = model.predict(sentence)
    result = le.inverse_transform(np.argmax(result_set, axis=-1))
    top_prob =  np.max(result_set)
    if top_prob < confidence_threshold:
        print("No confident result.\n\n")
        return None
    else:
        print(f"{result} : {top_prob}\n\n")
    return result_set[0].tolist()
