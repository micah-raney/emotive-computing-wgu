# MY IMPORTS
import os
# The following line suppresses a bunch of BS that prints when you import TF.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import text_to_word_sequence
import pandas as pd # DataFrames, CSV file I/O
import logging
from keras.callbacks import CSVLogger
from collections import Counter
import nltk
#nltk.download("stopwords", download_dir='./')
# above only gets run once; comment it out after the env is setup.
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


csv_logger = CSVLogger('ml.log', append=True)
ml_logger = logging.getLogger('ML_event_logger')

#TODO: Replace log with logger.log()
def log(string):
    ml_logger.debug(msg=string)
    print(string)

''' ML Program steps
1. Import data
    a. Get file paths programmatically
    b. Use readlines() to pull text into variables
2. Encode Data
    a. Word encoding
        i. Build a Dictionary of Unique Words
        ii. Create a numeric Encoding Scheme
        iii. Encode Relevant Data
    b. Emotion encoding
        i. Define emotion ideal labels (from data)
        ii. Define CCEF (Categorical Cross-Entropy Function)
    c. Final arrangement
        a. Define data shape (fill extra space with 0's)
        b. Co-assign X and y (List and Labels)
        c. Assign a lookup/translation function to turn X and y back into text
3. Build and Train Model
    a. Define Sequential Model layers
    b. Set run parameters
    c. Run model.fit()
4. Model is available for use in prediction

TODO: improve prediction accuracy
TODO: make emotion wheel 

'''

# DATA FILE NAMING AND IMPORT
log("***STARTUP***")
log("Loading reference files...")

def load_data(filepath):
    file_handle = open(filepath)
    data = file_handle.readlines()
    file_handle.close()
    return data

def load_raw(filepath):
    file_handle = open(filepath)
    data = file_handle.read()
    file_handle.close()
    return data

def import_fresh_data():
    #Type 1 Import
    file_paths = {}
    for dirname, _, filenames in os.walk('ml_model/kaggle/emotions-dataset-for-NLP/'):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            log(fullpath)
            file_paths[filename] = fullpath

    validation_file_path =  file_paths['val.txt']
    test_file_path =        file_paths['test.txt']
    train_file_path =       file_paths['train.txt']

    validation_data = load_data(validation_file_path)
    test_data =       load_data(test_file_path)
    training_data =   load_data(train_file_path)

    training_list, training_labels      = data_split(training_data, 'txt')
    test_list, test_labels              = data_split(test_data, 'txt')
    validation_list, validation_labels  = data_split(validation_data, 'txt')
    
    return test_list, training_list, validation_list, test_labels, training_labels, validation_labels


# CATEGORIZATION

#sadness (0), joy (1), love (2), anger (3), fear (4), surprise(5). Order defined by dataset 2...

emotion_to_index_dict = {'sadness':0, 'joy':1, 'love':2, 'anger':3,  'fear':4, 'surprise':5}
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] 

emotion_CCEF = [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
]

def emotion_result(emotion):
    emotion_label = emotion_to_index_dict[emotion]
    return emotion_CCEF[emotion_label]
    
def answer_key(emotion_ideal):
    value = emotion_CCEF.index(emotion_ideal)
    return emotion_to_index_dict.key(value)

def convert_index_to_emotion(x):
    return emotion_labels[x]

# Data processing
def data_split(data, data_type):
    ''' 
    Given a plaintext dataset, returns a split list of sentences 
    and mapping of each sentence to its emotion. 
    List: a list of sentences, cleaned up with all non-alphabet chars removed.
    Labels: contains a mapping of each sentence id to each emotion id. '''
    List = []
    Labels = []
    if data_type == 'txt':
        for line in data:
            sanitized_line, emotion = line.strip('\n').split(';')
            List.append(sanitized_line)
            Labels.append(emotion)
    elif data_type == 'csv':
        for line in data:
            sanitized_line, emotion_numeral = line.strip('\n').split(',')
            emotion = convert_index_to_emotion(int(emotion_numeral))
            List.append(sanitized_line)
            Labels.append(emotion)
    return List, Labels

def encode_emotions(labels):
    encoded_emotion_list = [] 
    for emotion in labels:
        encoded_emotion_list.append(emotion_result(emotion))
    return encoded_emotion_list

def process_data_lists(data_lists):
    ''' given a List[] of data_list objects (lists of sentences),
        return a list of all words and the longest sentence length (in # of words), 
        for the ML model to use as input_shape later.'''
    all_sentences = []
    for data_list in all_data_lists:
        for sentence in data_list:
            all_sentences.append(sentence)
    all_words = []
    max_sentence_len = 0 
    for sentence in all_sentences:
        # get longest sentence and populate all_words
        word_array = text_to_word_sequence(sentence)
        if len(word_array) > max_sentence_len:
            max_sentence_len = len(word_array)
        for word in word_array:
            all_words.append(word)
    return all_words, max_sentence_len

def encode_list(plaintext_list, dictionary, shape):
    '''returns converted sentence lists in encoded numbers, or None if any word is not present in the dictionary.'''
    pad='after'
    sentence_id=0
    data_list = plaintext_list
    # make a copy so that plaintext list is not changed
    for sentence in data_list:
        # these are not yet converted to sequences.
        sequence = text_to_word_sequence(sentence)
        sequence = strip_stopwords(sequence)
        new_sequence = []
        fail_flag = False
        if pad=='before':
            for i in range(shape - len(sequence)):
                new_sequence.append(0)
        for word in sequence:
            # assign each word its direct-hash ID
            try:
                word_code = dictionary[word]
            except KeyError:
                fail_flag = True
                ml_logger.warning("User attempted to encode an unsupported word: '%s', which is not present in the lexicon." % word)
                word_code = 0 #temporary to prevent a crash
            if fail_flag:
                continue
            new_sequence.append(word_code)
        if fail_flag:
            return None
        # fill new_sequence up with 0's until it is shape
        if pad=='after':
            for i in range(shape - len(new_sequence)):
                new_sequence.append(0)
        data_list[sentence_id] = new_sequence
        sentence_id+= 1
    return data_list

def strip_stopwords(sentence_list):
    result_set = []
    for word in sentence_list:
        if word not in stop_words:
            result_set.append(word)
    return result_set

# Main

#Version 1

word_lookup = {}
log("running main...")

test_list, training_list, validation_list, test_labels, training_labels, validation_labels = import_fresh_data()
log("pulled fresh text data from source files.")
auxilliary_list = ["hi my name is bob and i wanna kill george clooney"]

all_data_lists = [training_list, test_list, validation_list, auxilliary_list]

all_words, max_sentence_len = process_data_lists(all_data_lists)

word_counter = Counter(all_words)
all_distinct_words = [ k for k, v in sorted(word_counter.items(), key=lambda item: item[1], reverse=True)]
# This line sorts all_distinct_words by most-appearing to least-appearing. 
# The purpose of this is to create a lower digit weight for common words since they are unlikely to have emotional subtext.
# For example, the first few words of this list include "i", "feel", "and", "to", "the"... etc.

log("Loaded.")

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

negated_training_data   = get_negation_data(training_list, training_labels)
negated_test_data       = get_negation_data(test_list, test_labels)
negated_validation_data = get_negation_data(validation_list, validation_labels)
all_negated_data = negated_training_data + negated_test_data + negated_validation_data

i=1 # IMPORTANT: 0 is reserved for blanks. 
for word in all_distinct_words:
    word_lookup[word] = i
    i+=1
# Assigns a unique index to each distinct word, huzzah.
# Calling word_lookup[word] yields its ID.
# max_sentence_len yields the top length of 66, which can be used for array length.
# Finally, create a table of [word, id, numbers... 66n]
# Then, make sure to sort each by its emotion. 

log("created fresh dictionary. Encoding...")

encoded_training_list   = encode_list(training_list, word_lookup, max_sentence_len)
encoded_test_list       = encode_list(test_list, word_lookup, max_sentence_len)
encoded_validation_list = encode_list(validation_list, word_lookup, max_sentence_len)

encoded_training_labels   = encode_emotions(training_labels)
encoded_test_labels       = encode_emotions(test_labels)
encoded_validation_labels = encode_emotions(validation_labels)

# ANALYTICS
log("most words in a line:"+str(max_sentence_len))

# all_words = set(entire_lexicon) # create a Set of each distinct word
log("- There are "+str(len(all_words))+" words in the dataset (before any sanitization).") 
log("- There are "+str(len(all_distinct_words))+" distinct words used in the dataset.\n") 

log("training set size:")
log(len(training_list))
log("encoded training set:")
for i in range(0,5):
    log(str(i)+':'+str(training_list[i])+'\n')
log("answer key:")
for i in range(0,5):
    log(str(i)+':'+str(training_labels[i]))



# https://stackoverflow.com/questions/12282232/how-do-i-count-occurrence-of-unique-values-inside-a-list

# dataframe = pd.DataFrame.from_dict(test_dict)
# all arrays must be of the same length, so use DataFrame after hashing. 
# print(dataframe)

# ML Model

# Keras.Sequential() takes a list of layers as a parameter.
# an emotion is called a class. There are 6 emotion classes in this dataset.
# input_shape defines the standard length of data input (in our case, 66)
# layers.Dense is a standard neuron, pointing to every neuron.
# layers.Dropout is a 'forgetting' neuron. 
from keras import layers

TRAIN = True

result_class_count=6
vocab_size = len(all_distinct_words)
input_count = len(training_list)
shape_count = max_sentence_len
from keras.optimizers import Adam
from keras.layers import *
adam = Adam(learning_rate=0.005)


network='linear'

if network=='lstm':
    neural_net = keras.Sequential()
    neural_net.add(Embedding(vocab_size, 200, input_length=shape_count, trainable=False))
    neural_net.add(Bidirectional(LSTM(256, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
    neural_net.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
    neural_net.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2)))
    neural_net.add(Dense(6, activation='softmax'))

elif network=='linear':
    neural_net = keras.Sequential([
        layers.Dense(500, input_shape=(shape_count,), activation='relu'),
        layers.Dense(500, input_shape=(shape_count,), activation='relu'),
        layers.Dense(500, input_shape=(shape_count,), activation='relu'),
        layers.Dense(result_class_count, activation='sigmoid')
    ])
neural_net.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=adam)
#neural_net.summary()
if TRAIN:
    log("Training model...")
    training_history = neural_net.fit(x=encoded_training_list, y=encoded_training_labels, batch_size=32,
        validation_data=(encoded_validation_list, encoded_validation_labels), epochs=5, verbose=1, callbacks=[csv_logger])
    # research activation functions (relu)
    neural_net.save('neural_net')
else:
    neural_net = keras.models.load_model('neural_net')
