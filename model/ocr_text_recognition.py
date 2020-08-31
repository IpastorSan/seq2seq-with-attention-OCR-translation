# OCR for Chinese character recognition
import re
import unicodedata
import io
import jieba
from hanziconv import HanziConv
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from vanilla_seq2seq_model.py import tokenizer, translate_sentence, translate_corpus, open_data
from text_preprocessing.py import preprocessing_zh, unicode_to_ascii
from PIL import Image
from pytesseract import image_to_string

# We need the pre-trained files for recognizing both traditional and simplified Chinese
# Then you need to move or copy those files to the """tesseract-ocr/4.00/tessdata""" folder
zh_sim_file = "/content/gdrive/My Drive/tfm/chi_sim_vert.traineddata"
zh_tra_file = "/content/gdrive/My Drive/tfm/chi_tra_vert.traineddata"
shutil.copy(zh_sim_file, "/usr/share/tesseract-ocr/4.00/tessdata")  # This comes from Colab, adapt to your system
shutil.copy(zh_tra_file, "/usr/share/tesseract-ocr/4.00/tessdata")  # This comes from Colab, adapt to your system

"""
Preprocessing from previous script:
1)Eliminate all spaces and special characters
2)Recover tokenizer from training and apply it to new sequence
3)Recover decoding dictionary from training for decoding
"""
#recovering dataset from training
def open_data(dataset, num_examples=None):
    with open(dataset) as archivo:
        datos = [line.rstrip('\n') for line in archivo]
        corpus = datos[:num_examples]
        
    return corpus

# Training corpus, necessary for reproducing results
num_examples = 20000
corpus_zh = open_data("/content/gdrive/My Drive/tfm/dataset_zh.txt", num_examples=num_examples)
corpus_en = open_data("/content/gdrive/My Drive/tfm/dataset_en.txt", num_examples=num_examples)


def from_image_to_text(image):  # Input image is in Chinese
    im = Image.open(image)
    text_im = image_to_string(im, lang='chi_sim_vert+chi_tra_vert')

    text = re.sub(" ", "", text_im)  # Eliminate all spaces to apply our preprocessing
    return text

def create_dataset(data, language="zh"):
    q=preprocessing_zh(texto_imagen)
    lines = q.split("\n")
    dataset= [line.strip('。') for line in lines]
    return dataset

def max_len(datos):
    return len([line.split("/t") for line in str(datos)])


#tokenizing source with previously trained Tokenizer, using it to convert the new text to a sequence so it can be compatible with the trained model

max_len_str = max_len(data)

tokenizer_source = tf.keras.preprocessing.text.Tokenizer()
tokenizer_source.fit_on_texts(corpus_zh)
word_matrix = tokenizer_source.texts_to_sequences(data)
input_vocab_size = len(tokenizer_lang.word_index)

#decoding dictionary, necessary for decoding on inference
target_tokenizer = tf.keras.preprocessing.text.Tokenizer()
target_tokenizer.fit_on_texts(corpus_en)
decode_dictionary_target = {v:k for k, v in target_tokenizer.word_index.items()}


#####Recovering of previously trained models#####
"""As an example, we will recover the vanilla model.
All hardcoded numbers (GRU nodes, input dim of embedding layer, Dense layer output dimension) are specific to our models,
you can adapt it to the model that you trained, no magic numbers involved"""

# encoder-Chinese
encoder_input = Input(shape=(max_len_str,))
#mask_zero=True allows for the padding 0 at the end of the sequence to be ignored
encoder_embedding = Embedding(input_dim=10562, output_dim=300,\
                               mask_zero=True)(encoder_input) #we do not use pretrained weights in the embedding to avoid too much boilerplate
encoder_gru = GRU(1024, return_state=True, unroll = True, name="encoder_gru")
encoder_output, state_h= encoder_gru(encoder_embedding)

encoder_inf = Model(encoder_input, state_h)

encoder_inf.load_weights("/content/gdrive/My Drive/tfm/encoder_inf_weights_v4.h5") #adapt filepath


# decoder_model_inference - English
decoder_inf_state_input_h = Input(shape=(1024, ), name="encoder_hidden_state") #Encoder hidden states

# decoder_inputs
decoder_inf_input = Input(shape=(1,)) #Input fro inference decoding is 1 word at a time
decoder_inf_input_emb = Embedding(5381, 300, \
                                  mask_zero=True)(decoder_inf_input)#we do not use pretrained weights in the embedding to avoid too much boilerplate

# decoder
decoder_inf_gru = GRU(1024, return_sequences=True, unroll=True, return_state=True)
decoder_inf, h_inf = decoder_inf_gru(decoder_inf_input_emb, initial_state=decoder_inf_state_input_h)
decoder_inf_state = h_inf

decoder_inf_output = Dense(5381, activation="softmax")(decoder_inf)


decoder_inf = Model([decoder_inf_input, decoder_inf_state_input_h], \
                          [decoder_inf_output, decoder_inf_state])

decoder_inf.load_weights("/content/gdrive/My Drive/tfm/decoder_inf_weights_v4.h5")

# Translate image content

def translate_image(decoding_dictionary, max_features=1000):
    image_translation = translate_sentence(word_matrix, target_tokenizer,\
                        decoding_dictionary, output_max_len=max_len_str )

    return image_translation


###EXAMPLE WORKFLOW###
"""if __name__ == "__main__":

mock_image = "/content/gdrive/My Drive/mock_image.png"

#-------->Scan image with Tesseract and extract the text as string<-----------

text_from_image = from_image_to_text(imagen_prueba)
data = create_dataset(text_from_image)

--------->recover training data<---------

def open_data(dataset, num_examples=None):
    with open(dataset) as archivo:
        datos = [line.rstrip('\n') for line in archivo]
        corpus = datos[:num_examples]
        
    return corpus

# Training corpus, necessary for reproducing results
num_examples = 20000
corpus_zh = open_data("/content/gdrive/My Drive/tfm/dataset_zh.txt", num_examples=num_examples)
corpus_en = open_data("/content/gdrive/My Drive/tfm/dataset_en.txt", num_examples=num_examples)

#-------->preprocess the string<-----------
#tokenizing source with previously trained Tokenizer, using it to convert the new text to a sequence so it can be compatible with the trained model

max_len_str = max_len(data)

tokenizer_source = tf.keras.preprocessing.text.Tokenizer()
tokenizer_source.fit_on_texts(corpus_zh)
word_matrix = tokenizer_source.texts_to_sequences(data)
input_vocab_size = len(tokenizer_lang.word_index)

#decoding dictionary, necessary for decoding on inference
target_tokenizer = tf.keras.preprocessing.text.Tokenizer()
target_tokenizer.fit_on_texts(corpus_en)
decode_dictionary_target = {v:k for k, v in target_tokenizer.word_index.items()}

#-------->Load models<-----------

#ENCODER-Chinese
encoder_input = Input(shape=(max_len_str,))
#mask_zero=True allows for the padding 0 at the end of the sequence to be ignored
encoder_embedding = Embedding(input_dim=10562, output_dim=300,\
                               mask_zero=True)(encoder_input) #we do not use pretrained weights in the embedding to avoid too much boilerplate
encoder_gru = GRU(1024, return_state=True, unroll = True, name="encoder_gru")
encoder_output, state_h= encoder_gru(encoder_embedding)

encoder_inf = Model(encoder_input, state_h)

encoder_inf.load_weights("/content/gdrive/My Drive/tfm/encoder_inf_weights_v4.h5") #adapt filepath


#DECODER - English
decoder_inf_state_input_h = Input(shape=(1024, ), name="encoder_hidden_state") #Encoder hidden states

# decoder_inputs
decoder_inf_input = Input(shape=(1,)) #Input fro inference decoding is 1 word at a time
decoder_inf_input_emb = Embedding(5381, 300, \
                                  mask_zero=True)(decoder_inf_input)#we do not use pretrained weights in the embedding to avoid too much boilerplate

# decoder
decoder_inf_gru = GRU(1024, return_sequences=True, unroll=True, return_state=True)
decoder_inf, h_inf = decoder_inf_gru(decoder_inf_input_emb, initial_state=decoder_inf_state_input_h)
decoder_inf_state = h_inf

decoder_inf_output = Dense(5381, activation="softmax")(decoder_inf)


decoder_inf = Model([decoder_inf_input, decoder_inf_state_input_h], \
                          [decoder_inf_output, decoder_inf_state])

decoder_inf.load_weights("/content/gdrive/My Drive/tfm/decoder_inf_weights_v4.h5")

#-------->translate sequence<-----------
translation = translate_image(decode_dictionary_target)"""

