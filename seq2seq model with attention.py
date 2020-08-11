import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import io
from tensorflow.keras import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LSTM, GRU, Input, Dense, TimeDistributed, AdditiveAttention, Embedding, \
    Bidirectional, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import gensim
from gensim.models import Word2Vec, KeyedVectors
from nltk.translate.bleu_score import corpus_bleu

"""##2.1 Preparación de datos

Importar datos, tokenizacion, obtencion de parametros (Longitud maxima de la secuencia, vocabulario maximo de cada idioma)
"""


def open_data(dataset, num_examples=None):
    with open(dataset) as archivo:
        datos = [line.rstrip('\n') for line in archivo]
        corpus = datos[:num_examples]

    return corpus


def tokenizer(dataset, max_len, max_features, language="en"):
    tokenizer_lang = Tokenizer(num_words=max_features, filters="")

    tokenizer_lang.fit_on_texts(dataset)
    # asignamos un numero a cada palabra, es como un diccionario int-string
    # Las palabras mas frecuentes tienen numeros mas bajos, 0 queda reservado a padding

    word_tensor = tokenizer_lang.texts_to_sequences(dataset)
    word_tensor = tf.keras.preprocessing.sequence.pad_sequences(word_tensor, \
                                                                maxlen=max_len, \
                                                                padding="post")
    return tokenizer_lang, word_tensor


# Corpus
num_examples = 20000
corpus_zh = open_data("/content/gdrive/My Drive/tfm/dataset_zh.txt", num_examples=num_examples)
corpus_en = open_data("/content/gdrive/My Drive/tfm/dataset_en.txt", num_examples=num_examples)

print(corpus_zh[0:100])

print(corpus_en[0:100])


def max_len(datos):
    return max(len(line.split()) for line in datos)


# Longitud maxima de secuencia
zh_max_len = max_len(corpus_zh)
en_max_len = max_len(corpus_en)
global_max_len = max(en_max_len, zh_max_len)
print(f"max len zh : {zh_max_len}")
print(f"max len es : {en_max_len}")
print(f"max len global : {global_max_len}")

# Tokenizacion. Obtenemos tokenizador y el texto codificado con integers
zh_tokenizer_lang, zh_word_tensor = tokenizer(corpus_zh, global_max_len, max_features=3000)
en_tokenizer_lang, en_word_tensor = tokenizer(corpus_en, global_max_len, max_features=3000)

print(f"zh_word_tensor shape: {zh_word_tensor.shape}, en_word_tensor shape: {en_word_tensor.shape}")

# Longitud del vocabulario. Se muestran todas pero el modelo solo tiene en cuenta el n-maximo
# que hemos puesto en Tokenizer
zh_vocab_size = len(zh_tokenizer_lang.word_index)
en_vocab_size = len(en_tokenizer_lang.word_index)
print(zh_vocab_size)
print(en_vocab_size)

zh_train, zh_test, en_train, en_test = train_test_split(zh_word_tensor, en_word_tensor, \
                                                        test_size=0.1, random_state=13)

"""##2.2 Preparacion de los archivos de embedding

Fuente de los archivos: https://github.com/Kyubyong/wordvectors
"""

# Archivos embedding
embedding_file_en = "/content/gdrive/My Drive/tfm/en.bin"
embedding_file_zh = "/content/gdrive/My Drive/tfm/zh.bin"


def from_w2v_to_dict(embedding_file, lang="zh"):
    if lang == "en":
        model = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    else:
        model = Word2Vec.load(embedding_file)

    vector = model.wv.vectors
    palabras = model.wv.index2word
    union_embeddings = dict(zip(palabras, vector))
    return union_embeddings


embedding_zh = from_w2v_to_dict(embedding_file_zh)
embedding_en = from_w2v_to_dict(embedding_file_en, lang="en")


def get_embedding_weights(embedd_dict, tokenizer_index, max_features=1000):
    embed = list(embedd_dict.values())

    all_embs = np.stack(embed)
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]  # probar

    word_index = tokenizer_index.word_index
    nb_words = len(word_index)

    embedding_matrix = np.random.normal(emb_mean, emb_std, \
                                        size=(nb_words, embed_size))

    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = np.array(embedd_dict.get(word))
        if embedding_vector.size is not 1:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


emb_zh = get_embedding_weights(embedding_zh, zh_tokenizer_lang, max_features=zh_vocab_size)
emb_en = emb = get_embedding_weights(embedding_en, en_tokenizer_lang, max_features=en_vocab_size)

print(f"emb_zh shape: {emb_zh.shape}, emb_en shape: {emb_en.shape}")

"""##2.3)Modelo para entrenar (con Teacher Forcing)

###2.3.1 Generador de datos para entrenamiento en batches
"""


def generate_batch(X, y, global_max_len, vocab_size_out, batch_size):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, global_max_len), dtype='float32')
            decoder_input_data = np.zeros((batch_size, global_max_len), dtype='float32')
            decoder_target_data = np.zeros((batch_size, global_max_len, vocab_size_out), dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j + batch_size], y[j:j + batch_size])):
                for t, word in enumerate(input_text):
                    encoder_input_data[i, t] = word  # encoder input seq
                for t, word in enumerate(target_text):
                    if t < len(target_text) - 1:
                        decoder_input_data[i, t] = word  # decoder input seq
                    if t > 0:
                        # decoder target sequence (one hot encoded)
                        # does not include the START_ token
                        # Offset by one timestep
                        decoder_target_data[i, t - 1, word] = 1  # si el vector es (1, 2, 4) ese 4 es 0001
            yield ([encoder_input_data, decoder_input_data], decoder_target_data)


# Prueba
b_size = 128
batch = generate_batch(zh_train, en_train, global_max_len, en_vocab_size, b_size)
batch1 = next(batch)
batch2 = next(batch)

print(f"batch encoder_input shape (batch, global_max_len): {batch1[0][0].shape},\n \
batch decoder_input shape (batch, global_max_len): {batch1[0][1].shape}, \n \
batch decoder output shape (batch, global_max_len, output_num_words): {batch1[1].shape}")

print(batch1[0][0][0][0])
print(batch1[0][1][0][1])
print(batch1[1][0][0][11])

# Peso por batch
print(f"{(batch1[1].size / 1024) / 1024} MB con batch size = {b_size}")

"""###2.3.2 Modelo Entrenamiento

####2.3.2.1 Capa de Atencion Bahdanau
"""


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


"""####2.3.2.2 Modelo Seq2seq Encoder-Decoder"""

# training model zh-en
nodes_lstm = 1024
learning_rate = 0.0001
clip_value = 1
dropout = 0.1
# encoder-zh
encoder_input = Input(shape=(global_max_len,))
encoder_embedding = Embedding(zh_vocab_size, 300, input_length=global_max_len,
                              mask_zero=True)(encoder_input)
encoder_gru = Bidirectional(GRU(nodes_lstm, return_sequences=True, \
                                unroll=True, dropout=dropout, return_state=True, \
                                name="encoder_lstm"))
encoder_output, state_h_f, state_h_b = encoder_gru(encoder_embedding)
state_h = Concatenate(name="states_h")([state_h_f, state_h_b])

# Attention Layer
attention_layer = BahdanauAttention(nodes_lstm * 2)
context_vector, attention_weights = attention_layer(state_h, encoder_output)  # output del encoder y decoder
context_vector = tf.keras.layers.RepeatVector(global_max_len)(
    context_vector)  # repeat vector=longitud de secuencia objetivo

# decoder-en
decoder_input = Input(shape=(global_max_len), name="decoder_input")
decoder_emb = Embedding(en_vocab_size, 300, input_length=global_max_len, \
                        mask_zero=True)(decoder_input)

decoder_emb_attention = tf.concat([context_vector, decoder_emb], axis=-1)

decoder_gru = GRU(nodes_lstm * 2, return_sequences=True, return_state=True, \
                  unroll=True, dropout=dropout, name="decoder_lstm")

decoder_output, _ = decoder_gru(decoder_emb_attention, initial_state=state_h)

# Output del decoder
decoder_dense_output = Dense(en_vocab_size, activation="softmax")(decoder_output)

model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_dense_output])

# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, \
                                                 clipvalue=clip_value), \
              loss="categorical_crossentropy")
# Summarize compiled model
model.summary()
plot_model(model, to_file="/content/gdrive/My Drive/tfm/model_2_1.png", show_shapes=True)

"""####2.3.2.3 fit modelo"""

# fit model. hay que poner [encoder input data, decoder input data], target input data=dcoder input data=1 timestep
epochs = 50
b_size = 256
checkpoint = ModelCheckpoint("/content/gdrive/My Drive/tfm/model_weights_v3_1.h5", \
                             monitor="val_loss", verbose=1, save_best_only=True \
                             , mode="min", save_weights_only=True)
model.fit(generate_batch(zh_train, en_train, global_max_len, en_vocab_size, b_size), \
          steps_per_epoch=len(zh_train) // b_size, \
          epochs=epochs, \
          validation_data=generate_batch(zh_test, en_test, global_max_len, en_vocab_size, b_size), \
          validation_steps=len(zh_test) // b_size, \
          verbose=1, \
          callbacks=[checkpoint], )
# model.save("/content/gdrive/My Drive/tfm/model_complete_v3_1.h5")
# RMSprop bien

"""###2.3.3 Modelo para inferencia"""

# encoder_model_inference
encoder_inf = Model(encoder_input, [encoder_output, state_h])

encoder_inf.save("/content/gdrive/My Drive/tfm/encoder_inf_model_v3_1.h5")
encoder_inf.save_weights("/content/gdrive/My Drive/tfm/encoder_inf_weights_v3_1.h5")

# decoder_model_inference 1
decoder_inf_state_input_h = Input(shape=(nodes_lstm * 2,), name="encoder_hidden_state")
encoder_output_input = Input(shape=(global_max_len, nodes_lstm * 2))
# decoder_inputs
decoder_inf_input = Input(shape=(1,))
decoder_inf_input_one = Embedding(en_vocab_size, 300, \
                                  weights=[emb_en], mask_zero=True)(decoder_inf_input)

# Attention Layer
attention_layer = BahdanauAttention(nodes_lstm * 2)
context_vector, attention_weights = attention_layer(decoder_inf_state_input_h,
                                                    encoder_output_input)  # output del encoder y decoder
context_vector = tf.keras.layers.RepeatVector(1)(context_vector)  # repeat vector=longitud de secuencia objetivo

# decoder
decoder_emb_attention = tf.concat([context_vector, decoder_inf_input_one], axis=-1)
decoder_inf_gru = GRU(nodes_lstm * 2, return_sequences=True, return_state=True, unroll=True)
decoder_inf, h_inf = decoder_gru(decoder_emb_attention, initial_state=decoder_inf_state_input_h)
decoder_inf_state = h_inf

decoder_inf_output = Dense(en_vocab_size, activation="softmax")(decoder_inf)

decoder_inf_model = Model([decoder_inf_input, encoder_output_input, decoder_inf_state_input_h], \
                          [decoder_inf_output, decoder_inf_state])
decoder_inf_model.summary()

plot_model(decoder_inf_model, to_file="/content/gdrive/My Drive/tfm/decoder_inf_model_v2.png", \
           show_shapes=True, show_layer_names=True)

# decoder_inf_model.save("/content/gdrive/My Drive/tfm/decoder_inf_model_v3_1.h5")
decoder_inf_model.save_weights("/content/gdrive/My Drive/tfm/decoder_inf_weights_v3_1.h5")

"""###2.3.4 Loop para inferencia"""

# diccionario inverso integer-palabra
decoding_dict = {v: k for k, v in en_tokenizer_lang.word_index.items()}

decode_zh = {v: k for k, v in zh_tokenizer_lang.word_index.items()}
decode_en = {v: k for k, v in en_tokenizer_lang.word_index.items()}


def translate_sentence(input_sentence, target_tokenizer, decoding_dict, output_max_len):
    encoder_output, states_value = encoder_inf.predict(input_sentence)

    target_sequence = np.zeros((1, 1))
    target_sequence[0, 0] = target_tokenizer.word_index.get("start")
    end_of_sequence = target_tokenizer.word_index.get("end")

    translated_sentence = []

    for i in range(output_max_len):
        output_token, decoder_states = decoder_inf_model.predict([target_sequence, \
                                                                  encoder_output, states_value])
        id_word = np.argmax(output_token[0, -1, :])

        if id_word == end_of_sequence or input_sentence[0][i] == 0:  # esta ultima condicion esta mal
            break

        decoded_word = ""

        if id_word > 0:
            decoded_word = decoding_dict[id_word]
            translated_sentence.append(decoded_word)

        target_sequence[0, 0] = id_word
        states_value = decoder_states

    return " ".join(translated_sentence)


# Prueba
sentence = zh_train[0:1]
tr = translate_sentence(sentence, en_tokenizer_lang, decoding_dict=decoding_dict, \
                        output_max_len=global_max_len)
print(tr)


def translate_corpus(corpus, tokenizer, decoding_dict, output_max_len):
    corpus_list = list()
    for i in range(0, len(corpus)):
        line = translate_sentence(corpus[i:i + 1], tokenizer, decoding_dict, output_max_len)
        corpus_list.append(line)
    return corpus_list


"""###2.3.5 Evaluación modelo"""

decode_zh = {v: k for k, v in zh_tokenizer_lang.word_index.items()}
decode_en = {v: k for k, v in en_tokenizer_lang.word_index.items()}

decode_zh.update({0: "0"})
decode_en.update({0: "0"})


def decode_source(sentence):
    sent = np.ndarray.tolist(sentence)
    words = [decode_zh.get(letter) for letter in sent]
    my_texts = (["".join(words[i]) for i in range(len(words))])
    texto = [i for i in my_texts if i != "0"]
    return texto


def decode_target(sentence):
    sent = np.ndarray.tolist(sentence)
    words = [decode_en.get(letter) for letter in sent]
    my_texts = (["".join(words[i]) for i in range(len(words))])
    texto = [i for i in my_texts if i != "0"]
    return texto


# tokenizer = en_tokenizer_lang
# output_max_len = global_max_len
# decoding_dict = decode_zh
def evaluar_modelo(source_corpus, target_corpus):
    original = source_corpus  # formato: lista de frases que son listas de token
    actual = [decode_target(target_corpus[i]) for i in
              range(len(target_corpus))]  # formato: lista de frases que son listas de tokens
    predicted = translate_corpus(source_corpus, en_tokenizer_lang, \
                                 decoding_dict=decode_en, output_max_len=global_max_len)

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
