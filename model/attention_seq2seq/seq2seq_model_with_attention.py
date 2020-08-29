import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import GRU, Input, Dense, Embedding, Bidirectional, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec, KeyedVectors
from nltk.translate.bleu_score import corpus_bleu
import os, datetime

######Data Preparation#######

def open_data(dataset, num_examples=None):
    with open(dataset) as archivo:
        datos = [line.rstrip('\n') for line in archivo]
        corpus = datos[:num_examples]

    return corpus


# Corpus. Adapt the path to your directory
num_examples = 20000
corpus_zh = open_data("/content/gdrive/My Drive/tfm/dataset_zh.txt", num_examples=num_examples)
corpus_en = open_data("/content/gdrive/My Drive/tfm/dataset_en.txt", num_examples=num_examples)


def max_len(data):
    return max(len(line.split()) for line in data)

# Set max len
zh_max_len = max_len(corpus_zh)
en_max_len = max_len(corpus_en)
global_max_len = max(en_max_len, zh_max_len)  #This helps us when training the model, setting all lengths to the maximum


def tokenizer(dataset, max_len, max_features, language="en"):
    tokenizer_lang = Tokenizer(num_words=max_features, filters="")

    tokenizer_lang.fit_on_texts(dataset)

    word_tensor = tokenizer_lang.texts_to_sequences(dataset)
    word_tensor = tf.keras.preprocessing.sequence.pad_sequences(word_tensor,maxlen=max_len, padding="post")

    return tokenizer_lang, word_tensor  #dictionary with tokenized words/sequences translated to numbers


# Tokenizacion. We obtain tokenizer and sequences
zh_tokenizer_lang, zh_word_tensor = tokenizer(corpus_zh, global_max_len, max_features=3000)
en_tokenizer_lang, en_word_tensor = tokenizer(corpus_en, global_max_len, max_features=3000)


# Length of each set of vocabulary
zh_vocab_size = len(zh_tokenizer_lang.word_index)
en_vocab_size = len(en_tokenizer_lang.word_index)


#Preparation of embedding files.
#Source Chinese Embedding: https://github.com/Kyubyong/wordvectors
#Source English Embedding
embedding_file_en = "/content/gdrive/My Drive/tfm/en.bin"
embedding_file_zh = "/content/gdrive/My Drive/tfm/zh.bin"


def from_w2v_to_dict(embedding_file, lang="zh"):
    if lang == "en":
        model = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    else:
        model = Word2Vec.load(embedding_file)

    vector = model.wv.vectors
    words= model.wv.index2word
    union_embeddings = dict(zip(words, vector))
    return union_embeddings


embedding_zh = from_w2v_to_dict(embedding_file_zh)
embedding_en = from_w2v_to_dict(embedding_file_en, lang="en")

#Extract embedding weights to input into Keras Embedding Layer. Words not found in the embedding
#will be substitued with a random vector with normal distribution with mean=embedding mean and std=embedding std


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

########Batch Generator for training. We also implement teacher forcing here######

def generate_batch():
    while True:
        for j in range(0, len(zh_word_tensor)):
            encoder_input_data = np.zeros((global_max_len), dtype='float32')
            decoder_input_data = np.zeros((global_max_len), dtype='float32')
            decoder_target_data = np.zeros((global_max_len, en_vocab_size), dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(zh_word_tensor[j:j+1],en_word_tensor[j:j+1])):
                for t, word in enumerate(input_text):
                  encoder_input_data[t] = word  # encoder input seq
                for t, word in enumerate(target_text):
                    if t < len(target_text) - 1:
                       decoder_input_data[t] = word  # decoder input seq
                    if t > 0:
                        # decoder target sequence (one hot encoded)
                        # does not include the "start" token and is offset by one timestep
                        decoder_target_data[t - 1, word] = 1  # si el vector es (1, 2, 4) ese 4 es 0001
            yield ((encoder_input_data, decoder_input_data), decoder_target_data)

types = ((tf.float32,\
         tf.float32),\
         tf.float32)

shapes=((tf.TensorShape([global_max_len]),tf.TensorShape([global_max_len])),
        tf.TensorShape([None,en_vocab_size,]))

dataset = tf.data.Dataset.from_generator(generate_batch, types, shapes)

batch_size=256

train=int(len(en_word_tensor)*0.8)
test=int(len(en_word_tensor)*0.2)
dataset=dataset.shuffle(1000, reshuffle_each_iteration=False)
dataset_train = dataset.skip(test).batch(batch_size, drop_remainder=True)
dataset_test = dataset.take(test).batch(batch_size, drop_remainder=True)

###########Model for training. GRU units, Bidirectional, with Attention (Bahdanau)#########

#Bahdanau Attention (from Tensorflow Tutorial)

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

#Model seq2seq

nodes = 1024
learning_rate = 0.001
clip_value = 1
dropout = 0.1

# encoder-Chinese
encoder_input = Input(shape=(global_max_len,))
#mask_zero=True allows for the padding 0 at the end of the sequence to be ignored
encoder_embedding = Embedding(zh_vocab_size, 300, input_length=global_max_len, mask_zero=True)(encoder_input)

encoder_gru = Bidirectional(GRU(nodes, return_sequences=True,unroll=True,\
                                name="encoder_gru_1"))(encoder_embedding)

batch_norm = tf.keras.layers.BatchNormalization()(encoder_gru)

encoder_output, state_h_f, state_h_b = Bidirectional(GRU(nodes, return_sequences=True,unroll=True,\
                                return_state=True,name="encoder_gru_2"))(batch_norm)

state_h = Concatenate(name="states_h")([state_h_f, state_h_b])

# Attention Layer
attention_layer = BahdanauAttention(nodes * 2)
context_vector, attention_weights = attention_layer(state_h, encoder_output)  # output del encoder y decoder
context_vector = tf.keras.layers.RepeatVector(global_max_len)(
    context_vector)  # repeat vector=length of target sequence

# decoder-English
decoder_input = Input(shape=(global_max_len), name="decoder_input")

decoder_emb = Embedding(en_vocab_size, 300, input_length=global_max_len,mask_zero=True)(decoder_input)

decoder_emb_attention = tf.concat([context_vector, decoder_emb], axis=-1)

decoder_gru = GRU(nodes * 2, return_sequences=True,unroll=True,\
                  dropout=dropout,name="decoder_gru_1")(decoder_emb_attention, initial_state=state_h)

decoder_batch_norm = tf.keras.layers.BatchNormalization()(decoder_gru)

decoder_output, _ = decoder_gru = GRU(nodes* 2, return_sequences=True, return_state=True,unroll=True,\
                  dropout=dropout,name="decoder_gru_2")(decoder_batch_norm)


#Decoder Output
decoder_dense_output = Dense(en_vocab_size, activation="softmax")(decoder_output)

model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_dense_output])

# compile model
model.compile(optimizer=RMSprop(learning_rate=learning_rate, clipvalue=clip_value),\
              loss="categorical_crossentropy", metrics=["accuracy"])
# Summarize compiled model
model.summary()
plot_model(model, to_file="/content/gdrive/My Drive/tfm/model_3.png", show_shapes=True)


# fit model
epochs = 100
#batch_size=256
checkpoint = ModelCheckpoint("/content/gdrive/My Drive/tfm/model_weights_v3.h5", monitor="val_loss",\
                             verbose=1, save_best_only=True , mode="min", save_weights_only=True)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


history = model.fit(dataset_train, \
          epochs=epochs,
          steps_per_epoch= (len(zh_word_tensor)*0.8)//batch_size,
          validation_data=dataset_test,
          validation_steps=(len(zh_word_tensor)*0.2)//batch_size,
          verbose=1, \
          callbacks=[checkpoint, lr_scheduler, early_stopping, tensorboard_callback], )

##############Model modified for INFERENCE##################

# encoder_model_inference is the same
encoder_inf = Model(encoder_input, [encoder_output, state_h])

encoder_inf.save("/content/gdrive/My Drive/tfm/encoder_inf_model_v3_1.h5")
encoder_inf.save_weights("/content/gdrive/My Drive/tfm/encoder_inf_weights_v3_1.h5")

# decoder_model_inference 1
decoder_inf_state_input_h = Input(shape=(nodes * 2,), name="encoder_hidden_state") #Encoder hidden states
encoder_output_input = Input(shape=(global_max_len, nodes * 2)) #encoder output for attention layer
# decoder_inputs
decoder_inf_input = Input(shape=(1,)) #Input fro inference decoding is 1 word at a time
decoder_inf_input_one = Embedding(en_vocab_size, 300, \
                                  weights=[emb_en], mask_zero=True)(decoder_inf_input)

# Attention Layer
attention_layer = BahdanauAttention(nodes * 2)
context_vector, attention_weights = attention_layer(decoder_inf_state_input_h,
                                                    encoder_output_input)
context_vector = tf.keras.layers.RepeatVector(1)(context_vector)  # repeat vector=longitud de secuencia objetivo

# decoder
decoder_emb_attention = tf.concat([context_vector, decoder_inf_input_one], axis=-1)
decoder_inf_gru = GRU(nodes * 2, return_sequences=True, return_state=True, unroll=True)
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

#############Inference Loop#####################

#Reverse dictionary integer-word

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


def translate_corpus(corpus, tokenizer, decoding_dict, output_max_len):
    corpus_list = list()
    for i in range(0, len(corpus)):
        line = translate_sentence(corpus[i:i + 1], tokenizer, decoding_dict, output_max_len)
        corpus_list.append(line)
    return corpus_list


#######Model Evaluation######3

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
def evaluate_model(source_corpus, target_corpus):
    original = source_corpus  # format: list of sentences
    actual = [decode_target(target_corpus[i]) for i in
              range(len(target_corpus))]  # format: list of sentences
    predicted = translate_corpus(source_corpus, en_tokenizer_lang, \
                                 decoding_dict=decode_en, output_max_len=global_max_len)

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
