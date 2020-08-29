import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import GRU, Input, Dense, Embedding, Bidirectional, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec, KeyedVectors
from nltk.translate.bleu_score import corpus_bleu

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
zh_max_len = max_len(corpus_zh)+1
en_max_len = max_len(corpus_en)+1
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

#Split the dataset into train and test sets
zh_train, zh_test, en_train, en_test = train_test_split(zh_word_tensor, en_word_tensor, test_size=0.1, random_state=13)

#Preparation of embedding files.
#Source Chinese Embedding: https://github.com/Kyubyong/wordvectors
#Source English Embedding: https://fasttext.cc/docs/en/english-vectors.html
#All embeddings come from word2vec models

embedding_file_en = "/content/gdrive/My Drive/tfm/en.bin"
embedding_file_zh = "/content/gdrive/My Drive/tfm/zh.bin"


def from_w2v_to_dict(embedding_file, lang="zh"):
    if lang == "en":
        model = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    else:
        model = Word2Vec.load(embedding_file)

    vector = model.wv.vectors
    words = model.wv.index2word
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


###########Model for training. GRU units, without Attention#########

#Model seq2seq
nodes = 1024
learning_rate = 0.001
clip_value = 1

# encoder-Chinese
encoder_input = Input(shape=(global_max_len,))
#mask_zero=True allows for the padding 0 at the end of the sequence to be ignored
encoder_embedding = Embedding(input_dim=zh_vocab_size, output_dim=300, mask_zero=True)(encoder_input)
encoder_gru = GRU(nodes, return_state=True, unroll = True, name="encoder_gru")
encoder_output, state_h= encoder_gru(encoder_embedding)



# decoder-English
decoder_input = Input(shape=(global_max_len,), name="decoder_input")
decoder_emb = Embedding(en_vocab_size, 300,mask_zero=True)(decoder_input)


decoder_gru = GRU(nodes, return_sequences=True, unroll= True, return_state=True, name="decoder_gru")

decoder_output, _ = decoder_gru(decoder_emb, initial_state=state_h)

#Decoder Output
decoder_dense_output = Dense(en_vocab_size, activation="softmax")(decoder_output)

model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_dense_output])

# compile model
model.compile(optimizer=RMSprop(learning_rate=learning_rate, clipvalue=clip_value),\
              loss="categorical_crossentropy", metrics=["accuracy"])
# Summarize compiled model
model.summary()
plot_model(model, to_file="/content/gdrive/My Drive/tfm/model_v1.png", show_shapes=True)


# fit model
epochs = 100
#batch_size=256
checkpoint = ModelCheckpoint("/content/gdrive/My Drive/tfm/model_weights_v1.h5", monitor="val_loss",\
                             verbose=1, save_best_only=True , mode="min", save_weights_only=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

history = model.fit(dataset_train, \
          epochs=epochs,
          steps_per_epoch= (len(zh_word_tensor)*0.8)//batch_size,
          validation_data=dataset_test,
          validation_steps=(len(zh_word_tensor)*0.2)//batch_size,
          verbose=1, \
          callbacks=[checkpoint, lr_scheduler, early_stopping], )

model.save("/content/gdrive/My Drive/tfm/model_complete_v1.h5")


##############Model modified for INFERENCE##################

##############Model modified for INFERENCE##################

# encoder_model_inference is the same as in training
encoder_inf = Model(encoder_input, state_h)

encoder_inf.save("/content/gdrive/My Drive/tfm/encoder_inf_model_v1.h5")
encoder_inf.save_weights("/content/gdrive/My Drive/tfm/encoder_inf_weights_v1.h5")

# decoder_model_inference
decoder_inf_state_input_h = Input(shape=(nodes, ), name="encoder_hidden_state") #Encoder hidden states

# decoder_inputs
decoder_inf_input = Input(shape=(1,)) #Input fro inference decoding is 1 word at a time
decoder_inf_input_emb = Embedding(en_vocab_size, 300, \
                                  weights=[emb_en], mask_zero=True)(decoder_inf_input)

# decoder
decoder_inf_gru = GRU(nodes, return_sequences=True, unroll=True, return_state=True)
decoder_inf, h_inf = decoder_gru(decoder_inf_input_emb, initial_state=decoder_inf_state_input_h)
decoder_inf_state = h_inf

decoder_inf_output = Dense(en_vocab_size, activation="softmax")(decoder_inf)


decoder_inf_model = Model([decoder_inf_input, decoder_inf_state_input_h], \
                          [decoder_inf_output, decoder_inf_state])

decoder_inf_model.summary()

plot_model(decoder_inf_model, to_file="/content/gdrive/My Drive/tfm/decoder_inf_model_v1.png", \
           show_shapes=True, show_layer_names=True)

decoder_inf_model.save("/content/gdrive/My Drive/tfm/decoder_inf_model_v1.h5")
decoder_inf_model.save_weights("/content/gdrive/My Drive/tfm/decoder_inf_weights_v1.h5")
plot_model(decoder_inf_model, to_file="/content/gdrive/My Drive/tfm/model_inf_v1.png", show_shapes=True)

# decoder_inf_model.save("/content/gdrive/My Drive/tfm/decoder_inf_model_v3_1.h5")
decoder_inf_model.save_weights("/content/gdrive/My Drive/tfm/decoder_inf_weights_v1.h5")

#############Inference Loop#####################

#Reverse dictionary integer-word

decoding_dict = {v: k for k, v in en_tokenizer_lang.word_index.items()}

decode_zh = {v: k for k, v in zh_tokenizer_lang.word_index.items()}
decode_en = {v: k for k, v in en_tokenizer_lang.word_index.items()}


def translate_sentence(input_sentence, target_tokenizer, decoding_dict, output_max_len):
    states_value = encoder_inf.predict(input_sentence)

    target_sequence = np.zeros((1, 1))
    target_sequence[0, 0] = target_tokenizer.word_index.get("start")
    end_of_sequence = target_tokenizer.word_index.get("end")

    translated_sentence = []

    for i in range(output_max_len):
        output_token, decoder_states = decoder_inf_model.predict([target_sequence,  states_value])
        id_word = np.argmax(output_token[0, -1, :])

        if id_word == end_of_sequence or input_sentence[0][i] == 0:  #this is wrong so far
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
    text = [i for i in my_texts if i != "0"]
    return text


# tokenizer = en_tokenizer_lang
# output_max_len = global_max_len
# decoding_dict = decode_zh
def evaluate_model(source_corpus, target_corpus):
    original = source_corpus  # format: list of sentences as words
    actual = [decode_target(target_corpus[i]) for i in
              range(len(target_corpus))]  #format: list of sentence as tokens
    predicted = translate_corpus(source_corpus, en_tokenizer_lang, \
                                 decoding_dict=decode_en, output_max_len=global_max_len)

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
