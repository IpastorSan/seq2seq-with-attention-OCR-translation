# OCR for Chinese character recognition

# Import Image as string
from full workflow_data_seq2seq_attention.py import tokenizer, translate_corpus, decode_en
from PIL import Image
from pytesseract import image_to_string

# We need the pre-trained files for recognizing both traditional and simplified Chinese
# Then you need to move or copy those files to the """tesseract-ocr/4.00/tessdata""" folder
zh_sim_file = "/content/gdrive/My Drive/tfm/chi_sim_vert.traineddata"
zh_tra_file = "/content/gdrive/My Drive/tfm/chi_tra_vert.traineddata"
shutil.copy(zh_sim_file, "/usr/share/tesseract-ocr/4.00/tessdata")  # This comes from Colab, adapt to your system
shutil.copy(zh_tra_file, "/usr/share/tesseract-ocr/4.00/tessdata")  # This comes from Colab, adapt to your system

"""
Preprocessing form previous script:
1)Eliminate all spaces and special characters
2)Apply tokenizer function 
"""
def from_image_to_text(image):  # Input image is in Chinese
    im = Image.open(image)
    text_im = image_to_string(im, lang='chi_sim_vert+chi_tra_vert')

    text = re.sub(" ", "", text_im)  # Eliminate all spaces to apply our preprocessing
    return text

def create_dataset(data, language="zh"):
    q=preprocessing_zh(texto_imagen)
    lines = q.split("\n")
    dataset= [line.rstrip('。') for line in lines]
    return dataset

def max_len(datos):
    return len([line.split("/t") for line in datos])

tokenizer_lang, word_matrix = tokenizer(data, max_len=max_len_str, max_features=10000)

input_vocab_size = len(tokenizer_lang.word_index)

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

def translate_image(image_data, decoding_dictionary, max_features=1000):

    image_translation = translate_corpus(word_matrix, tokenizer_lang, decoding_dictionary, output_max_len=max_len_str)

    return image_translation

###EXAMPLE WORKFLOW###
"""if __name__ == "__main__":

mock_image = "/content/gdrive/My Drive/mock_image.png"

#-------->Scan image with Tesseract and extract the text as string<-----------

text_from_image = from_image_to_text(imagen_prueba)
data = create_dataset(text_from_image)

#-------->preprocess the string<-----------

max_len_str = max_len(data)
tokenizer_lang, word_matrix = tokenizer(data, max_len=max_len_str, max_features=10000)
input_vocab_size = len(tokenizer_lang.word_index)

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
translation = translate_image(texto_imagen, decoder_inf)"""

