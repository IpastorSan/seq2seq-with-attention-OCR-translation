# OCR for Chinese character recognition

# Import Image as string
from full workflow_data_seq2seq_attention.py import max_len, tokenizer, translate_corpus, decode_en
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


# Translate image content

def translate_image(image_data, decoding_dictionary, max_features=1000):
    max_len_str = max_len(image_data)
    tokenizer_lang, word_matrix = tokenizer(image_data, max_len=max_len_str, max_features=max_features)

    image_translation = translate_corpus(word_matrix, tokenizer_lang, decoding_dictionary, output_max_len=max_len_str)

    return image_translation

