"""#3) OCR texto en chino"""

! apt install tesseract-ocr
! apt install libtesseract-dev

pip install pytesseract

import shutil

#Archivos preentrenados de chino tradicional y simplificado para tesseract
zh_sim_file = "/content/gdrive/My Drive/tfm/chi_sim_vert.traineddata"
zh_tra_file = "/content/gdrive/My Drive/tfm/chi_tra_vert.traineddata"
shutil.copy(zh_sim_file, "/usr/share/tesseract-ocr/4.00/tessdata")
shutil.copy(zh_tra_file, "/usr/share/tesseract-ocr/4.00/tessdata")

"""##3.1 Importar imagen como string"""

from PIL import Image
from pytesseract import image_to_string

"""La imagen se importa con cada caracter separado. Esto dificulta la tokenizacion

Preprocesado:
1)Eliminar todos los espacios
2)Aplicar funcion inicial de preprocesado para separar por frases y tokenizar por palabras
"""

#Funciones de pre procesado
def from_image_to_text(image):
    im = Image.open(image)
    text_im = image_to_string((im), lang='chi_sim_vert+chi_tra_vert')

    text = re.sub(" ", "", text_im) #Eliminamos espacios para poder aplicar nuestro preprocesado
    return text

"""##3.2 Traducir contenido de la imagen"""

def translate_image(image_data, decoding_dictionary, max_features=1000):

    max_len_str = max_len(image_data)
    tokenizer_lang, word_matrix = tokenizer(image_data, max_len=max_len_str, max_features=max_features)

    image_translation = translate_corpus(word_matrix, tokenizer_lang,\
                        decoding_dictionary, output_max_len=max_len_str )

    return image_translation

imagen_prueba = "/content/gdrive/My Drive/tfm/img_prueba.png"

#Prueba
texto_imagen = from_image_to_text(imagen_prueba)
traducción = translate_image(texto_imagen, decode_en)