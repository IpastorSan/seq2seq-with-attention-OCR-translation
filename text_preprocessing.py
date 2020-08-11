"""#1) Dataset creation and preprocessing
hanziconv: It helps with the transformation between traditional and simplified Chinese characters
https://pypi.org/project/hanziconv/0.3/

jieba: Herramienta que ayuda a la tokenizacion, separando los caracteres chinos por palabras
https://github.com/fxsjy/jieba
"""

pip
install
hanziconv
pip
install
jieba

import re
import unicodedata
import io
import jieba
from hanziconv import HanziConv
import string

"""##1.1 Open File"""


# Funcion abrir archivos y guardarlos en variable
def open_txt(dataset):
    texto = open(dataset, mode="r", encoding="utf-8")
    datos = texto.read()
    texto.close
    return datos


path = "/content/gdrive/My Drive/tfm/manythings_en_zh.txt"
path_base = "/content/gdrive/My Drive/tfm/"

"""##1.2 unicode to ascii"""


# Funcion de limpieza: Convertir a ASCII, eliminar acentos, pasar a minuscula
def unicode_to_ascii(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto)
                   if unicodedata.category(c) != 'Mn')


"""##1.3 Preprocesado de texto"""


def preprocessing_zh(texto):
    t = unicode_to_ascii(texto.lower().strip())  # strip()quita espacios blancos de delante y atras
    # añadir un espacio entre las palabras y los simbolos de puntuacion
    t = re.sub(r"[^?.!,¿,。！？\u4e00-\u9fff]+", " ", t)  # deberia ir primero?
    t = re.sub(r"([?.!,¿])", r" \1 ", t)
    t = re.sub(r'[" "]+', " ", t)
    t = t.strip()
    t = t.lstrip(".")  # cuanto quitamos numeros de puntos del dia queda un punto suelto raro
    t = t.lstrip(",")
    t = jieba.cut(t,
                  cut_all=False)  # jieba, herramienta para dividir la frase en palabras y añadir espacios entre ellas
    t = " ".join(t)
    t = HanziConv.toTraditional(t)
    return t


def preprocessing_es(texto):
    t = unicode_to_ascii(texto.lower().strip())  # strip()quita espacios blancos de delante y atras
    # añadir un espacio entre las palabras y los simbolos de puntuacion
    t = re.sub(r"([?.!,¿])", r" \1 ", t)
    t = re.sub(r'[" "]+', " ", t)
    # t = re.sub(r'\w*\d\w*', '', texto) #Elimina el pinyin
    # reemplazar todo con espacio excepto y characteres chinos (a-z, A-Z, ".", "?", "!", ",")
    t = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", t)  # Mal, hay que conservar la puntuacion
    t = t.strip()
    t = re.sub(r'\w*\d\w*', '', t)  # Elimina el pinyin de tatoeba pairs
    t = t.lstrip(".")  # cuanto quitamos numeros de puntos del dia queda un punto suelto raro
    t = t.lstrip(",")
    t = re.sub(r'^.*?\\.', "", t)
    t = "<start> " + t + " <end>"
    return t


zh = ", 我会借这本特别好的书吗。hola que tal"
print(preprocessing_zh(zh))

zh_sentence = "我会借这本书?"
sp_sentence = "¿Can I borrow this book??"
print(preprocessing_zh(zh_sentence))
print(preprocessing_es(sp_sentence))

"""##1.4 Combinar en un dataset"""


# Datasets independientes
def create_dataset_ind(path, language="es"):
    lines = io.open(path, encoding="UTF-8").read().strip().split("\n")
    text = [l.split("\t", 2)[0:2] for l in lines]

    if language == "zh":
        dataset = [preprocessing_zh(text[t][1]) for t in range(len(text))]

    else:
        dataset = [preprocessing_es(text[t][0]) for t in range(len(text))]

    return dataset


dataset_zh = create_dataset_ind(path, language="zh")
dataset_en = create_dataset_ind(path)

"""##1.5 Guardar el dataset"""


def save_dataset(data, filename="dataset"):
    with open(path_base + filename + ".txt", mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(str(data[i]) for i in range(1, len(data))))


dataset_zh = save_dataset(dataset_zh, "dataset_zh")
dataset_en = save_dataset(dataset_en, "dataset_en")