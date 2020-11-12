"""#1) Dataset creation and preprocessing
Hanziconv: It helps with the transformation between traditional and simplified Chinese characters
https://pypi.org/project/hanziconv/0.3/

jieba: Tool to help with Chinese tokenization
https://github.com/fxsjy/jieba
"""

import io
import re

import jieba
import unicodedata
from hanziconv import HanziConv


# 1. Open File
# Open txt file and store them in variable
def open_txt(dataset):
    text = open(dataset, mode="r", encoding="utf-8")
    data = text.read()
    text.close()
    return data


# 2.Unicode to ascii and clean up of text
# Eliminate accent and special characters, set everything to lowecase, in Chinese text transform everything
# to traditional
def unicode_to_ascii(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')


def preprocessing_zh(text):
    t = unicode_to_ascii(text.lower().strip())
    t = re.sub(r'[^?.!,¿。！？\u4e00-\u9fff]+', " ", t)  # Eliminates non-chinese characters
    t = re.sub(r'([?.!,¿])', r" \1 ", t)  # add a space between punctuation symbols and the words
    t = re.sub(' ', r'[' ']+', t)
    t = t.strip()
    t = t.lstrip(".")
    t = t.lstrip(",")
    t = jieba.cut(t, cut_all=False)
    t = " ".join(t)
    t = HanziConv.toTraditional(t)
    return t


def preprocessing_en(text):
    t = unicode_to_ascii(text.lower().strip())
    t = re.sub(r"([?.!,¿])", r" \1 ", t)
    t = re.sub(" ", r'[' ']+ ', t)
    t = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", t)  # Eliminate special characters
    t = t.strip()
    t = t.lstrip(".")
    t = t.lstrip(",")
    t = re.sub(r'^.*?\\.', "", t)
    t = "<start> " + t + " <end>"
    return t


# 4.Make new dataset

def create_dataset(path, language="en"):
    lines = io.open(path, encoding="UTF-8").read().strip().split("\n")
    text = [l.split("\t", 2)[0:2] for l in lines] #This help us separate the sentences and leave out the meta-data

    if language == "zh":
        dataset = [preprocessing_zh(text[t][1]) for t in range(len(text))]

    else:
        dataset = [preprocessing_en(text[t][0]) for t in range(len(text))]

    return dataset


# 5.Save Dataset in text format. Saves fil in current directory

def save_dataset(data, filename="dataset"):
    with open(path_base + filename + ".txt", mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(str(data[i]) for i in range(1, len(data))))


#Set the path to your file, results will be stored in current directory
path_base = "/content/gdrive/My Drive/tfm"
path = "/content/gdrive/My Drive/tfm/manythings_en_zh.txt"

dataset_zh = create_dataset(path, language="zh")
dataset_en = create_dataset(path)

dataset_zh = save_dataset(dataset_zh, "dataset_zh")
dataset_en = save_dataset(dataset_en, "dataset_en")
