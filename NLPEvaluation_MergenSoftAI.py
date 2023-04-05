import gradio as gr
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from mergen import StringAnalyzer
def auth(username, password):
    if username == "MergenSoftAI" and password == "ERI5WBPY596EDH5J":
        return True
    else:
        return False

def lr_predict(text : str):
    # Model dosyasını yükle
    with open('model.pkl', 'rb') as f:
        lr = pickle.load(f)

    # Diğer dosyaları yükle
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Metin Ön İşleme
    text = StringAnalyzer(sentence=text).sentence_analyze()
    # Sayısallaştıma İşlemi
    x_test = vectorizer.transform([text])
    # Tahmin Yapma
    y_predict = lr.predict(x_test) #y_predict[0]
    #{'INSULT': 0, 'OTHER': 1, 'PROFANITY': 2, 'RACIST': 3, 'SEXIST': 4}
    if y_predict[0] == 'OTHER':
        print("0: NO_OFFENSIVE")
    elif y_predict[0] == 'INSULT':
        print(f'1: OFFENSIVE \nTarget: {y_predict[0]}')
    elif y_predict[0] == 'PROFANITY':
        print(f'1: OFFENSIVE \nTarget: {y_predict[0]}')
    elif y_predict[0] == 'RACIST':
        print(f'1: OFFENSIVE \nTarget: {y_predict[0]}')
    elif y_predict[0] == 'SEXIST':
        print(f'1: OFFENSIVE \nTarget: {y_predict[0]}')
    else:
        raise KeyError(f"{text} cümlesi ve/veya kelimesine ait kategori bulunamadı")

def lr_predicts(text : str):
    # Model dosyasını yükle
    with open('model.pkl', 'rb') as f:
        lr = pickle.load(f)

    # Diğer dosyaları yükle
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Metin Ön İşleme
    text = StringAnalyzer(sentence=text).sentence_analyze()
    # Sayısallaştıma İşlemi
    x_test = vectorizer.transform([text])
    # Tahmin Yapma
    y_predict = lr.predict(x_test)  # y_predict[0]
    # {'INSULT': 0, 'OTHER': 1, 'PROFANITY': 2, 'RACIST': 3, 'SEXIST': 4}
    if y_predict[0] == 'OTHER':
        return 1
    elif y_predict[0] == 'INSULT':
        return 0
    elif y_predict[0] == 'PROFANITY':
        return 2
    elif y_predict[0] == 'RACIST':
        return 3
    elif y_predict[0] == 'SEXIST':
        return 4
    else:
        raise KeyError(f"{text} cümlesi ve/veya kelimesine ait kategori bulunamadı")
def predict(df):
    """# TODO:
    df["offansive"] = 1
    df["target"] = None"""


    df['target'] = df['text'].apply(lambda x: lr_predicts(x))
    df['offansive'] = df['target'].apply(lambda x: 0 if x == 1 else 1)
    df['target'] = df['target'].apply(lambda x: 'OTHER' if x == 1 else 'INSULT' if x == 0 else 'PROFANITY' if x == 2 else 'RACIST' if x == 3 else 'SEXIST')
    return df


def get_file(file):
    output_file = "output_MergenSoftAI.csv"
    df = pd.read_csv(file.name, sep="|")
    texts = df["text"].tolist()
    targets = predict(texts)
    # For windows users, replace path seperator
    file_name = file.name.replace("\\", "/")

    df = pd.read_csv(file_name, sep="|")

    predict(df)
    df.to_csv(output_file, index=False, sep="|")
    return (output_file)


# Launch the interface with user password
iface = gr.Interface(get_file, "file", "file")

if __name__ == "__main__":
    iface.launch(share=True, auth=auth)

output_file = "output_Lingua.csv"
df = pd.read_csv(file.name, sep="|")

