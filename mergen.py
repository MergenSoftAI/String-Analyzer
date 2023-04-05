import re
import math
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from seqeval.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

class StringAnalyzer:
    def __init__(self, data=None, columns=None, sentence=None, values=None):
        with open('stop_words.txt', 'r', encoding='utf-8') as file:
            stop_word = [word.strip() for word in file.readlines()]

        self.data = data
        self.columns = columns
        self.sentence = sentence
        self.stop_words = stop_word
        self.values = values

    # to_lowercase() Metodu DatFrame ve Sentence veri yapılarında ki ifadeleri küçük harfe çevirir.
    def to_lowercase(self, data=None, columns=None, sentence=None):
        if (self.data is not None) and (self.columns is not None):
            self.data[self.columns] = self.data[self.columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
            return self.data

        if self.sentence is not None:
            self.sentence = self.sentence.lower()
            return self.sentence

        if (data is not None) and (columns is not None):
            data[columns] = data[columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
            return data

        if sentence is not None:
            sentence = sentence.lower()
            return sentence

    # remove_punctuation() Metodu DatFrame ve Sentence veri yapılarında ki ifadelerden noktalama işaretlerini kaldırır.
    def remove_punctuation(self, data=None, columns=None, sentence=None):
        if (self.data is not None) and (self.columns is not None):
            islower = self.data[self.columns].apply(lambda x: x.islower())
            if islower.all():
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: x.translate(str.maketrans('', '', string.punctuation)))
                return self.data
            else:
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: " ".join(x.lower() for x in x.split()))
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: x.translate(str.maketrans('', '', string.punctuation)))
                return self.data

        if self.sentence is not None:
            self.sentence = self.sentence.translate(str.maketrans('', '', string.punctuation))
            return self.sentence

        if (data is not None) and (columns is not None):
            islower = data[columns].apply(lambda x: x.islower())
            if islower.all():
                data[columns] = data[columns].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
                return data
            else:
                data[columns] = data[columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
                data[columns] = data[columns].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
                return data

        if sentence is not None:
            if sentence.islower():
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                return sentence
            else:
                sentence = sentence.lower()
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                return sentence

    # remove_number() Metodu DatFrame ve Sentence veri yapılarında ki ifadelerinden sayıları kaldırır.
    def remove_digits(self, data=None, columns=None, sentence=None):
        if (self.data is not None) and (self.columns is not None):
            islower = self.data[self.columns].apply(lambda x: x.islower())
            if islower.all():
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: ''.join([i for i in x if not i.isdigit()]))
                return self.data
            else:
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: " ".join(x.lower() for x in x.split()))
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: ''.join([i for i in x if not i.isdigit()]))
                return self.data

        if self.sentence is not None:
            self.sentence = re.sub(r'\d+', '', self.sentence)
            return self.sentence

        if (data is not None) and (columns is not None):
            islower = data[columns].apply(lambda x: x.islower())
            if islower.all():
                data[columns] = data[columns].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
                return data
            else:
                data[columns] = data[columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
                data[columns] = data[columns].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
                return data

        if sentence is not None:
            if sentence.islower():
                sentence = re.sub(r'\d+', '', sentence)
                return sentence
            else:
                sentence = sentence.lower()
                sentence = re.sub(r'\d+', '', sentence)
                return sentence

    # remove_hashtags() metodu DataFrame ve Sentence verileri içerisindeki hashtagsleri siler.
    def remove_hashtags(self, data=None, columns=None, sentence=None):
        if (self.data is not None) and (self.columns is not None):
            islower = self.data[self.columns].apply(lambda x: x.islower())
            if islower.all():
                self.data[self.columns] = self.data[self.columns].apply(lambda x: re.sub(r'#\w+', '', x))
                return self.data
            else:
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: " ".join(x.lower() for x in x.split()))
                self.data[self.columns] = self.data[self.columns].apply(lambda x: re.sub(r'#\w+', '', x))
                return self.data

        if self.sentence is not None:
            self.sentence = re.sub(r'#\w+', '', self.sentence)
            return self.sentence

        if (data is not None) and (columns is not None):
            islower = data[columns].apply(lambda x: x.islower())
            if islower.all():
                data[columns] = data[columns].apply(lambda x: re.sub(r'#\w+', '', x))
                return data
            else:
                data[columns] = data[columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
                data[columns] = data[columns].apply(lambda x: re.sub(r'#\w+', '', x))
                return data

        if sentence is not None:
            if sentence.islower():
                sentence = re.sub(r'#\w+', '', sentence)
                return sentence
            else:
                sentence = sentence.lower()
                sentence = re.sub(r'#\w+', '', sentence)
                return sentence

    # remove_stop_word() metodu DataFrame ve Sentence veri yapılarında bulunan stop words kelimelerini silmektedir.
    def remove_stop_word(self, data=None, columns=None, sentence=None):

        if (self.data is not None) and (self.columns is not None):
            islower = self.data[self.columns].apply(lambda x: x.islower())
            if islower.all():
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: ' '.join(x for x in x.split() if x.lower() not in self.stop_words))
                return self.data
            else:
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: " ".join(x.lower() for x in x.split()))
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: ' '.join(x for x in x.split() if x.lower() not in self.stop_words))
                return self.data

        if self.sentence is not None:
            words = self.sentence.lower().split()
            filtered_words = [word for word in words if word not in self.stop_words]
            self.sentence = ' '.join(filtered_words)
            return self.sentence

        if (data is not None) and (columns is not None):
            islower = data[columns].apply(lambda x: x.islower())
            if islower.all():
                data[columns] = data[columns].apply(
                    lambda x: ' '.join(x for x in x.split() if x.lower() not in self.stop_words))
                return data
            else:
                data[columns] = data[columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
                data[columns] = data[columns].apply(
                    lambda x: ' '.join(x for x in x.split() if x.lower() not in self.stop_words))
                return data

        if sentence is not None:
            if sentence.islower():
                sentence = [word for word in sentence.split() if word.lower() not in self.stop_words]
                sentence = ' '.join(sentence)
                return sentence
            else:
                sentence = sentence.lower()
                sentence = [word for word in sentence.split() if word.lower() not in self.stop_words]
                sentence = ' '.join(sentence)
                return sentence

    # remove_email() metodu DataFRame ve Sentence veri yapısı içerisinde bulunan email adresleri siler.
    def remove_email(self, data=None, columns=None, sentence=None):
        if (self.data is not None) and (self.columns is not None):
            islower = self.data[self.columns].apply(lambda x: x.islower())
            if islower.all():
                self.data[self.columns] = self.data[self.columns].apply(lambda x: ' '.join(word for word in x.split() if
                                                                                           word.lower() not in re.findall(
                                                                                               r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',
                                                                                               x)))
                return self.data
            else:
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: " ".join(x.lower() for x in x.split()))
                self.data[self.columns] = self.data[self.columns].apply(lambda x: ' '.join(word for word in x.split() if
                                                                                           word.lower() not in re.findall(
                                                                                               r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',
                                                                                               x)))
                return self.data

        if self.sentence is not None:
            if self.sentence.islower():
                self.sentence = [word for word in self.sentence.split()
                                 if
                                 word.lower() not in re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',
                                                                self.sentence)]
                self.sentence = ' '.join(self.sentence)
                return self.sentence
            else:
                self.sentence = self.sentence.lower()
                self.sentence = [word for word in self.sentence.split()
                                 if
                                 word.lower() not in re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',
                                                                self.sentence)]
                self.sentence = ' '.join(self.sentence)
                return self.sentence

        if (data is not None) and (columns is not None):
            islower = data[columns].apply(lambda x: x.islower())
            if islower.all():
                data[columns] = data[columns].apply(lambda x: ' '.join(word for word in x.split() if
                                                                       word.lower() not in re.findall(
                                                                           r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',
                                                                           x)))
                return data
            else:
                data[columns] = data[columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
                data[columns] = data[columns].apply(lambda x: ' '.join(word for word in x.split() if
                                                                       word.lower() not in re.findall(
                                                                           r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',
                                                                           x)))
                return data

        if sentence is not None:
            if sentence.islower():
                sentence = [word for word in sentence.split()
                            if word.lower() not in re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',
                                                              sentence)]
                sentence = ' '.join(sentence)
                return sentence
            else:
                sentence = sentence.lower()
                sentence = [word for word in sentence.split()
                            if word.lower() not in re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',
                                                              sentence)]
                sentence = ' '.join(sentence)
                return sentence

    # remove_https() metodu DataFrame ve Sentence verileri içerisinde ki https yapılarını siler.
    def remove_https(self, data=None, columns=None, sentence=None):
        if (self.data is not None) and (self.columns is not None):
            islower = self.data[self.columns].apply(lambda x: x.islower())
            if islower.all():
                self.data[self.columns] = self.data[self.columns].apply(lambda x: re.sub(r'(https?://|www\.)\S+', '', x))
                return self.data
            else:
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: " ".join(x.lower() for x in x.split()))
                self.data[self.columns] = self.data[self.columns].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
                return self.data

        if self.sentence is not None:
            if self.sentence.islower():
                self.sentence = re.sub(r'https?:\/\/\S+', '', self.sentence)
                return self.sentence
            else:
                self.sentence = self.sentence.lower()
                self.sentence = re.sub(r'https?:\/\/\S+', '', self.sentence)
                return self.sentence

        if (data is not None) and (columns is not None):
            islower = data[columns].apply(lambda x: x.islower())
            if islower.all():
                data[columns] = data[columns].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
                return data
            else:
                data[columns] = data[columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
                data[columns] = data[columns].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
                return data

        if sentence is not None:
            if sentence.islower():
                sentence = re.sub(r'https?:\/\/\S+', '', sentence)
                return sentence
            else:
                sentence = sentence.lower()
                sentence = re.sub(r'https?:\/\/\S+', '', sentence)
                return sentence

    # remove_this() metodu stop_words() metodu ile aynı mantıkla çalışmakatadır. Ancak stop words yerine kendi listemiz içerisinde bulunan değerleri kaldırmaktadır.
    def remove_this(self, data=None, columns=None, sentence=None, values=None):
        if (self.data is not None) and (self.columns is not None):
            islower = self.data[self.columns].apply(lambda x: x.islower())

            if not islower.all():
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: " ".join(x.lower() for x in x.split()))
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: ' '.join(x for x in x.split() if x.lower() not in self.values))
                return self.data
            else:
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: ' '.join(x for x in x.split() if x.lower() not in self.values))
                return self.data

        elif self.sentence is not None:
            words = self.sentence.lower().split()
            filtered_words = [word for word in words if word not in self.values]
            self.sentence = ' '.join(filtered_words)
            return self.sentence

        elif (data is not None) and (columns is not None):
            islower = data[columns].apply(lambda x: x.islower())
            if islower.all():
                data[columns] = data[columns].apply(lambda x: ' '.join(x for x in x.split() if x.lower() not in values))
                return data
            else:
                data[columns] = data[columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
                data[columns] = data[columns].apply(lambda x: ' '.join(x for x in x.split() if x.lower() not in values))
                return data

        elif sentence is not None:
            if sentence.islower():
                sentence = [word for word in sentence.split() if word.lower() not in values]
                sentence = ' '.join(sentence)
                return sentence
            else:
                words = sentence.lower().split()
                filtered_words = [word for word in words if word not in values]
                sentence = ' '.join(filtered_words)
                return sentence

    # remove_more_space() metodu veri içerisindeki fazla boşlukların silinmesi için kullanılmaktadır.
    def remove_more_space(self, data=None, columns=None, sentence=None):
        pattern = re.compile(r'\s{2,}')
        if (self.data is not None) and (self.columns is not None):
            islower = self.data[self.columns].apply(lambda x: x.islower())
            if islower.all():
                self.data[self.columns] = self.data[self.columns].apply(lambda x: re.sub(pattern, ' ', x))
                return self.data
            else:
                self.data[self.columns] = self.data[self.columns].apply(
                    lambda x: " ".join(x.lower() for x in x.split()))
                self.data[self.columns] = self.data[self.columns].apply(lambda x: re.sub(pattern, ' ', x))
                return self.data

        if self.sentence is not None:
            if self.sentence.islower():
                self.sentence = re.sub(pattern, ' ', self.sentence)
                return self.sentence
            else:
                self.sentence = self.sentence.lower()
                self.sentence = re.sub(pattern, ' ', self.sentence)
                return self.sentence

        if (data is not None) and (columns is not None):
            islower = data[columns].apply(lambda x: x.islower())
            if islower.all():
                data[columns] = data[columns].apply(lambda x: re.sub(pattern, ' ', x))
                return data
            else:
                data[columns] = data[columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
                data[columns] = data[columns].apply(lambda x: re.sub(pattern, ' ', x))
                return data

        if sentence is not None:
            if sentence.islower():
                sentence = re.sub(pattern, ' ', sentence)
                return sentence
            else:
                sentence = sentence.lower()
                sentence = re.sub(pattern, ' ', sentence)
                return sentence

    # remove_two_char() metodu metin içerisindeki 2 veya daha az kelimeleri silmektedir.
    def remove_two_char(self, data = None, columns = None, sentence = None):
        if (self.data is not None) and (self.columns is not None):
            islower = self.data[self.columns].apply(lambda x: x.islower())
            if islower.all():
                self.data[self.columns] = self.data[self.columns].apply(lambda x: " ".join([x for x in x.split() if len(x) > 2]))
                return self.data
            else:
                self.data[self.columns] = self.data[self.columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
                self.data[self.columns] = self.data[self.columns].apply(lambda x: " ".join([x for x in x.split() if len(x) > 2]))
                return self.data


        if (data is not None) and (columns is not None):
            islower = data[columns].apply(lambda x: x.islower())
            if islower.all():
                data[columns] = data[columns].apply(lambda x: " ".join([x for x in x.split() if len(x) > 2]))
                return data
            else:
                data[columns] = data[columns].apply(lambda x: " ".join(x.lower() for x in x.split()))
                data[columns] = data[columns].apply(lambda x: " ".join([x for x in x.split() if len(x) > 2]))
                return data

        if self.sentence is not None:
            if self.sentence.islower():
                kelimeler = self.sentence.split()  # metindeki kelimeleri ayır
                # kısa kelimeleri çıkar
                kelimeler = [kelime for kelime in kelimeler if len(kelime) > 2]
                temiz_metin = " ".join(kelimeler)
                return temiz_metin
            else:
                kelimeler = self.sentence.split()  # metindeki kelimeleri ayır
                # kısa kelimeleri çıkar
                kelimeler = [kelime for kelime in kelimeler if len(kelime) > 2]
                temiz_metin = " ".join(kelimeler)
                return temiz_metin

        if sentence is not None:
            if sentence.islower():
                kelimeler = sentence.split()  # metindeki kelimeleri ayır
                # kısa kelimeleri çıkar
                kelimeler = [kelime for kelime in kelimeler if len(kelime) > 2]
                temiz_metin = " ".join(kelimeler)
                return temiz_metin
            else:
                sentence = sentence.lower()
                kelimeler = sentence.split()  # metindeki kelimeleri ayır
                # kısa kelimeleri çıkar
                kelimeler = [kelime for kelime in kelimeler if len(kelime) > 2]
                temiz_metin = " ".join(kelimeler)
                return temiz_metin

    def normalization(self):
        pass

    def is_lower(self, data = None, columns = None, sentence = None):
        if self.data is not None and self.columns is not None:
            return self.data[self.columns].apply(lambda x: x.islower())

        if self.sentence is not None:
            return self.sentence.islower()

        if data is not None and columns is not None:
            return data[columns].apply(lambda x: x.islower())

        if sentence is not None:
            return sentence.islower()

    def is_punctuation(self, data = None, columns = None, sentence = None):
        punctuation = string.punctuation
        if self.data is not None and self.columns is not None:
            return self.data[self.columns].apply(lambda x: True if any(char in punctuation for char in x) else False)

        if self.sentence is not None:
            for char in self.sentence:
                if char in punctuation:
                    return True
            return False

        if data is not None and columns is not None:
            return data[columns].apply(lambda x: True if any(char in punctuation for char in x) else False)

        if sentence is not None:
            for char in self.sentence:
                if char in punctuation:
                    return True
            return False

    def is_digit(self, data = None, columns = None, sentence = None):
        digits = string.digits
        if self.data is not None and self.columns is not None:
            digit = '|'.join(digits)
            return self.data[self.columns].str.contains(digit)

        if self.sentence is not None:
            for char in self.sentence:
                if char in digits:
                    return True
            return False

        if data is not None and columns is not None:
            digit = '|'.join(digits)
            return data[columns].str.contains(digit)

        if sentence is not None:
            for char in sentence:
                if char in digits:
                    return True
            return False

    def is_stop_word(self, data = None, columns = None, sentence = None):
        stop_words_str = '|'.join(self.stop_words)
        if self.data is not None and self.data is not None:
            return self.data[self.columns].str.contains(stop_words_str)

        if self.sentence is not None:
            for word in self.sentence.lower().split():
                if word in self.stop_words:
                    return True
            return False

        if data is not None and columns is not None:
            return data[columns].str.contains(stop_words_str)

        if sentence is not None:
            for word in sentence.lower().split():
                if word in self.stop_words:
                    return True
            return False

    def is_hashtag(self, data = None, columns = None, sentence = None):
        if self.data is not None and self.columns is not None:
            return self.data[self.columns].str.contains('#')

        if self.sentence is not None:
            for char in self.sentence:
                if char in "#":
                    return True
            return False

        if data is not None and columns is not None:
            return data[columns].str.contains('#')

        if sentence is not None:
            for char in sentence:
                if char in "#":
                    return True
            return False

    def is_email(self, data = None, columns = None, sentence = None):
        if self.data is not None and self.columns is not None:
            return self.data[self.columns].str.contains('@')

        if self.sentence is not None:
            for char in self.sentence:
                if char in "@":
                    return True
            return False

        if data is not None and columns is not None:
            return data[columns].str.contains('@')

        if sentence is not None:
            for char in sentence:
                if char in '@':
                    return True
            return False

    def is_https(self, data = None, columns = None, sentence = None):
        if self.data is not None and self.columns is not None:
            return self.data[self.columns].str.contains('https://|www|com')

        if self.sentence is not None:
            liste = ['https', 'http', 'https://','www','com']
            for word in self.sentence.lower().split(" "):
                for _ in liste:
                    if word.startswith(_) or word.endswith(_):
                        return True
            return False

        if data is not None and columns is not None:
            return data[columns].str.contains('https://|www|com')

        if sentence is not None:
            liste = ['https', 'http', 'https://','www','com']
            for word in sentence.lower().split(" "):
                for _ in liste:
                    if word.startswith(_) or word.endswith(_):
                        return True
            return False

    def is_this(self, data = None, columns = None, sentence = None, values = None):
        if self.data is not None and self.columns is not None:
            values = '|'.join(self.values)
            return self.data[self.columns].str.contains(values)

        if self.sentence is not None:
            for word in self.sentence.lower().split():
                if word in self.values:
                    return True
            return False

        if data is not None and columns is not None:
            values = '|'.join(values)
            return data[columns].str.contains(values)

        if sentence is not None:
            for word in sentence.lower().split():
                if word in values:
                    return True
            return False


        if self.sentence is not None:
            pass

    def is_space(self, data = None, columns = None, sentence = None):
        if self.data is not None and self.columns is not None:
            return self.data[self.columns].apply(lambda x: True if x.count(' ') >= 2 else False)

        if self.sentence is not None:
            if self.sentence.count(' ') >= 2:
                return True
            else:
                return False

        if data is not None and columns is not None:
            return data[columns].apply(lambda x: True if x.count(' ') >= 2 else False)

        if sentence is not None:
            if sentence.count(' ') >= 2:
                return True
            else:
                return False

    def sentence_analyze(self):
        if self.values is None:
            self.sentence = self.to_lowercase()
            self.sentence = self.remove_https()
            self.sentence = self.remove_email()
            self.sentence = self.remove_hashtags()
            self.sentence = self.remove_digits()
            self.sentence = self.remove_punctuation()
            self.sentence = self.remove_two_char()
            self.sentence = self.remove_stop_word()
            self.sentence = self.remove_more_space()
            return self.sentence
        else:
            self.sentence = self.to_lowercase()
            self.sentence = self.to_lowercase()
            self.sentence = self.remove_https()
            self.sentence = self.remove_email()
            self.sentence = self.remove_hashtags()
            self.sentence = self.remove_digits()
            self.sentence = self.remove_punctuation()
            self.sentence = self.remove_stop_word()
            self.sentence = self.remove_this()
            self.sentence = self.remove_two_char()
            self.sentence = self.remove_more_space()
            return self.sentence

    def dataframe_analyze(self):
        if self.values is None:
            self.data = self.to_lowercase()
            self.data = self.remove_https()
            self.data = self.remove_email()
            self.data = self.remove_hashtags()
            self.data = self.remove_digits()
            self.data = self.remove_punctuation()
            self.data = self.remove_stop_word()
            self.data = self.remove_two_char()
            self.data = self.remove_more_space()
            return self.data
        else:
            self.data = self.to_lowercase()
            self.data = self.remove_https()
            self.data = self.remove_email()
            self.data = self.remove_hashtags()
            self.data = self.remove_digits()
            self.data = self.remove_punctuation()
            self.data = self.remove_stop_word()
            self.data = self.remove_this()
            self.data = self.remove_two_char()
            self.data = self.remove_more_space()
            return self.data


class StringProfiling:
    def __init__(self, data = None, columns = None, category_columns = None, values = None):
        with open('stop_words.txt', 'r', encoding='utf-8') as file:
            stop_word = [word.strip() for word in file.readlines()]
        self.data = data
        self.columns = columns
        self.category_columns = category_columns
        self.values = values
        self.stop_words = stop_word
        self.punctuation = set(string.punctuation)
        self.data_copy = self.data.copy()



    def write(self, graphs = False, describe = True, info = False):
        if describe:
            df1 = self.count_by_columns()
            df2 = self.count_words_by_category()
            df3 = self.count_stop_words_by_category()
            df4 = self.count_punctuations_by_category()
            df5 = self.count_numbers_by_category()
            df6 = self.count_values_by_category()
            df7 = self.count_https_by_category()
            df8 = self.count_hashtag_by_category()
            df9 = self.count_email_by_category()
            df2_9 = pd.concat([df2,df3,df4,df5,df6,df7,df8,df9],ignore_index=False,axis=1)
            result = pd.concat([df1,df2_9],ignore_index=False,axis=0)
            result.index.names = ['category']
            return result

        if graphs:
            df1 = self.count_by_columns()
            df2 = self.count_words_by_category()
            df3 = self.count_stop_words_by_category()
            df4 = self.count_punctuations_by_category()
            df5 = self.count_numbers_by_category()
            df6 = self.count_values_by_category()
            df7 = self.count_https_by_category()
            df8 = self.count_hashtag_by_category()
            df9 = self.count_email_by_category()
            df2_9 = pd.concat([df2,df3,df4,df5,df6,df7,df8,df9],ignore_index=False,axis=1)
            result = pd.concat([df1,df2_9],ignore_index=False,axis=0)
            result.index.names = ['category']
            fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(10, 10))
            sns.barplot(x = "total_digit_count", y = "total_word_count",hue = result.index,data = result,ax=axs[0,0])
            sns.barplot(x = "total_punctuation_count", y = "total_word_count",hue = result.index,data = result,ax=axs[0,1])
            sns.barplot(x = "total_stop_word_count", y = "total_word_count",hue = result.index,data = result,ax=axs[1,0])
            sns.barplot(x = "total_values_count", y = "total_word_count",hue = result.index,data = result,ax=axs[1,1])
            result.plot.barh(ax=axs[2,0])

        if info:
            return f"Veri Yapısı (satır, sütun) : {self.data.shape} ", self.data.info()

    def calculate_number_of_words_data(self, describe = False):
        if describe:
            self.data_copy['words'] = [len(x.split()) for x in self.data_copy[self.columns].tolist()]
            df = self.data_copy.groupby([self.category_columns])['words'].describe()
            return df
        else:
            self.data_copy['words'] = [len(x.split()) for x in self.data_copy[self.columns].tolist()]
            return self.data_copy

    def extract(self, minimum = None, maximum = None):
        df = self.calculate_number_of_words_data()
        if minimum is None and maximum is None:
            minimum = math.ceil(df.groupby([self.category_columns])['words'].describe()['min'].mean())
            maximum = math.ceil(df.groupby([self.category_columns])['words'].describe()['max'].mean())
            percent_25 = math.ceil(df.groupby([self.category_columns])['words'].describe()['25%'].mean())
            percent_50 = math.ceil(df.groupby([self.category_columns])['words'].describe()['50%'].mean())
            percent_75 = math.ceil(df.groupby([self.category_columns])['words'].describe()['75%'].mean())

            if minimum != 0 and maximum != 0:
                df = df[(df['words'] >= minimum) & (df['words'] <= maximum)]
                df = df.drop('words', axis=1)
                return df
            elif percent_25 != 0 and maximum != 0:
                df = df[(df['words'] >= percent_25) & (df['words'] <= maximum)]
                df = df.drop('words', axis=1)
                return df
            elif percent_50 != 0 and maximum != 0:
                df = df[(df['words'] >= percent_50) & (df['words'] <= maximum)]
                df = df.drop('words', axis=1)
                return df
            elif percent_75 != 0 and maximum != 0:
                df = df[(df['words'] >= percent_75) & (df['words'] <= maximum)]
                df = df.drop('words', axis=1)
                return df
            else:
                print("Hata")
        else:
            minimum = minimum
            maximum = maximum
            if minimum != 0 and maximum != 0:
                df = df[(df['words'] >= minimum) & (df['words'] <= maximum)]
                df = df.drop('words', axis=1)
                return df
            else:
                print("İşlem gerçekleştirilemedi!")

    def count_by_columns(self):
        # DataFrame Kopyalandı
        data = self.data.copy()

        # DataFrame için StringAnalyzer nesnesi oluşturuldu
        string_analyzer = StringAnalyzer(data=self.data, columns=self.columns)

        # Küçük Harfe Çevirme, Noktalama İşaretlerinin Kaldırılması, Sayıların Kaldırılması
        data_frame = string_analyzer.to_lowercase()
        data_frame = string_analyzer.remove_punctuation()
        data_frame = string_analyzer.remove_digits()

        # Toplam kelime sayısı
        total_word_count = len(' '.join(data_frame[self.columns]).split())

        # Toplam stop word sayısı
        total_stop_word_count = sum(data_frame[self.columns].apply(lambda x: len([word for word in x.split() if word.lower() in self.stop_words])))

        # Toplam values sayısı
        total_values = sum([len([word for word in x.split() if word.lower() in self.values]) for x in data_frame[self.columns].tolist()])

        # Kopyalanan DataFrame için StringAnalyzer nesnesi oluşturuldu.
        _string_analyzer = StringAnalyzer(data=data, columns=self.columns)

        # Kopylanan DataFrame Küçük Harfe Çevirme İşlemesi
        data = _string_analyzer.to_lowercase()

        # Toplam noktalama işareti sayısı
        total_punctuation_count = sum([len([char for char in x if char in string.punctuation]) for x in data[self.columns].tolist()])

        # Toplam rakamların sayısı
        def sayi_adeti(metin):
            # İçindeki tüm sayıları bulalım
            sayilar = re.findall('\d+', metin)
            # İki liste birleştirip toplam sayı adedini döndürelim
            return len(sayilar)
        data['sayi_adetleri'] = data[self.columns].apply(sayi_adeti)
        _numbers_count = data['sayi_adetleri'].sum()
        total_digit_count = _numbers_count

        # Toplam https sayısı
        total_https_count = sum(data[self.columns].str.count('https?://'))

        # Toplam hashtag sayısı
        total_hashtag_count = sum(data[self.columns].str.count('#\w+'))

        # Toplam email sayısı
        total_email_count = sum(data[self.columns].str.count('\w+@\w+\.\w+'))

        # Sonuçları dictionary olarak döndür
        result_dict = {
            'total_word_count': total_word_count,
            'total_stop_word_count': total_stop_word_count,
            'total_punctuation_count': total_punctuation_count,
            'total_digit_count': total_digit_count,
            'total_https_count': total_https_count,
            'total_hashtag_count': total_hashtag_count,
            'total_email_count': total_email_count,
            'total_values_count': total_values
        }

        return pd.DataFrame(result_dict, index=[self.columns])

    def count_words_by_category(self):
        word_counts = {}
        for category in self.data[self.category_columns].unique():
            category_data = self.data[self.data[self.category_columns] == category]
            category_text = ' '.join(category_data[self.columns])
            words = category_text.split()
            word_counts[category] = len(words)
        df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['total_word_count'])
        df.index.name = 'category'
        return df

    def count_stop_words_by_category(self):
        stop_word_counts = {}
        for category in self.data[self.category_columns].unique():
            category_data = self.data[self.data[self.category_columns] == category]
            category_text = ' '.join(category_data[self.columns])
            words = category_text.split()
            stop_words_count = len([word for word in words if word.lower() in self.stop_words])
            stop_word_counts[category] = stop_words_count
        df = pd.DataFrame.from_dict(stop_word_counts, orient='index', columns=['total_stop_word_count'])
        df.index.name = 'category'
        return df

    def count_punctuations_by_category(self):
        punctuation_counts = {}
        for category in self.data_copy[self.category_columns].unique():
            category_data = self.data_copy[self.data_copy[self.category_columns] == category]
            category_text = ' '.join(category_data[self.columns])
            punctuations_count = len([char for char in category_text if char in self.punctuation])
            punctuation_counts[category] = punctuations_count
        df = pd.DataFrame.from_dict(punctuation_counts, orient='index', columns=['total_punctuation_count'])
        df.index.name = 'category'
        return df

    def count_numbers_by_category(self):
        # Sayı adetlerini bulmak için bir fonksiyon yazalım len([word for word in category_text.split() if word.isnumeric()])
        def sayi_adeti(metin):
            # İçindeki tüm sayıları bulalım
            sayilar = re.findall('\d+', metin)
            # İki liste birleştirip toplam sayı adedini döndürelim
            return len(sayilar)
        number_counts = {}
        for category in self.data_copy[self.category_columns].unique():
            category_data = self.data_copy[self.data_copy[self.category_columns] == category]
            category_data['sayi_adetleri'] = category_data[self.columns].apply(sayi_adeti)
            numbers_count = category_data['sayi_adetleri'].sum()
            number_counts[category] = numbers_count
        df = pd.DataFrame.from_dict(number_counts, orient='index', columns=['total_digit_count'])
        df.index.name = 'category'
        return df

    def count_https_by_category(self):
        https_counts = {}
        for category in self.data_copy[self.category_columns].unique():
            category_data = self.data_copy[self.data_copy[self.category_columns] == category]
            category_text = ' '.join(category_data[self.columns])
            https_count = category_text.count('https://')
            https_counts[category] = https_count
        df = pd.DataFrame.from_dict(https_counts, orient='index', columns=['total_https_count'])
        df.index.name = 'category'
        return df

    def count_hashtag_by_category(self):
        hashtag_counts = {}
        for category in self.data_copy[self.category_columns].unique():
            category_data = self.data_copy[self.data_copy[self.category_columns] == category]
            category_text = ' '.join(category_data[self.columns])
            hashtag_count = category_text.count('#')
            hashtag_counts[category] = hashtag_count
        df = pd.DataFrame.from_dict(hashtag_counts, orient='index', columns=['total_hashtag_count'])
        df.index.name = 'category'
        return df

    def count_email_by_category(self):
        email_counts = {}
        for category in self.data_copy[self.category_columns].unique():
            category_data = self.data_copy[self.data_copy[self.category_columns] == category]
            category_text = ' '.join(category_data[self.columns])
            email_count = len(re.findall(r'\S+@\S+', category_text))
            email_counts[category] = email_count
        df = pd.DataFrame.from_dict(email_counts, orient='index', columns=['total_email_count'])
        df.index.name = 'category'
        return df

    def count_values_by_category(self):
        values_counts = {}
        for category in self.data[self.category_columns].unique():
            category_data = self.data[self.data[self.category_columns] == category]
            category_text = ' '.join(category_data[self.columns])
            words = category_text.split()
            stop_words_count = len([word for word in words if word.lower() in self.values])
            values_counts[category] = stop_words_count
        df = pd.DataFrame.from_dict(values_counts, orient='index', columns=['total_values_count'])
        df.index.name = 'category'
        return df


class StringClassification:
    def __init__(self, data = None, columns = None, category_columns = None):
        self.data = data
        self.columns = columns
        self.category_columns = category_columns
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None

    def LogisticRegression(self, test_size = 0.1):
        # Metin verilerinin sayısallaştırılması
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.data[self.columns])
        y = self.data[self.category_columns]

        # Eğitim ve test verilerinin ayrılması
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
        self.model = lr.fit(self.X_train, self.y_train)
        return self.model

    def classification_report(self):
        # Test verileri üzerinde tahmin işlemi yap
        y_pred = self.model.predict(self.X_test)

        # Sınıflandırma raporunu göster
        return classification_report(self.y_test, y_pred)


    def predicts(self, sentence : str):
        # Ön işleme işlemi
        text = StringAnalyzer(sentence=sentence).sentence_analyze()
        # Sayısallaştırma işlemi
        vectorizer = CountVectorizer()
        x_test = vectorizer.transform([text])
        # Tahmin işlemi
        y_predict = self.model.predict(x_test)
        return y_predict[0]

    def predict_categories(self, df):
        df['target'] = df['text'].apply(lambda x: self.predicts(sentence=x))
        df['offansive'] = df['target'].apply(lambda x: 0 if x == 1 else 1)
        df['target'] = df['target'].apply(lambda x: 'OTHER' if x == 1 else 'INSULT' if x == 0 else 'PROFANITY' if x == 2 else 'RACIST' if x == 3 else 'SEXIST')
        return df


    def LSTM(self):
        # The maximum number of words to be used. (most frequent)
        MAX_NB_WORDS = 20000
        # Max number of words in each complaint.
        MAX_SEQUENCE_LENGTH = 25
        # This is fixed.
        EMBEDDING_DIM = 100

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(self.data[self.columns].values)
        word_index = tokenizer.word_index
        X = tokenizer.texts_to_sequences(self.data[self.columns].values)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        Y = pd.get_dummies(self.data[self.category_columns]).values
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        epochs = 5
        batch_size = 256

        self.model = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    def predicts_lstm(self, sentence):
        new_complaint = [sentence]
        new_complaint = StringAnalyzer(sentence=new_complaint).sentence_analyze()
        tokenizer = Tokenizer()
        seq = tokenizer.texts_to_sequences(new_complaint)
        padded = pad_sequences(seq, maxlen=25)
        pred = self.model.predict(padded)
        labels = self.data[self.category_columns].unique().tolist()
        return np.argmax(pred)

    def save(self, file_name : str):
        self.model.save(f'{file_name}.h5')

    def load_model(self, file_name : str):
        import tensorflow as tf
        new_model = tf.keras.models.load_model(f'{file_name}.h5')
        return new_model