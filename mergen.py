import re
import string


class StringAnalyzer:
    def __init__(self, data=None, columns=None, sentence=None, values=None):
        with open('stop_words.txt', 'r', encoding='utf-8') as file:
            stop_word = [word.strip() for word in file.readlines()]

        self.data = data
        self.columns = columns
        self.sentence = sentence
        self.stop_words = stop_word
        self.values = values

    # `lower()` Metodu DatFrame ve Sentence veri yapılarında ki ifadeleri küçük harfe çevirir.
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

    # `punctuation()` Metodu DatFrame ve Sentence veri yapılarında ki ifadelerden noktalama işaretlerini kaldırır.
    def punctuation(self, data=None, columns=None, sentence=None):
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

    # number() Metodu DatFrame ve Sentence veri yapılarında ki ifadelerinden sayıları kaldırır.
    def number(self, data=None, columns=None, sentence=None):
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

    # hashtags() metodu DataFrame ve Sentence verileri içerisindeki hashtagsleri siler.
    def hashtags(self, data=None, columns=None, sentence=None):
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

    # stop_word() metodu DataFrame ve Sentence veri yapılarında bulunan stop words kelimelerini silmektedir.
    def stop_word(self, data=None, columns=None, sentence=None):

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

    # email() metodu DataFRame ve Sentence veri yapısı içerisinde bulunan email adresleri siler.
    def email(self, data=None, columns=None, sentence=None):
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

    # https() metodu DataFrame ve Sentence verileri içerisinde ki https yapılarını siler.
    def https(self, data=None, columns=None, sentence=None):
        if (self.data is not None) and (self.columns is not None):
            islower = self.data[self.columns].apply(lambda x: x.islower())
            if islower.all():
                self.data[self.columns] = self.data[self.columns].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
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

    # this() metodu stop_words() metodu ile aynı mantıkla çalışmakatadır. Ancak stop words yerine kendi listemiz içerisinde bulunan değerleri kaldırmaktadır.
    def this(self, data=None, columns=None, sentence=None, values=None):
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
                sentence = sentence.lower()
                sentence = [word for word in sentence.split() if word.lower() not in values]
                sentence = ' '.join(sentence)
                return sentence

    # more_space() metodu veri içerisindeki fazla boşlukların silinmesi için kullanılmaktadır.
    def more_space(self, data=None, columns=None, sentence=None):
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

    def sentence_analyze(self):
        if self.values is None:
            self.sentence = self.to_lowercase()
            self.sentence = self.https()
            self.sentence = self.email()
            self.sentence = self.hashtags()
            self.sentence = self.number()
            self.sentence = self.punctuation()
            self.sentence = self.stop_word()
            self.sentence = self.more_space()
            return self.sentence
        else:
            self.sentence = self.to_lowercase()
            self.sentence = self.https()
            self.sentence = self.email()
            self.sentence = self.hashtags()
            self.sentence = self.number()
            self.sentence = self.punctuation()
            self.sentence = self.stop_word()
            self.sentence = self.this()
            self.sentence = self.more_space()
            return self.sentence

    def dataframe_analyze(self):
        if self.values is None:
            self.data = self.to_lowercase()
            self.data = self.https()
            self.data = self.email()
            self.data = self.hashtags()
            self.data = self.number()
            self.data = self.punctuation()
            self.data = self.stop_word()
            self.data = self.more_space()
            return self.data
        else:
            self.data = self.to_lowercase()
            self.data = self.https()
            self.data = self.email()
            self.data = self.hashtags()
            self.data = self.number()
            self.data = self.punctuation()
            self.data = self.stop_word()
            self.data = self.this()
            self.data = self.more_space()
            return self.data