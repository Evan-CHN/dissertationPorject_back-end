import re

from pandas import DataFrame


# 清洗文本数据
def processor(text):
    text = re.sub('<[^>]*>', "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    emotionicons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', " ", text.lower().strip()) + " ".join(emotionicons).replace('-', ''))
    return text


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    import nltk
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = processor(text)

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text


def text_process(csv_file):
    processed_text = DataFrame(columns=('index', 'review', 'sentiment'))
    for item in csv_file['review']:
        from nltk.corpus import stopwords
        meaningful_words = utils_preprocess_text(item, flg_stemm=True, flg_lemm=True,
                                                 lst_stopwords=stopwords.words('english'))
        index = csv_file[csv_file.review == item].index.tolist()[0]
        sentiment_value = 1 if csv_file['sentiment'][index] == 'positive' else 0
        processed_text = processed_text.append(
            {'index': index, 'review': meaningful_words,
             'sentiment': sentiment_value},
            ignore_index=True)
    return processed_text
