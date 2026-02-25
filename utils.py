from underthesea import word_tokenize

def load_stopwords(stopword_txtfile='vietnamese-stopwords-dash.txt'):
    with open(stopword_txtfile, "r", encoding="utf-8") as file:
        data = file.read()
        stopwords = set(data.split())
        return stopwords

VI_STOPWORDS = load_stopwords()

def vietnamese_tokenizer(text):
    tokenized = word_tokenize(text)
    filter_tokenized = [word for word in tokenized if word not in VI_STOPWORDS]
    return filter_tokenized


if __name__ == '__main__':
    text = 'Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò'
    print(vietnamese_tokenizer(text))