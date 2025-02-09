import re
import jieba
import jieba.posseg as pseg
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


nltk.data.path.append('./.venv/nltk_data')  # 使用自定义下载路径


# 首次运行需要下载资源
# nltk.download('stopwords', download_dir='./.venv/nltk_data')
# nltk.download('punkt', download_dir='./.venv/nltk_data')
# nltk.download('punkt_tab', download_dir='./.venv/nltk_data')
# nltk.download('wordnet',download_dir='./.venv/nltk_data')
# nltk.download('averaged_perceptron_tagger_eng',download_dir='./.venv/nltk_data')

def load_custom_stopwords(stopwords_file):
    """从文件中加载自定义停用词"""
    with open(stopwords_file, "r", encoding="utf-8") as f:
        custom_stopwords = set(f.read().splitlines())
    return custom_stopwords

def pos_tagging(tokens):
    """词性标注"""
    return pos_tag(tokens)


def post_tagging_chinese(tokens):
    """中文词性标注"""
    text = ''.join(tokens)
    words = pseg.cut(text)
    return [(word, flag) for word, flag in words]

def clean_text(text):
    """文本清理：小写，去标点，去数字"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text


def tokenize_english(text):
    """英文分词"""
    tokens = word_tokenize(text)
    return tokens


def tokenize_chinese(text):
    """中文分词"""
    tokens = jieba.lcut(text)
    return tokens


def remove_stopwords(tokens, lang="english", custom_stopwords=None):
    """去掉停用词"""
    stop_words = set(stopwords.words(lang)) if lang == "english" else set()
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    return [word for word in tokens if word not in stop_words]


def lemmatize_words(tokens):
    """词形还原（仅适用于英文）"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]


if __name__ == "__main__":
    text_en = "Natural Language Processing (NLP) is exciting! It includes text processing and machine learning."
    text_cn = "自然语言处理是人工智能的一个重要方向，它涉及机器学习和深度学习"

    # 加载自定义停用词文件
    # custom_stopwords = load_custom_stopwords("./stopwords.txt")# 自定义的停用词文件

    # 英文处理
    cleaned_text_en = clean_text(text_en)
    tokens_en = tokenize_english(cleaned_text_en)
    # tokens_en = remove_stopwords(tokens_en,"english", custom_stopwords)
    tokens_en = remove_stopwords(tokens_en, "english")
    tokens_en = lemmatize_words(tokens_en)
    # 词性标注
    tokens_en_tagged = pos_tagging(tokens_en)

    cleaned_text_cn = clean_text(text_cn)
    tokens_cn = tokenize_chinese(cleaned_text_cn)
    # 词性标注
    tokens_cn_tagged =post_tagging_chinese(tokens_cn)

    print(tokens_en_tagged)
    print(tokens_cn_tagged)
