import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import joblib

nltk.download('punkt')
nltk.download('punkt_tab')

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data 

def remove_irrelevant_columns(data, features):
    return data.drop(features, axis = 1)

def check_duplicate_rows(data):
    return data.duplicated().sum()

def clean_text(text):
    # convert to lowercase
    text = text.lower()

    # remove digits
    text = re.sub(r"\d+", "", text)

    # remove punctuation and special characters
    text = re.sub(r"[^a-z\s]", "", text)

    return text 

def avg_sentence_length(text):
    sentences = sent_tokenize(str(text))
    if len(sentences) == 0:
        return 0
    words_per_sentence = [len(word_tokenize(s)) for s in sentences]
    return np.mean(words_per_sentence)

def vocab_diversity(text):
    words = word_tokenize(str(text).lower())

    if len(words) == 0:
        return 0
    return len(set(words))/len(words)

def plot_distribution(data, col1, col2, title, xlabel, ylabel):
    plt.figure(figsize = (10,5))
    sns.histplot(data, x = col1, bins = 30, hue = col2, kde = True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(title + ".png")


def TFIDF_vectorization(data):
    tfidf = TfidfVectorizer(max_features = 10000, 
                            ngram_range=(1, 2), 
                            stop_words = "english")
    return tfidf.fit_transform(data)

def visualize_classifiers(data, title):
    class_count = data.value_counts()
    plt.figure(figsize = (6, 4))
    sns.barplot(data = class_count)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.savefig(title + ".png")

def resampling_data(X, y):
    smote = SMOTE(k_neighbors = 1, random_state = 42)
    smt = SMOTETomek(smote = smote, random_state = 42)
    X_resampled, y_resampled = smt.fit_resample(X, y)
    return X_resampled, y_resampled

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test

def save_preprocessed_dataset(X_train, X_test, y_train, y_test, filepath):
    joblib.dump((X_train, X_test, y_train, y_test), filepath)



if __name__ == "__main__":
    # load the dataset
    essays_df = load_data("data/essays.csv")

    # read first five rows of the data 
    print(essays_df.head())

    # remove irrelevant columns
    essays_df = remove_irrelevant_columns(essays_df, ['id', 'prompt_id'])

    # check for duplicated rows
    print("\n Number of duplicated rows: {}".format(check_duplicate_rows(essays_df)))

    # calculate the number of words per text
    essays_df['word_count'] = essays_df['text'].apply(lambda x: len(word_tokenize(str(x))))
    # visualize the answer length distribution 
    plot_distribution(essays_df, "word_count", "generated", "Answer Length Distribution (Words)", "Number of Words", "Frequency")

    # calculate average sentence length per text (no. of words per sentence)
    essays_df['avg_sentence_length'] = essays_df['text'].apply(avg_sentence_length)
    # visualize the average sentence length distribution 
    plot_distribution(essays_df, "avg_sentence_length", "generated", "Average Sentence Length Distribution", "Words per Sentence", "Frequency")

    # calculate vocabulary diversity (no. of unique words/ total words per text)
    essays_df['vocab_diversity'] = essays_df['text'].apply(vocab_diversity)
    # visualize the vocabulary diversity distribution 
    plot_distribution(essays_df, "vocab_diversity", "generated", "Vocabulary Diversity Distribution", "Unique Words/ Total Words", "Frequency")

    # perform text cleaning for each rows
    essays_df['cleaned_text'] = essays_df["text"].apply(clean_text)

    # read first five rows of the data
    print(essays_df.head())

    # feature vectorization 
    X = TFIDF_vectorization(essays_df['cleaned_text'])
    y = essays_df["generated"]

    # visualize the number of classes before resampled  
    visualize_classifiers(essays_df["generated"], "Class Distribution (Before Resampling)")

    # resampled data
    X_resampled, y_resampled = resampling_data(X, y)

    # visualize the number of classes after resampled 
    visualize_classifiers(y_resampled, "Class Distribution (After Resampling)")

    # split the dataset
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)

    # save the preprocessed dataset
    save_preprocessed_dataset(X_train, X_test, y_train, y_test, "preprocessed_data.pkl")



    


