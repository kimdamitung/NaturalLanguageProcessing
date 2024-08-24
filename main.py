import numpy as np
import pandas as pd
import nltk
import math
from collections import Counter
from frozendict import frozendict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from typing import List, Dict

class TextClassificationModel:
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        self.df_train = df_train
        self.df_test = df_test
        self.freqs = None
        self.idf = None

    def tokenize_df(
        self, dfs: List[pd.DataFrame], keys=('text', 'keyword', 'location'),
        stemmer=True, preserve_case=True, reduce_len=False, strip_handles=True,
        use_stopwords=True
    ) -> List[List[str]]:
        tokenizer = nltk.TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len, strip_handles=strip_handles)
        porter = nltk.PorterStemmer()
        stopwords = set(nltk.corpus.stopwords.words('english') + ['nan'])

        output = []
        for df in dfs:
            for _, row in df.iterrows():
                tokens = [tokenizer.tokenize(str(row[key] or "")) for key in keys]
                tokens = [token for token_list in tokens for token in token_list]  # flatten
                if use_stopwords:
                    tokens = [token for token in tokens if token.lower() not in stopwords and len(token) >= 2]
                if stemmer:
                    tokens = [porter.stem(token) for token in tokens]
                output.append(tokens)
        return output

    def word_frequencies(self, df: pd.DataFrame, **kwargs) -> Dict[int, Counter]:
        tokens = {
            0: self.tokenize_df([df[df['target'] == 0]], **kwargs),
            1: self.tokenize_df([df[df['target'] == 1]], **kwargs)
        }
        freqs = {target: Counter([item for sublist in tokens[target] for item in sublist]) for target in [0, 1]}
        return freqs

    def inverse_document_frequency(self, tokens: List[str]) -> Counter:
        tokens = [token for sublist in tokens for token in sublist]  # flatten
        idf = {token: math.log(len(tokens) / count) for token, count in Counter(tokens).items()}
        return Counter(idf)

    def inverse_document_frequency_df(self, dfs: List[pd.DataFrame]) -> Counter:
        tokens = [self.tokenize_df([df]) for df in dfs]
        tokens = [token for sublist in tokens for token in sublist]  # flatten
        return self.inverse_document_frequency(tokens)

    def extract_features(self, df: pd.DataFrame, use_idf=True, use_log=True, **kwargs) -> np.array:
        tokens = self.tokenize_df([df], **kwargs)
        features = []
        for token_list in tokens:
            positive = sum(self.freqs[0].get(token, 0) * self.idf.get(token, 1) for token in token_list)
            negative = sum(self.freqs[1].get(token, 0) * self.idf.get(token, 1) for token in token_list)
            features.append([positive, negative])
        features = np.array(features)
        if use_log:
            features = np.log(features)
        return features

    def train(self, **kwargs):
        self.freqs = self.word_frequencies(self.df_train, **kwargs)
        self.idf = self.inverse_document_frequency_df([self.df_train, self.df_test])
        Y_train = self.df_train['target'].to_numpy()
        X_train = self.extract_features(self.df_train, **kwargs)
        model = LinearRegression().fit(X_train, Y_train)
        return model

    def predict(self, model, df: pd.DataFrame, **kwargs) -> np.array:
        X_test = self.extract_features(df, **kwargs)
        prediction = model.predict(X_test)
        return np.round(prediction).astype(int)

    def evaluate(self, splits=3, **kwargs) -> float:
        accuracy = 0.0
        for _ in range(splits):
            train, test = train_test_split(self.df_train, test_size=1/splits)
            model = self.train(**kwargs)
            prediction = self.predict(model, test, **kwargs)
            Y_test = test['target'].to_numpy()
            accuracy += np.sum(Y_test == prediction) / len(Y_test) / splits
        return accuracy

    def hyperparameter_search(self) -> Counter:
        results = Counter()
        jobs = []
        for stemmer, preserve_case, reduce_len, use_stopwords, use_idf, use_log in product([1, 0], repeat=6):
            def fn(stemmer, preserve_case, reduce_len, use_stopwords, use_idf, use_log):
                kwargs = {
                    "stemmer": stemmer, "preserve_case": preserve_case, "reduce_len": reduce_len,
                    "use_stopwords": use_stopwords, "use_idf": use_idf, "use_log": use_log
                }
                accuracy = self.evaluate(**kwargs)
                return (frozendict(kwargs), accuracy)
            jobs.append(delayed(fn)(stemmer, preserve_case, reduce_len, use_stopwords, use_idf, use_log))
        results = Counter(dict(Parallel(n_jobs=-1)(jobs)))
        return Counter(dict(results.most_common()))

    def submit_predictions(self, submission_file='submission.csv'):
        model = self.train()
        prediction = self.predict(model, self.df_test)
        submission = pd.DataFrame({"id": self.df_test.index, "target": prediction})
        submission.to_csv(submission_file, index=False)


# Usage
df_train = pd.read_csv('dataset/train.csv', index_col=0)
df_test = pd.read_csv('dataset/test.csv', index_col=0)
model = TextClassificationModel(df_train, df_test)
accuracy_results = model.hyperparameter_search()

for label, value in accuracy_results.items():
    print(f'{value:.5f} |', "  ".join(f"{k.split('_')[-1]} = {v}" for k, v in label.items()))

print('Train accuracy =', model.evaluate())
model.submit_predictions()
