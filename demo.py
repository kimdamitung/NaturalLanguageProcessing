import numpy  as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import pydash
import math
import os
import itertools

from pydash import flatten, flatten_deep
from collections import Counter, OrderedDict
from frozendict import frozendict
from humanize import intcomma
from operator import itemgetter
from typing import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from itertools import product, combinations
from joblib import Parallel, delayed
df_train = pd.read_csv('dataset/train.csv', index_col=0)
df_test  = pd.read_csv('dataset/test.csv', index_col=0)
def tokenize_df(
    dfs: List[pd.DataFrame], 
    keys          = ('text', 'keyword', 'location'), 
    stemmer       = True, 
    preserve_case = True, 
    reduce_len    = False, 
    strip_handles = True,
    use_stopwords = True,
    **kwargs,
) -> List[List[str]]:
    # tokenizer = nltk.TweetTokenizer(preserve_case=True,  reduce_len=False, strip_handles=False)  # defaults 
    tokenizer = nltk.TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len, strip_handles=strip_handles) 
    porter    = nltk.PorterStemmer()
    stopwords = set(nltk.corpus.stopwords.words('english') + [ 'nan' ])

    output    = []
    for df in flatten([ dfs ]):
        for index, row in df.iterrows():
            tokens = flatten([
                tokenizer.tokenize(str(row[key] or ""))
                for key in keys    
            ])
            if use_stopwords:
                tokens = [ 
                    token 
                    for token in tokens 
                    if token.lower() not in stopwords
                    and len(token) >= 2
                ]                
            if stemmer:
                tokens = [ 
                    porter.stem(token) 
                    for token in tokens 
                ]
            output.append(tokens)

    return output


def word_frequencies(df, **kwargs) -> Dict[int, Counter]:
    tokens = {
        0: flatten(tokenize_df( df[df['target'] == 0], **kwargs )),
        1: flatten(tokenize_df( df[df['target'] == 1], **kwargs )),
    }
    freqs = { 
        target: Counter(dict(Counter(tokens[target]).most_common())) 
        for target in [0, 1]
    }  # sort and cast
    return freqs

freqs = word_frequencies(df_train)
print('freqs[0]', len(freqs[0]), freqs[0].most_common(10))
print('freqs[1]', len(freqs[1]), freqs[1].most_common(10))

def inverse_document_frequency( tokens: List[str] ) -> Counter:
    tokens = flatten_deep(tokens)
    idf = {
        token: math.log( len(tokens) / count ) 
        for token, count in Counter(tokens).items()
    }
    idf = Counter(dict(Counter(idf).most_common()))  # sort and cast
    return idf

def inverse_document_frequency_df( dfs ) -> Counter:
    tokens = flatten_deep([ tokenize_df(df) for df in flatten([ dfs ]) ])
    return inverse_document_frequency(tokens)

idf = inverse_document_frequency_df([ df_train, df_test ])
list(reversed(idf.most_common()))[:20]
def extract_features(df, freqs, use_idf=True, use_log=True, **kwargs) -> np.array:
    features = []
    tokens   = tokenize_df(df, **kwargs)
    for n in range(len(tokens)):
        bias     = 1  # bias term is implict when using sklearn
        positive = 1
        negative = 1        
        for token in tokens[n]:
            if use_idf:
                positive += freqs[0].get(token, 0) * idf.get(token, 1) 
                negative += freqs[1].get(token, 0) * idf.get(token, 1)
            else:
                positive += freqs[0].get(token, 0) 
                negative += freqs[1].get(token, 0) 
        features.append([ positive, negative ])  

    features = np.array(features)   # accuracy = 0.7166688559043741
    if use_log:
        features = np.log(features) # accuracy = 0.7136477078681204
    return features


Y_train = df_train['target'].to_numpy()
X_train = extract_features(df_train, freqs)
X_test  = extract_features(df_test,  freqs)

print('df_train', df_train.shape)
print('df_test ', df_test.shape)
print('Y_train ', Y_train.shape)
print('X_train ', X_train.shape)
print('X_test  ', X_test.shape)
print(X_test[:5])
def predict_df(df_train, df_test, **kwargs):
    freqs   = word_frequencies(df_train, **kwargs)

    Y_train = df_train['target'].to_numpy()
    X_train = extract_features(df_train, freqs, **kwargs)
    X_test  = extract_features(df_test,  freqs, **kwargs) if df_train is not df_test else X_train

    model      = LinearRegression().fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction = np.round(prediction).astype(int)
    return prediction


def get_train_f1_score(splits=3, **kwargs):
    """ K-Fold Split F1-score """
    f1 = 0.0
    for _ in range(splits):
        train, test = train_test_split(df_train, test_size=1/splits)      
        prediction  = predict_df(train, test, **kwargs)
        Y_train     = test['target'].to_numpy()
        f1         += f1_score(Y_train, prediction, average='weighted') / splits  # Sử dụng trung bình trọng số
    return f1
    
    
def train_f1_score_hyperparameter_search():
    results = Counter()
    jobs    = []
    
    for keys in [('text', 'keyword', 'location')]: 
        strip_handles = 1  
        for stemmer, preserve_case, reduce_len, use_stopwords, use_idf, use_log in product([1,0],[1,0],[1,0],[1,0],[1,0],[1,0]):
            def fn(keys, stemmer, preserve_case, reduce_len, strip_handles, use_stopwords, use_idf, use_log):
                kwargs = {
                    "stemmer":        stemmer,          
                    "preserve_case":  preserve_case, 
                    "reduce_len":     reduce_len, 
                    "use_stopwords":  use_stopwords,    
                    "use_idf":        use_idf,          
                    "use_log":        use_log,          
                }
                label = frozendict({**kwargs})
                f1 = get_train_f1_score(**kwargs)  # tính F1-score cho từng tham số
                return (label, f1)
            
            jobs.append(delayed(fn)(keys, stemmer, preserve_case, reduce_len, strip_handles, use_stopwords, use_idf, use_log))
            
    results = Counter(dict(Parallel(-1)(jobs)))
    results = Counter(dict(results.most_common())) 
    return results

results = train_f1_score_hyperparameter_search()
for label, value in results.items():
    print(f'{value:.5f} |', "  ".join(f"{k.split('_')[-1]} = {v}" for k,v in label.items()))  # in kết quả F1-score
print('train_f1_score = ', get_train_f1_score())
df_submission = pd.DataFrame({
    "id":     df_test.index,
    "target": predict_df(df_train, df_test)
})
df_submission.to_csv('submission.csv', index=False)