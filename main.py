import pandas as pd

def preprocess_text(df, text_column):
    """
    Tiền xử lý văn bản: chuyển đổi thành chữ thường, loại bỏ ký tự đặc biệt và số.
    """
    df['cleaned_text'] = df[text_column].str.lower().str.strip()
    df['cleaned_text'] = df['cleaned_text'].str.replace('[^\w\s]', '', regex=True)
    df['cleaned_text'] = df['cleaned_text'].str.replace('\d+', '', regex=True)
    df['cleaned_text'] = df['cleaned_text'].str.replace('\s+', ' ', regex=True)
    return df

def tokenize_text(df, cleaned_text_column):
    """
    Tách văn bản thành các tokens (từ).
    """
    df['tokens'] = df[cleaned_text_column].str.split()
    return df

def calculate_word_frequency(df, tokens_column):
    """
    Tính tần suất xuất hiện của từng từ trong dữ liệu.
    """
    word_freq = pd.Series([word for tokens in df[tokens_column] for word in tokens]).value_counts()
    return word_freq

def extract_keywords(df, tokens_column, word_freq, min_freq=5):
    """
    Trích xuất từ khóa dựa trên tần suất từ.
    """
    keywords = word_freq[word_freq >= min_freq].index.tolist()
    df['keywords'] = df[tokens_column].apply(lambda tokens: [word for word in tokens if word in keywords])
    return df

def extract_important_words(df, tokens_column, min_length=4):
    """
    Trích xuất các từ quan trọng dựa trên chiều dài từ.
    """
    df['important_words'] = df[tokens_column].apply(lambda tokens: [word for word in tokens if len(word) > min_length])
    return df

def create_word_pairs(df, tokens_column):
    """
    Tạo các cặp từ từ các tokens.
    """
    df['word_pairs'] = df[tokens_column].apply(lambda tokens: [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)])
    return df

def predict_labels(df, keywords_column, target_word='disaster'):
    """
    Dự đoán nhãn dựa trên sự xuất hiện của từ khóa cụ thể.
    """
    df['predicted_label'] = df[keywords_column].apply(lambda x: 1 if target_word in x else 0)
    return df

def nlp_pipeline(df, text_column):
    """
    Quy trình xử lý NLP hoàn chỉnh từ tiền xử lý đến dự đoán nhãn.
    """
    df = preprocess_text(df, text_column)
    df = tokenize_text(df, 'cleaned_text')
    word_freq = calculate_word_frequency(df, 'tokens')
    df = extract_keywords(df, 'tokens', word_freq)
    df = extract_important_words(df, 'tokens')
    df = create_word_pairs(df, 'tokens')
    df = predict_labels(df, 'keywords')
    return df

if __name__ == "__main__":
    # Đọc dữ liệu từ file CSV
    train_df = pd.read_csv('dataset/train.csv')

    # Áp dụng quy trình xử lý NLP
    result_df = nlp_pipeline(train_df, 'text')

    # Lưu kết quả
    result_df[['id', 'predicted_label']].to_csv('output.csv', index=False)
