import pandas as pd
import numpy as np
import re
import string
import joblib
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('bot_detection_data.csv')

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def lower(text):
    text = str(text).lower()
    return text

df['Tweet'] = df['Tweet'].apply(clean_text)
df['TweetLower'] = df['Tweet'].apply(lower)
print(df['Tweet'],df['TweetLower'])