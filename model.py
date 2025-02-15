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
    #text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def lower(text):
    text = str(text).lower()
    return text

df['Tweet'] = df['Tweet'].apply(clean_text)
df['TweetLower'] = df['Tweet'].apply(lower)
df['Verified'] = df['Verified'].astype(int)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_text_embeddings = np.array(embedder.encode(df['Tweet'].tolist()))

df['Tweet_Length'] = df['Tweet'].apply(len)
df['Hashtag_Count'] = df['Tweet'].apply(lambda x: x.count('#'))
df['Uppercase_Count'] = df['Tweet'].apply(lambda x: sum(1 for c in x if c.isupper()))

X_numerical = df[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Tweet_Length', 'Hashtag_Count', 'Uppercase_Count']].astype(float).values
X = np.hstack((X_text_embeddings, X_numerical))
y = df['Bot Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

joblib.dump(best_model, "bot_detector.pkl")
joblib.dump(embedder, "text_embedder.pkl")

y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def predict_bot(tweet, retweet_count, mention_count, follower_count, verified):
    embedder = joblib.load("text_embedder.pkl")
    best_model = joblib.load("bot_detector.pkl")
    
    tweet_cleaned = clean_text(tweet)
    tweet_embedding = np.array(embedder.encode([tweet_cleaned]))
    features = np.hstack((tweet_embedding, [[retweet_count, mention_count, follower_count, int(verified), len(tweet), tweet.count('#'), sum(1 for c in tweet if c.isupper())]]))
    
    prediction = best_model.predict(features)
    return "Bot" if prediction[0] == 1 else "Human"

if __name__ == "__main__":
    example_tweet = "This is an automated message, please ignore."
    print("Prediction:", predict_bot(example_tweet, 10, 2, 1000, False))