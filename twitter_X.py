import tweepy
import os
from dotenv import load_dotenv
import time

load_dotenv("token.env")

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
SPARE_TOKEN = os.getenv("SPARE_BEARER_TOKEN")

client = tweepy.Client(BEARER_TOKEN)
sclient = tweepy.Client(SPARE_TOKEN)

def getDetails(username):
    try:
        info = client.get_user(username=username, user_fields=["public_metrics"])
        data = info.data
        tweet = client.get_users_tweets(id=data.id, max_results=2)
        print("User Info:")
        print(f"Username: {data.username}")
        print(f"Name: {data.name}")
        print(f"ID: {data.id}")
        print(f"Bio: {data.description}")
        print(f"Location: {data.location}")
        print(f"Profile Image URL: {data.profile_image_url}")
        print(f"URL: {data.url}")
        print(f"Verified: {data.verified}")
        public = data.public_metrics
        if public:
            print(f"Followers Count: {public['followers_count']}")
            print(f"Following Count: {public['following_count']}")
            print(f"Tweet Count: {public['tweet_count']}")
            print(f"Listed Count: {public['listed_count']}")
        else:
            print("Public metrics not available")

        # Print tweet information
        if tweet.data:
            print("Tweet Info:")
            for t in tweet.data:
                print(f"Tweet ID: {t.id}")
                print(f"Tweet Text: {t.text}")
                print(f"Tweet Created At: {t.created_at}")
                print(f"Retweet Count: {t.public_metrics['retweet_count']}")
                print(f"Reply Count: {t.public_metrics['reply_count']}")
                print(f"Like Count: {t.public_metrics['like_count']}")
                hashtags = t.entities.get('hashtags', [])
                mentions = t.entities.get('mentions', [])
                urls = t.entities.get('urls', [])
                print(f"Hashtags: {hashtags}")
                print(f"Mentions: {mentions}")
                print(f"URLs: {urls}")
                
                # Geo location of the tweet (if available)
                if t.geo:
                    print(f"Geo: {t.geo}")
                else:
                    print("Geo: None")
                    
                print("-" * 30)
        else:
            print("No tweets found for this user.")
    except tweepy.TooManyRequests as e:
        reset_time = e.response.headers.get("x-rate-limit-reset")
        sleep = (int(reset_time)) - (int(time.time())) + 5
        print(f"Error: {e}, sleeping for {sleep} seconds")
        time.sleep(sleep)

# Example usage
getDetails("timesofindia")
