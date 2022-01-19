"""
This file generates the initial message features (please see Figure 1(b) and Section 3.2 of the paper for more details).
To leverage the semantics in the data, we generate document feature for each message,
which is calculated as an average of the pre-trained word embeddings of all the words in the message
We use the word embeddings pre-trained by en_core_web_lg, while other options, 
such as word embeddings pre-trained by BERT, are also applicable.
To leverage the temporal information in the data, we generate temporal feature for each message,
which is calculated by encoding the times-tamps: we convert each timestamp to OLE date, 
whose fractional and integral components form a 2-d vector.
The initial feature of a message is the concatenation of its document feature and temporal feature.
"""
import numpy as np
import pandas as pd
import en_core_web_lg
from datetime import datetime

load_path = '../datasets/Twitter/'
save_path = '../datasets/Twitter/'

# load dataset
p_part1 = load_path + '68841_tweets_multiclasses_filtered_0722_part1.npy'
p_part2 = load_path + '68841_tweets_multiclasses_filtered_0722_part2.npy'
df_np_part1 = np.load(p_part1, allow_pickle=True)
df_np_part2 = np.load(p_part2, allow_pickle=True)
df_np = np.concatenate((df_np_part1, df_np_part2), axis = 0)
print("Loaded data.")
df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
    "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
    "words", "filtered_words", "sampled_words"])
print("Data converted to dataframe.")
print(df.shape)
print(df.head(10))

# Calculate the embeddings of all the documents in the dataframe, 
# the embedding of each document is an average of the pre-trained embeddings of all the words in it
def documents_to_features(df):
    nlp = en_core_web_lg.load()
    features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector).values
    return np.stack(features, axis=0)

# encode one times-tamp
# t_str: a string of format '2012-10-11 07:19:34'
def extract_time_feature(t_str):
    t = datetime.fromisoformat(str(t_str))
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

# encode the times-tamps of all the messages in the dataframe
def df_to_t_features(df):
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    return t_features


d_features = documents_to_features(df)
print("Document features generated.")
t_features = df_to_t_features(df)
print("Time features generated.")
combined_features = np.concatenate((d_features, t_features), axis=1)
print("Concatenated document features and time features.")
np.save(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy', combined_features)
print("Initial features saved.")
combined_features = np.load(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy')
print("Initial features loaded.")
print(combined_features.shape)