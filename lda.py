
import joblib
import kagglehub
import os
from spacy import tokens
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from pathlib import Path
from gensim.matutils import Sparse2Corpus
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from pathlib import Path
import logging
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# path = kagglehub.dataset_download("jeet2016/us-financial-news-articles")
# print("Dataset Downloaded to:", path)
# print("Dataset Folders:", os.listdir(path))

# subfolder = os.listdir(path)[0]
# print("Subfolder:", subfolder)
# print(os.listdir(os.path.join(path, subfolder))[:10])

# path = "/Users/ws/.cache/kagglehub/datasets/jeet2016/us-financial-news-articles/versions/1"
# total_articles = 0

# for folder in os.listdir(path):
#     folder_path = os.path.join(path, folder)
#     if os.path.isdir(folder_path):
#         json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
#         count = len(json_files)
#         total_articles += count
#         print(f"{folder}: {count} articles")

# print("\nTotal articles in dataset:", total_articles)

# # -----------------------------------------------------------------------------------------------------------------

onepath = (
    "/Users/ws/.cache/kagglehub/datasets/jeet2016/"
    "us-financial-news-articles/versions/1/"
    "2018_03_112b52537b67659ad3609a234388c50a/"
    "news_0047188.json"
)
with open(onepath, "r", encoding="utf-8") as f:
    data = json.load(f)

# print(data)
print("Data Keys:", data.keys())
print("Data Text:", data["text"])

# # -----------------------------------------------------------------------------------------------------------------

# def load_all_articles(path):
#     all_texts = []
#     count = 0  
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith(".json"):
#                 with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#                     if isinstance(data, dict) and "text" in data:
#                         all_texts.append(data["text"])
#                 count += 1
#                 if count % 50000 == 0:
#                     print(f"{count} files processed...")
#     return all_texts

# all_texts = load_all_articles(path)
# print(f"Total articles loaded: {len(all_texts)}")

# data_path = Path("./data")
# with open(data_path / "all_texts.txt", "w", encoding="utf-8") as f:
#     for article in all_texts:
#         f.write(article.replace("\n", " ") + "\n")

# # -----------------------------------------------------------------------------------------------------------------

# with open(data_path / "all_texts.txt", "r", encoding="utf-8") as f:
#     all_texts = [line.strip() for line in f.readlines()]

# # -----------------------------------------------------------------------------------------------------------------

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
from spacy.lang.en.stop_words import STOP_WORDS
stop_words = STOP_WORDS
import re

# def clean_doc(doc, stop_words, allowed_pos=["NOUN", "ADJ", "VERB"]):
#     tokens = []
#     for t in doc:
#         lemma = t.lemma_.lower()
#         if any([
#             t.is_stop, t.is_digit, not t.is_alpha, t.is_punct, t.is_space, 
#             t.lemma_ == "-PRON-", len(lemma) <= 2, t.pos_ not in allowed_pos, 
#             re.fullmatch(r"(.)\1{2,}|(..+?)\1+", lemma)]):
#             continue
#         tokens.append(lemma)
#     return tokens

# # -----------------------------------------------------------------------------------------------------------------

# def preprocess(texts, nlp, stop_words):
#     cleaned_texts = []
#     for i, doc in enumerate(tqdm(nlp.pipe(texts, batch_size=1000))):
#         tokens = clean_doc(doc, stop_words)
#         cleaned_texts.append({
#             "id": i,
#             "tokens": tokens})
#     return cleaned_texts

# with open(data_path / "all_texts.txt", "r", encoding="utf-8") as f:
#     all_texts = [line.strip() for line in f.readlines()]

# cleaned_texts = preprocess(all_texts, nlp, stop_words)
# print("original cleaned_texts:", cleaned_texts[5])

# results_path = Path("./results")  
# with open(results_path / "cleaned_texts.jsonl", "w", encoding="utf-8") as f:
#     for doc in cleaned_texts:
#         f.write(json.dumps(doc) + "\n")

# -----------------------------------------------------------------------------------------------------------------

# cleaned_texts_path = Path("./results/cleaned_texts.jsonl")  
# cleaned_texts = []
# with open(cleaned_texts_path, "r", encoding="utf-8") as f:
#     for line in f:
#         cleaned_texts.append(json.loads(line))
# print("json cleaned_texts:", cleaned_texts[5])

# -----------------------------------------------------------------------------------------------------------------

# corpus = [" ".join(doc["tokens"]) for doc in cleaned_texts]
# print("corpus cleaned_texts:", corpus[5])
# with open(results_path / "corpus_texts.txt", "w", encoding="utf-8") as f:
#     for doc in corpus:
#         f.write(doc + "\n")

# file_path = results_path / "corpus_texts.txt"
# with open(file_path, "r", encoding="utf-8") as f:
#     corpus = [line.strip() for line in f]

# # -----------------------------------------------------------------------------------------------------------------

from collections import Counter

# article_length, token_count = [], Counter()

# for i, doc in enumerate(corpus, 1):
#     if i % 1e6 == 0:
#         print(i, end=' ', flush=True)
#     d = doc.lower().split()  # split into tokens
#     article_length.append(len(d))
#     token_count.update(d)

# # Summary statistics of article lengths
# length_stats = pd.Series(article_length).describe(percentiles=np.arange(.1, 1.0, .1))
# print(length_stats)

# # -----------------------------------------------------------------------------------------------------------------

# vectorizer = TfidfVectorizer(
#     tokenizer=str.split,
#     preprocessor=None,
#     token_pattern=None,
#     lowercase=True,
#     stop_words='english',
#     min_df=0.007,
#     max_df=0.3,
#     ngram_range=(1, 1),
#     binary=False
# )

# dtm = vectorizer.fit_transform(corpus)
# print("TF-IDF matrix shape:", dtm.shape)

# # -----------------------------------------------------------------------------------------------------------------

vectorizer = joblib.load("tfidf_vectorizer.pkl")
word2id = vectorizer.vocabulary_

# doc_freq = np.array((dtm > 0).sum(axis=0)).flatten()
# df_stats = pd.DataFrame({
#     "word_id": [word2id[w] for w in vectorizer.get_feature_names_out()],
#     "word": vectorizer.get_feature_names_out(),
#     "doc_frequency": doc_freq
# })
# print(df_stats.sort_values("doc_frequency", ascending=True).head(10))
# print(df_stats.sort_values("doc_frequency", ascending=False).head(10))

# # -----------------------------------------------------------------------------------------------------------------

import re
# def is_noise_pattern(word):
#     return bool(
#         re.fullmatch(r"(.)\1{2,}", word) or    
#         re.fullmatch(r"(..+?)\1+", word)       
#     )

# df_stats["is_noise"] = df_stats["word"].apply(is_noise_pattern)
# repeated_letter = df_stats[df_stats["is_noise"]]
# print("Number of repeated_letter:", len(repeated_letter))
# print("Total repeated words:", repeated_letter["doc_frequency"].sum())
# print(repeated_letter)

# # -----------------------------------------------------------------------------------------------------------------

# gensim_corpus = Sparse2Corpus(dtm, documents_columns=False)

# # -----------------------------------------------------------------------------------------------------------------

id2word = {idx: word for word, idx in word2id.items()}

# # -----------------------------------------------------------------------------------------------------------------

# logging.basicConfig(filename='./models/gensim.log',
#                     format="%(asctime)s:%(levelname)s:%(message)s",
#                     level=logging.DEBUG)
# logging.root.level = logging.DEBUG

model_path = Path("models")  

num_topics = [5, 10, 15, 20]
# lda_models = {}

# # for topics in num_topics:
# prog_bar = tqdm(num_topics, desc="Training LDA models", total=len(num_topics))
# for topics in prog_bar:
#     prog_bar.set_description(f"Training LDA with {topics} topics")
#     # print(f"\nTraining LDA with {topics} topics...")
#     lda_model = LdaModel(
#             corpus=gensim_corpus,
#             id2word=id2word,
#             num_topics=topics,
#             chunksize=1000,
#             update_every=1,
#             alpha='auto',
#             eta='auto',
#             decay=0.5,
#             offset=1.0,
#             eval_every=1,
#             passes=10,
#             iterations=50,
#             gamma_threshold=0.001,
#             minimum_probability=0.01,
#             minimum_phi_value=0.01,
#             random_state=42)
    
#     # Save model
#     lda_model.save(f"{model_path}/lda_{topics}_model")
#     lda_models[topics] = lda_model
    
#     # Print top words
#     for idx, topic in lda_model.show_topics(formatted=False):
#         words = [w for w, _ in topic]
#         print(f"Topic {idx}: {', '.join(words[:10])}")

# # -----------------------------------------------------------------------------------------------------------------
######### Calculate coherence, perplexity

from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# dictionary = Dictionary()
# dictionary.id2token = id2word
# dictionary.token2id = {v: k for k, v in id2word.items()}


# # Path to save metrics
# metrics_path = Path("./results/lda_metrics.json") 
# pyldavis_path = Path("./pyldavis")

# lda_metrics = {}

# for topics in num_topics:
#     model = LdaModel.load(f"models/lda_{topics}_model")
    
#     # Compute Coherence
#     coherence = CoherenceModel(
#         model=model, texts=[doc["tokens"] for doc in cleaned_texts], 
#         dictionary=dictionary, coherence='c_v', processes=1).get_coherence()
    
#     # Compute Perplexity
#     perplexity = 2 ** (-model.log_perplexity(gensim_corpus))
#     print(f"Topics: {topics}, Coherence: {coherence}, Perplexity: {perplexity}")
    
#     vis = gensimvis.prepare(model, gensim_corpus, dictionary)
#     pyLDAvis.save_html(vis, str(pyldavis_path / f'lda_{topics}.html'))

#     # Save metrics
#     lda_metrics[topics] = {
# 	    "topic": topics,
#         "coherence": coherence,
#         "perplexity": perplexity
#     }

# # Save metrics to JSON
# with open(metrics_path, "w", encoding="utf-8") as f:
#     json.dump(lda_metrics, f, indent=4)

# # -----------------------------------------------------------------------------------------------------------------

# # Selected model
model10 = LdaModel.load("./models/lda_10_model")

# -----------------------------------------------------------------------------------------------------------------
##### Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# # 1. Raw texts
# all_tokens = " ".join(all_texts).split()
# top_raw = Counter(all_tokens).most_common(20)
# article_length, token_count = [], Counter()

# for i, doc in enumerate(all_texts, 1):
#     if i % 1e6 == 0:
#         print(i, end=' ', flush=True)
#     d = doc.lower().split()
#     article_length.append(len(d))
#     token_count.update(d)
# fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

# # Top 25 tokens (excluding stop words)
# (pd.DataFrame(token_count.most_common(), columns=['token', 'count'])
#  .pipe(lambda x: x[~x.token.str.lower().isin(stop_words)])
#  .set_index('token') .squeeze() .iloc[:25] .sort_values()
#  .plot .barh(ax=axes[0], title='Raw Text: Most Frequent Tokens'))

# # Article length distribution
# sns.boxenplot(x=pd.Series(article_length), ax=axes[1])
# axes[1].set_xscale('log')
# axes[1].set_xlabel('Word Count (log scale)')
# axes[1].set_title('Raw Text: Article Length Distribution')

# sns.despine()
# fig.tight_layout()
# fig.savefig("demo/raw_text.png", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()

# # 2. Cleaned texts
# cleaned_tokens = [token for doc in cleaned_texts for token in doc["tokens"]]
# top_cleaned = Counter(cleaned_tokens).most_common(20)
# article_length, token_count = [], Counter()

# for i, doc in enumerate(cleaned_texts, 1):
#     if i % 1e6 == 0:
#         print(i, end=' ', flush=True)
#     d = doc["tokens"]
#     article_length.append(len(d))
#     token_count.update(d)

# fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

# # Top 25 tokens (cleaned)
# (pd.DataFrame(token_count.most_common(), columns=['token', 'count'])
#  .set_index('token')
#  .squeeze()
#  .iloc[:25]
#  .sort_values()
#  .plot
#  .barh(ax=axes[0], title='Cleaned Text: Most Frequent Tokens'))

# # Article length distribution
# sns.boxenplot(x=pd.Series(article_length), ax=axes[1])
# axes[1].set_xscale('log')
# axes[1].set_xlabel('Word Count (log scale)')
# axes[1].set_title('Cleaned Text: Article Length Distribution')

# sns.despine()
# fig.tight_layout()
# fig.savefig("demo/cleaned_text.png", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()

# # 3. TF-IDF df_stats
# top_dfstats = df_stats.sort_values("doc_frequency", ascending=False).head(25)
# plt.figure(figsize=(12,8))
# sns.barplot(
#     x="doc_frequency",
#     y="word",
#     data=top_dfstats)
# plt.xlabel("Number of Documents")
# plt.ylabel("Word")
# plt.title("Top 25 TF-IDF Words")
# plt.tight_layout()
# plt.show()
# plt.close()

# # -----------------------------------------------------------------------------------------------------------------
# #### Heatmap

# num_topics = model10.num_topics
# n_top_words = 10
# topic_words = []
# topic_probs = []

# for topic_id in range(num_topics):
#     words_probs = model10.show_topic(topic_id, topn=n_top_words)

#     words = [w for w, p in words_probs]
#     probs = [p for w, p in words_probs]
#     topic_words.append(words)
#     topic_probs.append(probs)

# prob_matrix = np.array(topic_probs).T
# word_matrix = np.array(topic_words).T

# topic_labels = {
#     0: "Finance reports",
#     1: "Corporate Info",
#     2: "Markets & commodities",
#     3: "Legal",
#     4: "Sports",
#     5: "Investor statements",
#     6: "Geopolitics",
#     7: "Accounting",
#     8: "Business deals",
#     9: "Healthcare"
# }

# heatmap_df = pd.DataFrame(prob_matrix, columns=[f"{i} - {topic_labels[i]}" for i in range(num_topics)])

# plt.figure(figsize=(16,7))

# sns.heatmap(
#     heatmap_df, annot=word_matrix, fmt="", cmap="Blues",
#     cbar=True, linewidths=0.5, linecolor="lightgray"
# )

# plt.xlabel("Topic interpretation")
# plt.ylabel("Top words rank")
# plt.title("Top words per topic (darker = higher probability)")

# plt.xticks(rotation=40, ha="right")  
# plt.tight_layout()

# plt.savefig("demo/heatmap.png", dpi=150)
# plt.show()
# plt.close()

# # -----------------------------------------------------------------------------------------------------------------
from gensim.corpora import Dictionary

modelid2word = {word: idx for idx, word in (model10.id2word).items()}

# print("word2id:")
# print(list(word2id.items())[:1])

# print("modelid2word:")
# print(list(modelid2word.items())[:1])

# print(word2id == modelid2word)

# # -----------------------------------------------------------------------------------------------------------------
#### TOPIC DISTRIBUTION FUNCTION

# def get_doc_topic_by_id(doc_id, jsonl_path, lda_model):

#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line in f:
#             doc = json.loads(line)
#             if doc["id"] == doc_id:
#                 tokens = doc["tokens"]
#                 break
#         else:
#             raise ValueError(f"Document with id {doc_id} not found.")
    
#     counts = Counter(tokens)
#     bow_vector = [(modelid2word[w], c) for w, c in counts.items() if w in modelid2word]
#     topic_probs = lda_model.get_document_topics(bow_vector, minimum_probability=0.0)
#     topic_probs = sorted(topic_probs, key=lambda x: x[1], reverse=True)

#     return topic_probs

# jsonl_path = Path("./results/cleaned_texts.jsonl")

# ### get topic distribution per id
# doc_id = 17
# topics = get_doc_topic_by_id(doc_id, jsonl_path, model10)
# topics_line = ", ".join([f"Topic {topic_id}: {prob:.4f}" for topic_id, prob in topics[:10]])
# print(f"Doc {doc_id} top topics → {topics_line}")

# ### get topic distribution multiple ids
# doc_ids = [0, 10, 999]
# for doc_id in doc_ids:
#     topics = get_doc_topic_by_id(doc_id, jsonl_path, model10)
#     topics_line = ", ".join([f"Topic {tid}: {p:.4f}" for tid, p in topics[:10]])
#     print(f"Doc {doc_id} top topics → {topics_line}")

# # -----------------------------------------------------------------------------------------------------------------

# all_topics = {}

# for doc in tqdm(cleaned_texts, desc="Processing documents"):
#     doc_id = int(doc["id"])
#     tokens = doc["tokens"]

#     counts = Counter(tokens)
#     bow_vector = [(modelid2word[w], c) for w, c in counts.items() if w in modelid2word]

#     topic_probs = model10.get_document_topics(bow_vector, minimum_probability=0.0)
#     topic_probs = [(int(tid), float(prob)) for tid, prob in topic_probs]
#     topic_probs = sorted(topic_probs, key=lambda x: x[1], reverse=True)[:10]

#     all_topics[doc_id] = topic_probs  

# with open(results_path /"topics.json", "w", encoding="utf-8") as f:
#     json.dump(all_topics, f)

# # -----------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------------------------------------------

#### FOR HTML DEMO

# demo_path = Path("./demo")

# with open("results/topics.json", "r", encoding="utf-8") as f:
#     all_topics = json.load(f)

# # sample 50k
# demo_topics = {k: all_topics[k] for k in map(str, range(10000)) if k in all_topics}

# # save in demo folder
# with open("demo_topics.json", "w", encoding="utf-8") as f:
#     json.dump(demo_topics, f)

# # -----------------------------------------------------------------------------------------------------------------

# # create all_text.json

# txt_file = Path("./data/all_texts.txt")  

# all_texts = {}
# with open(txt_file, "r", encoding="utf-8") as f:
#     for idx, line in enumerate(f, 0):  # start from 1 if you want to match your numbering
#         all_texts[str(idx)] = line.strip()

# all_texts = dict(list(all_texts.items())[:10000])

# with open("demo_texts.json", "w", encoding="utf-8") as f:
#     json.dump(all_texts, f, ensure_ascii=False, indent=2)

# # -----------------------------------------------------------------------------------------------------------------

# corpus_path = Path("./results/corpus_texts.txt")
# demo_corpus = Path("./demo_corpus.txt")

# # Read first 50,000 lines and write to a new file
# with open(corpus_path, "r", encoding="utf-8") as f_in, \
#      open(demo_corpus, "w", encoding="utf-8") as f_out:
    
#     for i, line in enumerate(f_in):
#         if i >= 10000:
#             break
        # f_out.write(line)

# jsonl_path = Path("./results/cleaned_texts.jsonl")
demo_cleaned = Path("./demo_cleaned.jsonl")

# with open(jsonl_path, "r", encoding="utf-8") as f_in, \
#      open(demo_cleaned, "w", encoding="utf-8") as f_out:
    
#     for i, line in enumerate(f_in):
#         if i >= 10000:
#             break
#         f_out.write(line)

# # -----------------------------------------------------------------------------------------------------------------

def demo_get_topic(doc_id, jsonl_path, lda_model):

    with open(demo_cleaned, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["id"] == doc_id:
                tokens = doc["tokens"]
                break
        else:
            raise ValueError(f"Document with id {doc_id} not found.")
    
    counts = Counter(tokens)
    bow_vector = [(modelid2word[w], c) for w, c in counts.items() if w in modelid2word]
    topic_probs = lda_model.get_document_topics(bow_vector, minimum_probability=0.0)
    topic_probs = sorted(topic_probs, key=lambda x: x[1], reverse=True)

    return topic_probs

# jsonl_path = Path("./demo_cleaned.jsonl")

### get topic distribution per id
doc_id = 17
topics = demo_get_topic(doc_id, demo_cleaned, model10)
topics_line = ", ".join([f"Topic {topic_id}: {prob:.4f}" for topic_id, prob in topics[:10]])
print(f"Doc {doc_id} top topics → {topics_line}")

### get topic distribution multiple ids
doc_ids = [0, 10, 999]
for doc_id in doc_ids:
    topics = demo_get_topic(doc_id, demo_cleaned, model10)
    topics_line = ", ".join([f"Topic {tid}: {p:.4f}" for tid, p in topics[:10]])
    print(f"Doc {doc_id} top topics → {topics_line}")

# # -----------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------------------------------------------
































