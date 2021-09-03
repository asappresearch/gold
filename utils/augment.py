import os, pdb, sys
import pandas as pd
import json
import numpy as np
import random
import torch

from utils.load import load_glove
from tqdm import tqdm as progress_bar

from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from assets.static_vars import device

def cache_util(source, technique, embedder, utterances):
  save_path = f"assets/cache/source/{source}_{technique}s.npy"  
  if os.path.exists(save_path):
    print(f"Using cached source embeddings from {save_path}")
    embeddings = np.load(open(save_path, 'rb'))
  else:
    print(f"Encoding source and saving to {save_path}")
    embeddings = embedder(utterances)
    np.save(open(save_path, 'wb'), embeddings)
  return embeddings

def embed_by_paraphrase(utterances, source):
  model = SentenceTransformer('paraphrase-distilroberta-base-v1')
  def embedder(samples):
    with torch.no_grad():
      embeddings = model.encode(samples)
    return embeddings

  embeddings = cache_util(source, 'paraphrase', embedder, utterances)
  return embeddings, embedder

def embed_by_robin(utterances, source):
  tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
  model = RobertaModel.from_pretrained('roberta-large')

  def embedder(samples):
    embeddings = []

    for sample in progress_bar(samples, total=len(samples)):
      tokens = tokenizer(sample, return_tensors="pt")
      with torch.no_grad():
        outputs = model(**tokens, return_dict=True)  # dict contains last_hidden_state and pooler_output
      embeddings.append(outputs['pooler_output'])
    # returns tensor of shape (num_samples, hidden_dim)
    embed_tensor = torch.cat(embeddings)
    return embed_tensor.detach().cpu().numpy()

  embeddings = cache_util(source, 'robin', embedder, utterances)  
  return embeddings, embedder

def embed_by_model(utterances, source):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')

  def embedder(samples):
    embeddings = []

    for sample in progress_bar(samples, total=len(samples)):
      tokens = tokenizer(sample, return_tensors="pt")
      with torch.no_grad():
        outputs = model(**tokens, return_dict=True)  # dict contains last_hidden_state and pooler_output
      embeddings.append(outputs['pooler_output'])
    # returns tensor of shape (num_samples, hidden_dim)
    embed_tensor = torch.cat(embeddings)
    return embed_tensor.detach().cpu().numpy()

  embeddings = cache_util(source, 'encoding', embedder, utterances)
  return embeddings, embedder

def embed_by_length(utterances):

  def embedder(samples):
    return [len(sample) for sample in samples]

  embeddings = embedder(utterances)
  return embeddings, embedder

def embed_by_bow(utterances, glove_map=None):
  if glove_map is None:
    glove_map = load_glove(device, size=300)

  def embedder(samples):
    vectors = []
    for sample in samples:
      sample = sample.replace('?',' ?').replace('.',' .').replace(',',' ,').replace('!', ' !')
      words = sample.lower().split()
      
      bag = []
      for word in words:
        try:
          bag.append(glove_map[word])
        except(KeyError):
          continue

      if len(bag) == 0:
        default = glove_map['empty']
        bag.append(default)
      vectors.append( np.sum(np.array(bag), axis=0) )
    return np.stack(vectors)

  embeddings = embedder(utterances)
  return embeddings, embedder

def embed_by_tfidf(utterances):
  stops = set(ENGLISH_STOP_WORDS)
  vectorizer = TfidfVectorizer(max_df=0.95, lowercase=True, stop_words=stops, max_features=7000)
  # embeddings is a (batch_size, max_features) numpy matrix
  vectors = vectorizer.fit_transform(utterances)
  embeddings = vectors.toarray()
  
  def embedder(samples):
    vectors = vectorizer.transform(samples)   # 100 x hidden_dim
    embeds = vectors.toarray()
    return embeds

  return embeddings, embedder



