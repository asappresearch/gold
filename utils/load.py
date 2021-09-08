import os, pdb, sys
import json
import re
import sqlite3
import random
import errno

import pickle as pkl
import numpy as np
import torch

from tqdm import tqdm as progress_bar
from transformers import BertTokenizer, RobertaTokenizer

def load_best_model(args, model, device):
  print('Loading best finetuned model ...')
  load_dir = model.load_dir
  filenames = [f for f in os.listdir(load_dir) if f.endswith('.pt')]
  top_score, top_filename = 0, ''

  for fname in filenames:
    if model.name == 'basic':
      re_str = r'aupr([0-9]{3})\.pt$'
    else:
      re_str = r'acc([0-9]{3})\.pt$'
    current_score = re.findall(re_str, fname)
    score = int(current_score[0]) if len(current_score) > 0 else 0
    parts = fname.split('_')
    source_dataset = parts[0][6:]

    if score > top_score: # and source_dataset == args.source_data:
      top_score = score
      top_filename = fname

  if len(top_filename) == 0:
    raise RuntimeError(f'No models were found in {load_dir}')
  ckpt_path = os.path.join(load_dir, top_filename)
  checkpoint = torch.load(ckpt_path, map_location='cpu')
  model.load_state_dict(checkpoint)
  model.eval()
  model.to(device)

  print(f'Loaded {ckpt_path} as best model')
  return model

def load_ontology(args):
  ont_path = os.path.join(args.input_dir, 'target', 'ontology.json')
  ontology = json.load(open(ont_path, 'r'))
  labels = ontology[args.task]["Intents"]
  return labels

def load_glove(device, size=300):
  if device.type == 'cpu':
    root_path = "/Users/derekchen"
  else:
    root_path = "/persist"
  path_name = ".embeddings/glove/"
  # file_name = "common_crawl_840:300.db"
  file_name = f"glove.6B.{size}d.txt"
  full_path = os.path.join(root_path, path_name, file_name)
  glove_embeddings = {}
  print(f'Loading {full_path} ...')
  with open(full_path, 'r') as file:
    for line in progress_bar(file, total=400000):
      row = line.split()
      token = row[0]
      glove_embeddings[token] = np.array( [float(x) for x in row[1:]] )
  return glove_embeddings

def load_data(args, datatype):
  if datatype == 'target':
    data_path = os.path.join(args.input_dir, datatype, f'{args.task}.json')
  elif datatype == 'source':
    data_path = os.path.join(args.input_dir, datatype, f'{args.source_data}.json')

  if os.path.isfile(data_path):
    data = json.load(open(data_path, 'r'))
  else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_path)
  return data

def load_tokenizer(args):
  # ontology = json.load(open('data/ontology.json', 'r'))
  special = { 'additional_special_tokens': ['<customer>', '<agent>', '<kb>']  }
  if args.model == 'bert':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  elif args.model == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
  tokenizer.add_special_tokens(special)
  return tokenizer
