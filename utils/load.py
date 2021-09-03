import os, pdb, sys
import json
import re
import sqlite3
import random
import errno
import glob

import pickle as pkl
import numpy as np
import torch

from tqdm import tqdm as progress_bar
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, BartTokenizer
from components.models import BaseModel

def load_best_model(args, model, device):
  print('Loading best finetuned model ...')
  load_dir = model.load_dir
  filenames = [f for f in os.listdir(load_dir) if f.endswith('.pt')]
  top_score, top_filename = 0, ''

  for fname in filenames:
    if model.name == 'basic':
      re_str = r'fs([0-9]{3})\.pt$'
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
  target_path = os.path.join(args.input_dir, args.task, 'ontology.json')
  ontology = json.load(open(target_path, 'r'))
  labels = ontology["Intents"]
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
      
  # db = sqlite3.connect(full_path)
  # c = db.cursor()
  # c.execute('create table if not exists embeddings(word text primary key, emb blob)')
  # db.commit()
  '''
  ontology = json.load(open('data/ontology.json', 'r'))
  glove_cache = {}

  for piece in ['topic', 'dialogue_act']:
    skip_embed = list(glove_embeddings['skip'])
    glove_cache[piece] = {'<skip>': skip_embed }

    for item in ontology[piece]:
      if item == 'acknow':
        glove_cache[piece][item] = list(glove_embeddings['acknowledge'])
      elif '_' in item:
        glove_arr = np.zeros(100)
        for token in item.split('_'):
          glove_arr += glove_embeddings[token]
        glove_cache[piece][item] = list(glove_arr)
      else:
        glove_cache[piece][item] = list(glove_embeddings[item])
  json.dump(glove_cache, open('data/glove_cache.json', 'w'))
  sys.exit()
  '''
  return glove_embeddings

def load_augmentations(args):
  """ It is assumed at this point that
  1) the original pickled train data has been cached
  2a) for extraction, that data was used to fine-tune a model to help identify augments
  2b) for paraphrasing, some of that data was used as a seed for generating augments
  3) we now what to load the augment data which we blindly label as OOS """
  augmentations = {'extracted': [], 'paraphrased': [], 'swapped': []}

  if args.version == 'both':
    extract_file = f'normal_thresh{args.sample_size}_extracted.pkl'
    extract_path = os.path.join(args.output_dir, args.task, 'embedding', extract_file)
    extracted = pkl.load( open( extract_path, 'rb' ) )
    print(f"Loaded {len(extracted)} extracted data from", extract_path)
    augmentations['extracted'] = extracted

  if args.version == 'extract':
    extract_file = f'normal_thresh{args.augment_level}_extracted.pkl'
    extract_path = os.path.join(args.output_dir, args.task, args.augment_type, extract_file)
    extracted = pkl.load( open( extract_path, 'rb' ) )
    print(f"Loaded {len(extracted)} extracted data from", extract_path)
    augmentations['extracted'] = extracted

  if args.version == 'swap' or args.version == 'both':
    swap_method, data_source = args.augment_type.split('#')
    swap_file = f'{swap_method}_{args.augment_level}_{data_source}_{args.seed}.json'
    swap_path = os.path.join(args.input_dir, 'swap', 'anywhere', swap_file)
    swapped = json.load( open( swap_path, 'r' ) )
    print(f"Loaded {len(swapped)} swapped data from", swap_path)
    augmentations['swapped'] = swapped

  if args.version == 'paraphrase':
    paraphrase_file = f'{args.task}_{int(args.augment_level)}_{args.augment_type}.json'
    paraphrase_path = os.path.join(args.input_dir, 'paraphrase', paraphrase_file)
    paraphrased = json.load( open( paraphrase_path, 'r' ) )
    print(f"Loaded {len(paraphrased)} paraphrased data from", paraphrase_path)
    augmentations['paraphrased'] = paraphrased

  return augmentations

def load_data(args):
  source_data, target_data = [], []
  if args.ignore_cache:
    print("Skipping data loading")
    return source_data, target_data

  data_path = os.path.join(args.input_dir, 'cache', args.model, f'{args.task}.pkl')
  if os.path.isfile(data_path):
    target_data = pkl.load(open(data_path, 'rb'))
  else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_path)

  if args.version == 'augment':
    source_data = load_source(args, target_data, args.source_data)
    print(f"Loaded {len(source_data)} source examples from {args.source_data}")
  return source_data, target_data

def load_source(args, target_data, source):
  if source == 'US':
    source_ctx = [exp.context for exp in target_data['dev'] if exp.oos_label == 1]
    source_ctx.extend([exp.context for exp in target_data['train']])
    source_data = source_ctx if args.task == 'rostd' else [sc[-1] for sc in source_ctx]
    '''
    source_data = [exp for exp in target_data['dev'] if exp.oos_label == 1]
    source_data.extend(target_data['train'])
    '''
  else:
    source_path = os.path.join(args.input_dir, args.version, 'candidates', f'{source}.json')
    source_data = json.load(open(source_path, 'r'))
    if source == 'OOO':
      source_data = source_data['clinc']   # file also contains 'rostd' data, which we ignore

  return source_data

def load_tokenizer(args):
  # ontology = json.load(open('data/ontology.json', 'r'))
  special = { 'additional_special_tokens': ['<customer>', '<agent>', '<kb>']  }
  if args.model == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # del special['pad_token']
  elif args.model == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  elif args.model == 'gpt':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  elif args.model == 'bart':
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
  else:
    print(f'{args.model} not supported at this time')
    sys.exit()

  tokenizer.add_special_tokens(special)
  return tokenizer


def load_optimizer_scheduler(args, model, t_total):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    opt_path = os.path.join(args.save_dir, "optimizer.pt")
    schedule_path = os.path.join(args.save_dir, "scheduler.pt")
    if os.path.isfile(opt_path) and os.path.isfile(schedule_path):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(opt_path))
        scheduler.load_state_dict(torch.load(schedule_path))
    return optimizer, scheduler
