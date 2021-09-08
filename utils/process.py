import os, pdb, sys
import json
import random
import pickle as pkl
import numpy as np
import torch

from copy import deepcopy
from tqdm import tqdm as progress_bar
from components.datasets import BaseInstance, DirectDataset, IntentDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def get_dataloader(args, dataset, split='train'):
  sampler = RandomSampler(dataset) if split == 'train' else SequentialSampler(dataset)
  collate = dataset.collate_func
  # batch size is 1 to avoid gradient of multiple examples from mixing together
  b_size = 1 if args.method == 'gradient' else args.batch_size
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=b_size, collate_fn=collate)
  print(f"Loaded {split} data with {len(dataloader)} batches")
  return dataloader

def load_mixture(args):
  results = {}
  sources = ['QQP', 'PC', 'OSQ']

  for src in sources:
    augment_path = os.path.join(args.input_dir, 'augments', f'{args.task}_{src}.pkl')
    if not os.path.exists(augment_path):
      raise ValueError(f"All augmentations must be present to use MIX, but you are missing {src}")
    
    res = pkl.load( open( augment_path, 'rb' ) )
    print(f'Loaded augmentations from {augment_path} for MIX')
    if len(results) == 0:
      results = res.copy()
    else:
      aug = [x for x in res['train'] if x.oos_label == 1]
      results['train'].extend(aug)

  return results

def check_cache(args):
  source = 'baseline' if args.version == 'baseline' else args.source_data
  folder = 'augments' if args.version == 'augment' else 'cache'
  cache_path = os.path.join(args.input_dir, folder, f'{args.task}_{source}.pkl')
  use_cache = not args.ignore_cache

  if source == 'MIX':
    mix_results = load_mixture(args)
    return mix_results, True

  if os.path.exists(cache_path) and use_cache:
    print(f'Loading features from cache at {cache_path}')
    results = pkl.load( open( cache_path, 'rb' ) )
    return results, True
  else:
    print(f'Creating new input features ...')
    return cache_path, False

def prepare_features(args, target_data, tokenizer, cache_path):
  all_features = {}
  for split, examples in target_data.items():
    
    feats = []
    for example in progress_bar(examples, total=len(examples)):
      if isinstance(example['context'], list):
        conversation = ' '.join(example['context'])
      else:
        conversation = example['context']
      embed_data = tokenizer(conversation, padding='max_length', truncation=True, max_length=args.max_len)
      instance = BaseInstance(embed_data, example)
      feats.append(instance)
    all_features[split] = feats
    print(f'Number of {split} features:', len(feats))

  if args.version == 'baseline':
    pkl.dump(all_features, open(cache_path, 'wb'))
  return all_features

def process_data(args, features, tokenizer, ontology):
  train_size, dev_size = len(features['train']), len(features['dev'])
  if args.verbose:
    print(f"Running with {train_size} train and {dev_size} dev features")

  datasets = {}
  for split, feat in features.items():
    if args.version == 'augment':
      datasets[split] = DirectDataset(feat, tokenizer, ontology, split)
    elif args.version == 'baseline':
      if split == 'train' or (split == 'dev' and args.do_train):
        ins_data = [x for x in feat if x.oos_label == 0]
        datasets[split] = IntentDataset(ins_data, tokenizer, ontology, split)
      else:
        datasets[split] = DirectDataset(feat, tokenizer, ontology, split)

  return datasets
