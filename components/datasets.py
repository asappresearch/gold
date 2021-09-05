import os, pdb, sys
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class BaseInstance(object):
  def __init__(self, embed_data, example):
    self.embedding = embed_data['input_ids']
    self.segments = embed_data['token_type_ids']
    self.input_mask = embed_data['attention_mask']

    self.guid = example['guid']
    self.split = example['split']
    self.context = example['context']   # in natural language text
    self.agent_response = example['agent_text']

    self.oos_label = example['oos_label']
    self.intent_label = example['intent_label']
    self.label_text = example['label_text']

class DirectDataset(Dataset):
  def __init__(self, data, tokenizer, targets, split='train'):
    self.data = data
    self.tokenizer = tokenizer
    self.ontology = targets
    self.split = split

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

  def collate_func(self, batch):
    ''' convert a batch of feature instances with attributes into
    a groups of attributes each which have batch_size number of features '''
    input_ids = torch.tensor([f.embedding for f in batch], dtype=torch.long)
    segment_ids = torch.tensor([f.segments for f in batch], dtype=torch.long)
    input_masks = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
    label_ids = torch.tensor([f.oos_label for f in batch], dtype=torch.float)
    label_texts = [f.label_text for f in batch]
    return input_ids, segment_ids, input_masks, label_ids, label_texts

class IntentDataset(Dataset):
  def __init__(self, data, tokenizer, targets, split='train'):
    self.data = data
    self.tokenizer = tokenizer
    self.ontology = targets
    self.split = split

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

  def collate_func(self, batch):
    ''' convert a batch of feature instances with attributes into
    a groups of attributes each which have batch_size number of features '''
    input_ids = torch.tensor([f.embedding for f in batch], dtype=torch.long)
    segment_ids = torch.tensor([f.segments for f in batch], dtype=torch.long)
    input_masks = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
    label_ids = torch.tensor([f.intent_label for f in batch], dtype=torch.long)

    label_texts = [f.label_text for f in batch]
    return input_ids, segment_ids, input_masks, label_ids, label_texts
