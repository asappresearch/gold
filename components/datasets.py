import os, pdb, sys
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class BaseInstance(object):
  def __init__(self, guid, split, embed_data, example):
    embedding, segments, input_masks = embed_data
    context, label = example['x'], example['y']    

    self.guid = guid
    self.split = split
    self.embedding = embedding
    self.segments = segments
    self.input_mask = input_masks

    self.context = context   # in natural language text
    if 'z' in example:
      self.agent_response = example['z']

    self.oos_label = label['binary']
    self.intent_label = label['multiclass']
    self.label_text = label['text']

class ParaphraseInstance(object):
  def __init__(self, guid, encoder_data, decoder_data, label_ids):
    self.guid = guid
    self.labels = label_ids

    self.enc_inputs = encoder_data['input_ids']
    self.enc_mask = encoder_data['attention_mask']
    self.enc_text = encoder_data['raw_text']

    self.dec_inputs = decoder_data['input_ids']
    self.dec_mask = decoder_data['attention_mask']
    self.dec_text = decoder_data['raw_text']

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

class AppDataset(Dataset):
  def __init__(self, data, tokenizer, targets):
    self.data = data
    self.tokenizer = tokenizer
    self.ontology = targets

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
    label_ids = torch.tensor([f.label_id for f in batch], dtype=torch.long)
    guid_to_data = {f.guid: f for f in batch}
    return input_ids, segment_ids, input_masks, label_ids, guid_to_data
