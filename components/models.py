import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from assets.static_vars import baseline_methods, direct_modes, device
# from transformers import GPT2Model

class BaseModel(nn.Module):
  # Main model for predicting Unified Meaning Representations of act, topic and modifier
  def __init__(self, args, ontology, tokenizer):
    super().__init__()
    self.version = args.version
    self.ontology = ontology
    self.name = 'basic'

    self.tokenizer = tokenizer
    self.model_setup(args)
    self.verbose = args.verbose
    self.debug = args.debug

    self.classify = nn.Linear(args.embed_dim, 1)
    self.sigmoid = nn.Sigmoid()
    self.criterion = nn.BCEWithLogitsLoss()
 
    self.save_dir = args.save_dir
    if args.version == 'augment':
      self.load_dir = args.save_dir
    elif args.version == 'baseline':
      self.load_dir = os.path.join(args.output_dir, args.task, 'baseline')
    self.opt_path = os.path.join(self.save_dir, f"optimizer_{args.version}.pt")
    self.schedule_path = os.path.join(self.save_dir, f"scheduler_{args.version}.pt")

  def model_setup(self, args):
    print(f"Setting up {args.model} model")
    if args.method == 'dropout' or args.version == 'augment':
      if args.model == 'bert':
        configuration = BertConfig(hidden_dropout_prob=args.threshold)
        self.encoder = BertModel(configuration)
      elif args.model == 'roberta':
        configuration = RobertaConfig(hidden_dropout_prob=args.threshold)
        self.encoder = RobertaModel(configuration)
    else:
      if args.model == 'bert':
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
      elif args.model == 'roberta':
        self.encoder = RobertaModel.from_pretrained('roberta-base')

    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check
    self.model_type = args.model

  def forward(self, inputs, targets, outcome='logit'):
    enc_out = self.encoder(**inputs)
    hidden = enc_out['last_hidden_state'][:, 0, :]   # batch_size, embed_dim
    logits = self.classify(hidden).squeeze(1)         # batch_size
    
    loss = self.criterion(logits, targets)
    output = logits if outcome == 'logit' else self.sigmoid(logits)
    return output, loss

  @classmethod
  def from_checkpoint(cls, args, targets, tokenizer, checkpoint_path):
    return cls(args, targets, tokenizer, checkpoint_path)

  def setup_optimizer_scheduler(self, learning_rate, total_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0 },
        {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup = int(total_steps * 0.2)
    self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                      num_warmup_steps=warmup, num_training_steps=total_steps)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(self.opt_path) and os.path.isfile(self.schedule_path):
        # Load in optimizer and scheduler states
        self.optimizer.load_state_dict(torch.load(opt_path))
        self.scheduler.load_state_dict(torch.load(schedule_path))

class IntentModel(BaseModel):
  # Main model for predicting the intent of a utterance, rather than binary OOS
  def __init__(self, args, ontology, tokenizer):
    super().__init__(args, ontology, tokenizer)
    if args.task == 'star':
      target_size = len(ontology['regular'])
    elif args.task == 'rostd':
      target_size = len(ontology['finegrain'])
    elif args.task == 'flow':
      allowed = list(ontology.keys())
      allowed.remove('Fence')   # OOS examples are never a valid intent
      target_size = 0
      for category in allowed:
        target_size += len(ontology[category])
    
    self.temperature = args.temperature
    self.dropout = nn.Dropout(args.drop_rate)
    self.classify = Classifier(args, target_size)
    self.softmax = nn.LogSoftmax(dim=1)
    self.criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    self.name = 'nlu'

  def forward(self, inputs, targets, outcome='loss'):
    enc_out = self.encoder(**inputs)
    sequence, pooled = enc_out['last_hidden_state'], enc_out['pooler_output']

    hidden = sequence[:, 0, :]                      # batch_size, embed_dim
    if outcome == 'odin':
      noise = torch.randn(hidden.shape) * 1e-6      # standard deviaton of epsilon = 1e-6
      hidden += noise.to(device)
    else:
      hidden = self.dropout(hidden)
    logit = self.classify(hidden, outcome)          # batch_size, num_intents    
    
    loss = torch.zeros(1)    # set as default loss
    if outcome == 'loss':   # used by default for 'intent' and 'direct' training
      output = logit     # logit is a FloatTensor, targets should be a LongTensor
      loss = self.criterion(logit, targets)
    elif outcome == 'gradient':   # we need to hallucinate a pseudo_label for the loss
      output = logit     # this output will be ignored during the return
      pseudo_label = torch.argmax(logit)
      loss = self.criterion(logit, pseudo_label.unsqueeze(0))
    elif outcome in ['dropout', 'maxprob']:
      output = self.softmax(logit)
    elif outcome in ['odin', 'entropy']:
      output = self.softmax(logit / self.temperature)
    else:                   # used by the 'direct' methods during evaluation
      output = logit

    return output, loss

class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden, outcome):
    # hidden has shape (batch_size, 2 * embed_dim)
    middle = self.relu(self.top(hidden))
    # hidden now is (batch_size, hidden_dim)
    logit = self.bottom(middle)
    # logit has shape (batch_size, num_slots)
    return middle if outcome in ['bert_embed', 'mahalanobis', 'gradient'] else logit
