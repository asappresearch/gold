import os, sys, pdb
import numpy as np
import random
import torch

from tqdm import tqdm as progress_bar
from components.logger import ExperienceLogger
from components.models import BaseModel, IntentModel
from assets.static_vars import device, debug_break, direct_modes

from utils.help import set_seed, setup_gpus, check_directories, prepare_inputs
from utils.process import process_data, get_dataloader
from utils.arguments import solicit_params
from utils.load import load_data, load_tokenizer, load_ontology, load_best_model
from utils.evaluate import generate_clusters, process_diff, process_drop
from utils.evaluate import eval_quantify, eval_qualify, run_inference
from app import augment_features

def run_train(args, model, datasets, tokenizer, exp_logger):
  train_dataloader = get_dataloader(args, datasets['train'], split='train')
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  model.setup_optimizer_scheduler(args.learning_rate, total_steps)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    train_metric = ''
    model.train()

    for step, batch in enumerate(train_dataloader):
      inputs, labels = prepare_inputs(batch, model)
      pred, batch_loss = model(inputs, labels)
      exp_logger.tr_loss += batch_loss.item()
      loss = batch_loss / args.grad_accum_steps
      loss.backward()

      if args.verbose:
        train_results = eval_quantify(args, pred.detach(), labels.detach(), exp_logger, "train")
        train_metric = train_results[exp_logger.metric]
      if (step + 1) % args.grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        model.optimizer.step()  # backprop to update the weights
        model.scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        exp_logger.log_train(step, train_metric)
      if args.debug and step >= debug_break*args.log_interval:
        break

    eval_res = run_eval(args, model, datasets, tokenizer, exp_logger)
    if args.do_save and eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      exp_logger.save_best_model(model, tokenizer)
    early_stop = exp_logger.end_epoch()
    if early_stop: break

  return exp_logger.best_score

def run_eval(args, model, datasets, tokenizer, exp_logger, split='dev'):
  dataloader = get_dataloader(args, datasets[split], split)
  if split == 'test':
    if args.version == 'augment':
      model.load_dir = model.save_dir
    model = load_best_model(args, model, device)

  outputs = run_inference(args, model, dataloader, exp_logger, split)
  if args.quantify or split == 'dev':
    if args.version == 'baseline' and args.method in ['bert_embed', 'rob_embed', 'gradient']:
      clusters = generate_clusters(args, datasets['train'], model, exp_logger, split)
      outputs = process_diff(args, clusters, *outputs)
    elif args.version == 'baseline' and args.method == 'dropout':
      outputs = process_drop(args, *outputs, exp_logger)
    results = eval_quantify(args, *outputs, split)
  if args.qualify:
    results = eval_qualify(args, *outputs, tokenizer)
  return results
  
if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  raw_data = load_data(args)
  tokenizer = load_tokenizer(args)
  ontology = load_ontology(args)

  if args.version == 'augment' and not args.do_eval:
    features = augment_features(args, raw_data, tokenizer, ontology)
  else: 
    _, features = raw_data
  datasets = process_data(args, features, tokenizer, ontology)

  if args.version in ['direct', 'augment']:
    model = BaseModel(args, ontology, tokenizer).to(device)
  elif args.version in ['baseline', 'intent']:
    model = IntentModel(args, ontology, tokenizer).to(device)
  exp_logger = ExperienceLogger(args, model.save_dir)
  
  if args.do_train:
    best_score = run_train(args, model, datasets, tokenizer, exp_logger)
  if args.do_eval:
    run_eval(args, model, datasets, tokenizer, exp_logger, split='test')
