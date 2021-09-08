import os, pdb, sys
import re
import json
import torch
import numpy as np
import random

from torch import nonzero
from tqdm import tqdm as progress_bar
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from assets.static_vars import device, debug_break, direct_modes, lorem_ipsum
from scipy.stats import entropy

from utils.help import *

def qualify(args, preds, targets, contexts, texts, tokenizer):
  results = []
  for pred, target, context, text in zip(preds, targets, contexts, texts):
    if target and pred != target:
      context_ids = [x for x in context if x not in [0, 101, 102]]
      history = tokenizer.decode(context_ids)
      history = lorem_ipsum + history
      history = history[-200:]

      print(history)
      print('Predicted:', 'ambiguous' if pred else 'in scope')
      print('Actual:', 'ambiguous' if target else 'in scope', text)
      pdb.set_trace()

  return results

def quantify(args, predictions, targets, exp_logger, split):
  if exp_logger.version == 'intent':
    preds = torch.argmax(predictions, axis=1)
    results = accuracy_eval(args, preds, targets, exp_logger)
  elif args.debug or split == 'train':
    y_true = targets.to(torch.int8).cpu().numpy()
    y_score = predictions.cpu().numpy()
    results = {'epoch': exp_logger.epoch, 'aupr': average_precision_score(y_true, y_score) }
  else:
    results = binary_curve(args, predictions, targets, exp_logger)

  if split != 'train':
    exp_logger.log_info(results)
  return results

def accuracy_eval(args, preds, targets, exp_logger):
  correct = torch.sum(preds == targets)
  total = len(targets)
  acc = correct.item() / float(total)

  eval_loss = round(exp_logger.eval_loss, 3)
  results = {'epoch': exp_logger.epoch, 'accuracy': round(acc, 4), 'loss': eval_loss }
  return results

def stable_sum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumulative sum and check that final value matches sum
    arr (array-like): To be cumulatively summed as flat
    rtol (float):  Relative tolerance, see ``np.allclose``
    atol (float):  Absolute tolerance   """
    cumulative = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(cumulative[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum unstable: last element does not correspond to sum')
    return cumulative

def fpr_at_recall(y_true, y_score, level):
    # make y_true a boolean vector
    y_true = (y_true == 1.)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_sum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]
    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - level))
    return fps[cutoff] / (np.sum(np.logical_not(y_true)))

def binary_curve(args, preds, targets, exp_logger=None):
  y_true = targets.to(torch.int8).numpy()
  try:
    y_score = preds.numpy()
  except(AttributeError):
    y_score = preds
  # y_pred = preds.to(torch.int8).numpy()
  # sklearn_score = sklearn_f1(y_true, y_pred)  <<  same output but does not give pre/rec numbers
  auroc = roc_auc_score(y_true, y_score)
  aupr = average_precision_score(y_true, y_score)
  fpra95 = fpr_at_recall(y_true, y_score, level=0.95)
  fpra90 = fpr_at_recall(y_true, y_score, level=0.9)

  if exp_logger is None:
    results = {'threshold': args.threshold }
  else:
    results = {'epoch': exp_logger.epoch }
  results['auroc'] = round(auroc, 4)
  results['aupr'] = round(aupr, 4)
  results['fpr@0.95'] = round(fpra95, 4)
  results['fpr@0.90'] = round(fpra90, 4)
  return results

def binary_eval(args, preds, targets, exp_logger=None):
  try:
    pred_indices = nonzero(preds, as_tuple=False)
  except(TypeError):
    preds = torch.tensor(preds)
    pred_indices = nonzero(preds, as_tuple=False)
  positives = list(pred_indices.detach().cpu().numpy())
  negatives = [x for x in range(len(targets)) if x not in positives]

  true_indices = nonzero(targets, as_tuple=False)
  actuals = list(true_indices.detach().cpu().numpy())
  precision, recall, f1_score = calculate_f1(positives, negatives, actuals, args.verbose)
  
  if args.version == 'baseline':
    results = {'threshold': args.threshold}
  else:
    results = {'epoch': exp_logger.epoch}
  results['precision'] = round(precision, 4)
  results['recall'] = round(recall, 4)
  results['f1_score'] = round(f1_score, 4)
  return results

def has_majority_vote(max_preds):
  a_equals_b = max_preds[0] == max_preds[1]
  b_equals_c = max_preds[1] == max_preds[2]
  c_equals_a = max_preds[2] == max_preds[0]
  return a_equals_b or b_equals_c or c_equals_a

def gradient_inference(args, model, dataloader):
  gradients, scope_labels = [], []

  for batch in progress_bar(dataloader, total=len(dataloader)):
    inputs, labels = prepare_inputs(batch, model)
    scope_labels.append(labels.detach().cpu())       # each label is (batch_size * 2,)

    _, instance_loss = model(inputs, labels, outcome='gradient')
    instance_loss.backward()
    
    for name, params in model.named_parameters():
      # if name == "classify.top.weight":                        # (300,768) 
      if name == "encoder.embeddings.word_embeddings.weight":  # (30525, 768)
        gradient = params.grad.detach().cpu().sum(axis=0)     
        gradients.append(gradient)
    # zero out gradients so they don't accumulate between instances
    model.zero_grad()

  # requires stacking because batch size = 1, so we need to add back the extra dim
  vectors = torch.stack(gradients)         # num_train_examples, hidden_dim
  scope_targets = torch.cat(scope_labels)
  return vectors, scope_targets

def dropout_inference(args, model, dataloader):
  ''' Evaluation of uncertainty based on ensemble created through dropout '''
  predictions_ensemble, targets_ensemble = [], []
  ensemble_size = 3
  for iteration in range(ensemble_size):
    print(f'Starting round {iteration+1} of {ensemble_size}')
    model.train()   # because we want to maintain dropout
    
    predictions, targets = [], []
    for step, batch in enumerate(dataloader):
      inputs, labels = prepare_inputs(batch, model)
      with torch.no_grad():
        preds, batch_loss = model(inputs, labels, outcome='skip')
        predictions.append(preds.detach().cpu())
        targets.append(labels.detach().cpu())
    
    grouped_preds = torch.cat(predictions, axis=0)
    grouped_targets = torch.cat(targets)
    predictions_ensemble.append(torch.argmax(grouped_preds, axis=1))
    targets_ensemble.append(grouped_targets)

  return predictions_ensemble, targets_ensemble, ensemble_size

def process_drop(args, predictions_ensemble, targets_ensemble, ensemble_size, exp_logger):
  # we only need the first set of targets since they are all the same
  targets = targets_ensemble[0]
  num_examples = len(targets)
  
  intent_preds = [] # list with length equal to number of test examples
  for i in range(num_examples):
    # each item is another list with length equal to ensemble_size
    collated_pred = [predictions_ensemble[j][i] for j in range(ensemble_size)]
    intent_preds.append(collated_pred)

  uncertainty_preds = []   # list used for final OOS detection
  for max_preds in intent_preds:
    # if the ensemble has a majority vote, then it is confident, which means
    # the system makes a prediction and is not uncertain
    uncertainty_preds.append(0 if has_majority_vote(max_preds) else 1)

  return torch.tensor(uncertainty_preds), targets, exp_logger

def centroid_cache(args):
  cache_path = os.path.join(args.input_dir, 'cache', f'{args.task}_{args.method}.pt')
  use_cache = not args.ignore_cache

  if os.path.exists(cache_path) and use_cache:
    print(f'Loading pre-computed centroids from {cache_path}')
    vectors = torch.load(cache_path)
    return vectors, True
  else:
    print(f'Creating new clusters from vectors ...')
    return cache_path, False

def compute_centroids(vectors, labels):
  '''cluster dialogues into different known groups based on labels  
    vectors - a matrix of size (num_examples, hidden_dim)
    labels - vector of size (num_examples)
  Returns:
    centroids - matrix of size (num_intents, hidden_dim)
  '''
  clusters = defaultdict(list)
  for vector, label in zip(vectors, labels):
    key = 'intent_' + str(int(label.item() + 1))
    clusters[key].append(vector)

  centers = []
  for intent, nodes in clusters.items():
    cluster = torch.stack(nodes)           # (variable, hidden_dim)
    center = torch.mean(cluster, axis=0)   # (hidden_dim, )
    centers.append(center)

  centroids = torch.stack(centers)          # (num_intents, hidden_dim)
  return centroids

def make_clusters(args, dataloader, model, exp_logger, split):
  ''' create the clusters and store in cache, number of clusters should equal the number
  of intents.  Each cluster is represented by the coordinates of its centroid location '''
  cache_results, already_done = centroid_cache(args)
  if already_done:
    return cache_results

  vectors, labels, _ = run_inference(args, model, dataloader, exp_logger, split)
  centroids = compute_centroids(vectors, labels)
  torch.save(centroids, cache_results)
  print(f'Saved centroids of shape {centroids.shape} to {cache_results}')
  return centroids

def make_covariance_matrix(args, vectors, clusters):
  if args.verbose:
    print("Creating covariance matrix")

  num_intents, hidden_dim = clusters.shape
  covar = torch.zeros(hidden_dim, hidden_dim)
  for cluster in progress_bar(clusters, total=num_intents):
    for vector in vectors:
      diff = (vector - cluster).unsqueeze(1)  # hidden_dim, 1
      covar += torch.matmul(diff, diff.T)           # hidden_dim, hidden_dim
  covar /= len(vectors)                       # divide by a scalar throughout
  inv_cov_matrix = np.linalg.inv(covar)
  return torch.tensor(inv_cov_matrix)

def mahala_dist(x, mu, VI):
  ''' 50% faster than using scipy or sklearn since we keep in torch tensor format '''
  x_minus_mu = (x - mu).unsqueeze(0)
  left_term = torch.matmul(x_minus_mu, VI)
  score = torch.matmul(left_term, x_minus_mu.T)
  return torch.sqrt(abs(score))

def process_diff(args, clusters, vectors, targets, exp_logger):
  ''' figure out how far from clusters '''
  inv_cov_matrix = make_covariance_matrix(args, vectors, clusters)
  uncertainty_preds = []
  
  for vector in progress_bar(vectors, total=len(vectors)):
    if args.method == 'mahalanobis':
      distances = [mahala_dist(vector, cluster, inv_cov_matrix) for cluster in clusters]
      min_distance = min(distances)
    else:
      distances =  torch.cdist(vector.unsqueeze(0), clusters, p=2)  # 2 is for L2-norm
      min_distance = torch.min(distances)  # each distance is a scalar
    uncertainty_preds.append(min_distance.item())
    # if the min_dist is greater than some threshold, then it is uncertain
    # uncertainty_preds.append(min_distance.item() > args.threshold)
  return torch.tensor(uncertainty_preds), targets, exp_logger

def run_inference(args, model, dataloader, exp_logger, split):
  if args.method == 'gradient':
    vectors, scope_targets = gradient_inference(args, model, dataloader)
    return vectors, scope_targets, exp_logger
  elif args.method == 'dropout':
    predictions_ensemble, targets_ensemble, ensemble_size = dropout_inference(args, model, dataloader)
    return predictions_ensemble, targets_ensemble, ensemble_size

  predictions, all_targets = [], []
  all_contexts, texts = [], []

  for batch in progress_bar(dataloader, total=len(dataloader)):
    inputs, labels, text = prepare_inputs(batch, model, use_text=True)
    texts.extend(text)

    out = args.method if args.version == 'baseline' else 'loss'
    with torch.no_grad():
      pred, batch_loss = model(inputs, labels, outcome=out)
    predictions.append(pred.detach().cpu())
    all_targets.append(labels.detach().cpu())
    all_contexts.append(inputs['input_ids'].detach().cpu())
    
    if split == 'dev':
      exp_logger.eval_loss = batch_loss.mean().item()
      exp_logger.eval_step += 1
      if args.debug and exp_logger.eval_step >= debug_break: break

  preds = combine_preds(predictions, exp_logger.version)
  targets = torch.cat(all_targets)
  return preds, targets, exp_logger

def combine_preds(predictions, mode):
  # unlike compute_preds, combine_preds is threshold independent
  if mode == 'augment':
    return torch.cat(predictions)
  elif mode in ['maxprob', 'odin']:
    joined = torch.cat(predictions, axis=0)          # num_examples, num_classes
    expo = torch.exp(joined)                         # exponentiation is required due to LogSoftmax
    values, indexes = torch.max(expo, axis=1)
    return -values
  elif mode == 'entropy':
    # produces results based on the prediction entropy which considers the overall distribution
    joined = torch.cat(predictions, axis=0)          # num_examples, num_classes
    expo = torch.exp(joined)                         # exponentiation is required due to LogSoftmax
    entropy_vals = entropy(expo, axis=1)
    return entropy_vals
  else:   # mahalanobis, bert_embed, gradient, intent
    return torch.cat(predictions, axis=0)

def calculate_f1(pos_list, neg_list, actual_list):
  '''Inputs: suppose we have a batch_size of N binary predictions
    pos_list - list of indexes where we predicted "yes", this is likely to be
      the list of examples that were above a certain threshold
    neg_list - list of indexes where we predicted "no", the total size of the
      len(pos_list) + len(neg_list) = N, thus this represents all remaining indexes
    actual_list - a list containing the indexes that were truly "yes", ideally
      the pos_list and actual_list will match exactly
  '''
  true_pos, false_pos = 0, 0
  for pos in pos_list:
    if pos in actual_list:
      true_pos += 1
    else:
      false_pos += 1

  true_neg, false_neg = 0, 0
  for neg in neg_list:
    if neg in actual_list:
      false_neg += 1
    else:
      true_neg += 1

  """print(f'true_pos {true_pos} + false_pos {false_pos} = all_pos {len(pos_list)}')
  print(f'true_pos {true_pos} + false_neg {false_neg} = actual_pos {len(actual_list)}')
  print(f'true_neg {true_neg}, number of all_negatives = {len(neg_list)}')"""
  epsilon = 1e-9
  precision = true_pos / (true_pos + false_pos + epsilon)
  recall = true_pos / (true_pos + false_neg + epsilon)
  f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
  return precision, recall, f1_score


if __name__ == '__main__':
  args = solicit_params()
  args.multiwoz_version = '2.1'
  args.use_action = True
  args.use_knowledge = True

  random.seed(14)
  np.random.seed(14)
  # joint_acc, _ = eval_dst(args)
  joint_acc, _ = eval_confidence(args)
  print('joint accuracy: {}%'.format(joint_acc))


