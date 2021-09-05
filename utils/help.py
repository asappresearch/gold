import os
import numpy as np
import torch
import random
import re

from assets.static_vars import device

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_gpus(args):
    n_gpu = 0  # set the default to 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    if n_gpu > 0:   # this is not an 'else' statement and cannot be combined
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args

def check_directories(args):
    task_path = os.path.join(args.output_dir, args.task)
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        print(f"Created {task_path} directory")
    
    if args.version == 'baseline':
        folder = 'baseline' if args.do_train else args.method
    elif args.version == 'augment':
        folder = args.technique
    
    save_path = os.path.join(task_path, folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created {save_path} directory")
    args.save_dir = save_path

    cache_path = os.path.join(args.input_dir, 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        print(f"Created {cache_path} directory")
    augmentation_path = os.path.join(args.input_dir, 'augments')
    if not os.path.exists(augmentation_path):
        os.mkdir(augmentation_path)
        print(f"Created {augmentation_path} directory")

    if args.debug:
        args.log_interval /= 10

    return args

def prepare_inputs(batch, model, use_text=False):
    ''' Inputs: the batches from the collate function in the Dataset
    Outputs: something that a nn.Module model can consume, with inputs on device '''
    # batch = [b.to(device) for b in batch]
    # inputs = {'input_ids': batch[0], 'token_type_ids': batch[1], 'attention_mask': batch[2]} 
    # targets = batch[3]
    btt = [b.to(device) for b in batch[:4]]
    inputs = {'input_ids': btt[0], 'token_type_ids': btt[1], 'attention_mask': btt[2]} 
    targets = btt[3]

    if use_text:
        target_text = batch[4]
        return inputs, targets, target_text
    else:
        return inputs, targets

def contains(smaller, larger, use_stop=False):
    ''' checks is the larger list contains the smaller list and returns the index if True '''
    for index, element in enumerate(larger):
        if element == smaller[0]:
            stop = index + len(smaller)
            if larger[index:stop] == smaller:
                return stop if use_stop else index
    return -1

def clean_unnecessary_spaces(out_string):    
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string

def memstat(message):
    malloc = torch.cuda.memory_allocated()
    human_malloc = str(round( (malloc / 1000000), 2)) + "MB"
    maxmem = torch.cuda.max_memory_allocated()
    human_maxmem = str(round( (maxmem / 1000000), 2)) + "MB"
    print(f"{message} -- Current memory: {human_malloc}, Max: {human_maxmem}")
