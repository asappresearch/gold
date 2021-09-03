import torch

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}

# DEFINE special tokens
special_tokens = {
    'SOS_token': 0,
    'EOS_token': 1,
    'UNK_token': 2,
    'PAD_token': 3,
}

baseline_methods = ['maxprob', 'entropy', 'dropout', 'bert_embed', 'rob_embed' 'gradient']
direct_modes = ['direct', 'augment']
augment_versions = ['swap', 'extract', 'paraphrase', 'both']
alternative_intent = ['maxprob', 'entropy', 'intent']

debug_break = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer interdum enim nec ligula porttitor, hendrerit congue sem lobortis. Praesent sed justo in ex maximus tincidunt."
