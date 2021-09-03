import argparse
import os

def solicit_params():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input-dir", default='datasets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--task", default='star', type=str,
                choices=['clinc', 'flow', 'star', 'master', 'rostd'])
    parser.add_argument("--model", default='bert', type=str,
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--seed", default=42, type=int)

    # Custom paper parameters
    parser.add_argument("--source-data", default='PC', type=str, 
                choices=['US', 'OOO', 'TM', 'PC', 'QQP', 'MIX', 'MQA'])
    parser.add_argument("--reweight", default=-1.0, type=float,
                help="Use weighted BCE loss during model training.")
    parser.add_argument("--num-matches", default=24, type=int, 
                help="The number of matches to swap in, set to -1 for using all matches")
    parser.add_argument("--threshold", default=0.5, type=float,
                help="Various threshold levels, used for confidence uncertainty grid")
    parser.add_argument("--temperature", default=1.0, type=float,
                help="Various temperature levels for scaling the softmax output")
    parser.add_argument("--metric", default='curve', type=str,
                choices=['efwon', 'curve', 'range', 'fpran'])

    # Experiment options
    parser.add_argument("--version", default="direct", type=str,
                help="Version of the code to train or evaluate",
                choices=['baseline', 'augment', 'direct', 'intent'])
    parser.add_argument("--method", default="maxprob", type=str,
                help="The baseline method to use when detecting OOS utterances",  # 'likelihood'
                choices=['maxprob', 'entropy', 'bert_embed', 'rob_embed', 'gradient', 'dropout', 'odin'])
    parser.add_argument("--technique", default="tfidf", type=str, 
                choices=['paraphrase', 'encoder', 'glove', 'tfidf', 'length', 'random'],
                help="Extraction technique for extracting candidates from source data")
    parser.add_argument("--distance", default="euclidean", type=str, 
                help="Distance metric", choices=['euclidean', 'cosine', 'mahalanobis'])
    parser.add_argument("--ensemble", default="cat_majority", type=str,
                help="Ensemble configuration used to filter for matches from candidates",
                choices=['cat_complete', 'cat_majority', 'top_complete', 'top_majority'])

    # Key settings
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--debug", action="store_true",
                help="Whether to run in debug mode which is exponentially faster")
    parser.add_argument("--verbose", action="store_true",
                help="Whether to run with extra prints to help debug")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the dev set.")
    parser.add_argument("--do-save", action="store_true",
                help="Whether to save models, which override previous checkpoints")
    parser.add_argument("--bootstrap", action="store_true",
                help="Self train without the seed known OOS data.")
    parser.add_argument("--breakdown", action="store_true",
                help="Whether to run evaluation with breakdown of stats")
    parser.add_argument("--log-interval", type=int, default=500,
                help="Log every X updates steps.")
    
    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=16, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument('--grad-accum-steps', default=1, type=int,
                help='Number of steps for gradient accumulation')
    parser.add_argument("--learning-rate", default=3e-5, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=300, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--embed-dim", default=768, type=int,
                help="The embedding dimension of pretrained LM.")
    parser.add_argument("--weight-decay", default=0.0, type=float,
                help="Weight decay if we apply some.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=3, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--max-len", default=256, type=int,
                help="maximum sequence length to look back")
    parser.add_argument("--warmup-steps", default=0, type=int, 
                help="Linear warmup over warmup-steps.")

    # Evaluation options
    parser.add_argument("--eval-mode", default="all", type=str,
                         help="Whether to use attention during intent training")
    parser.add_argument("--qualify", action='store_true',
                         help="Whether to include joint accuracy scores during evaluation")
    parser.add_argument("--quantify", action='store_true',
                         help="Whether to include inform/success/BLEU scores during evaluation")
    parser.add_argument("--interact", action='store_true',
                         help="Whether to include inform/success/BLEU scores during evaluation")

    args = parser.parse_args()
    return args
