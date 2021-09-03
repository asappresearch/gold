# Generating Out-of-scope Labels with Data augmentation (GOLD)
This respository contains the code and data for GOLD: Improving Out-of-Scope Detection in Dialogues using Data Augmentation ([Chen](https://twitter.com/derekchen14) et al., 2021)

## Introduction

Dialogue systems deployed in the real-world frequently encounter ambiguous scenarios, unclear requests and shifting domains that fall outside of the scope of what they were originally trained to handle. When faced with such issues, the system must make a critical decision whether to proceed as usual or to escalate the situation to further personnel.  In order to do so, the system must be equipped with the ability to detect out-of-distribution utterances and dialogue breakdowns, which we group together as out-of-scope (OOS) cases.

We propose *G*enerating *O*ut-of-scope *L*abels with *D*ata augmentation (GOLD) as a method of generating OOS samples to improve detection accuracy.  GOLD operates by extracting potential matches from an auxiliary source dataset, generating OOS candidates from the matches, and then filtering those candidates to keep only the best ones for inclusion into the training set.  Our experiments show relative gains of around 50% against the average baseline across three dialogue datasets.  For more details, please see the full paper linked below.

Paper link: https://arxiv.org/abs/TBD

Blog link: https://www.asapp.com/blog/generating-oos-labels-with-data-augmentation/

![Data Augmentation Pipeline](/assets/images/pipeline.png)

## Usage
All code is run by executing the corresponding command within the shell script `run.sh`, which will kick off the data preparation and training within `main.py`.  To use, first unzip the files representing the three datasets found in `data` folder using the `gunzip` command (or similar).  Then comment or uncomment the appropriate lines in the shell script to get desired behavior. Finally, enter `sh run.sh` into the command line to get started.  Use the `--help` option of argparse for flag details or read through the file located within `utils/arguments.py`.

### Preparation
TBD

### Training
To specify the task for training, simply use the `--task` option with either `star`, `flow` or `rostd`, for Schema-guided Dialog Dataset, SM Calendar Flow, and Real Out-of-Domain Sentences respectively.  Options for different source datasets are 'OOO', 'TM', 'PC', 'QQP', and 'MIX' which are specified with the `--source-data` flag.  Loading scripts can be tuned to offer various other behaviors.

### Evaluation
Activate evaluation using the `--do-eval` flag.  By default, `run.sh` will perform evaluation for AUROC and AUPR.  To include other metrics, add the appropriate options of `--metric` or `--method`.

## Data
TBD

## Contact
Please email dchen@asapp.com for questions or feedback.

## Citation
TBD

