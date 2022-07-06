# Generating Out-of-scope Labels with Data augmentation (GOLD)
This respository contains the code and data for GOLD: Improving Out-of-Scope Detection in Dialogues using Data Augmentation ([Chen](https://twitter.com/derekchen14) et al., 2021)

## Introduction

Dialogue systems deployed in the real-world frequently encounter ambiguous scenarios, unclear requests and shifting domains that fall outside of the scope of what they were originally trained to handle. When faced with such issues, the system must make a critical decision whether to proceed as usual or to escalate the situation to further personnel.  In order to do so, the system must be equipped with the ability to detect out-of-distribution utterances and dialogue breakdowns, which we group together as out-of-scope (OOS) cases.

We propose *G*enerating *O*ut-of-scope *L*abels with *D*ata augmentation (GOLD) as a method of generating OOS samples to improve detection accuracy.  GOLD operates by extracting potential matches from an auxiliary source dataset, generating OOS candidates from the matches, and then filtering those candidates to keep only the best ones for inclusion into the training set.  Our experiments show relative gains of around 50% against the average baseline across three dialogue datasets.  For more details, please see the full paper linked below.

Paper link: https://arxiv.org/abs/2109.03079

Blog link: https://www.asapp.com/blog/generating-oos-labels-with-data-augmentation/

![Data Augmentation Pipeline](/components/images/pipeline.png)

### Data Prep
Before running any of the models, the data will need to be in the right place.  To use, first unzip the data object `data.tar.gz` found in root directory using the `tar -xzvf` command (or similar). This should create an `assets` folder with two new directories and a handful of other files.  The two new directories contain three target datasets and six source datasets for training and augmentation, respectively.  Once the data is in place, you can remove the original data object if desired.  When running for the first time, additional pre-processing will occur, with the results saved to cache.

## General Usage
All experiment code is run by executing the corresponding command within the shell script `run.sh`, which will kick off the data preparation and training within `main.py`.  Please comment or uncomment the appropriate lines in the shell script to get desired behavior. For example, the code to pretrain the baseline methods are at the top and the evaluation of those baselines are in the next section.

To get started, install all major dependencies within `requirements.txt` in a new environment. Next, create an folder to store the model outputs, where the default is folder name is `results`.  As with all other settings, this can be changed by updating the params within `run.sh`.

## Training and Evaluation
The entire system is governed by argument flags. Kick off training with the `--do-train` option. Activate evaluation using the `--do-eval` flag.  To specify the task for training, simply use the `--task` option with either `star`, `flow` or `rostd`, for Schema-guided Dialog Dataset, SM Calendar Flow, and Real Out-of-Domain Sentences respectively.  Options for different source datasets are 'OOO', 'TM', 'PC', 'QQP', and 'MIX' which are specified with the `--source-data` flag. Extraction techniques depend on the `--technique` flag.  Check for other argument details in the `utils/arguments.py` file or use the `--help` option of argparse.

## Contact
Please email dchen@asapp.com for questions or feedback.

## Citation
```
@inproceedings{chen2021gold,
    title = "{GOLD}: Improving Out-of-Scope Detection in Dialogues using Data Augmentation",
    author = "Chen, Derek  and
      Yu, Zhou",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.35",
    doi = "10.18653/v1/2021.emnlp-main.35",
    pages = "429--442"
}

```