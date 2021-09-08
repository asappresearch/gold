import os, pdb
import numpy as np
import torch
import random
import logging
import re
import glob
import shutil
import time as tm

from transformers import logging as tlog
tlog.set_verbosity_error()

class ExperienceLogger:
    def __init__(self, args, save_dir):
        self.args = args
        self.learning_rate = args.learning_rate
        self.task = args.task
        self.source = args.source_data
        self.save_path = save_dir

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        log_path = os.path.join(save_dir, f'v{args.task}.log')
        self.logger.addHandler(logging.FileHandler(log_path))
        self.logger.debug(args)      
        
        if args.version == 'baseline':
            if args.do_train:
                self.metric = 'accuracy'
                self.version = 'intent'
            else:
                self.metric = 'aupr'
                self.version = args.method
        elif args.version == 'augment':
            self.metric = 'aupr'
            self.version = 'augment'
            self.log_info(args)

        self.global_step = 0
        self.eval_step = 0
        self.log_interval = args.log_interval
        self.epoch = 1   # epoch count
        self.num_epochs = args.n_epochs

        self.best_score = {'epoch': 1, 'accuracy': 0, 'aupr': 0}
        self.do_save = args.do_save
        self.differences = []
        self.logging_loss = 0.0
        self.tr_loss = 0.0
        self.eval_loss = 0.0

    def log_info(self, text):
        self.logger.info(text)

    def start_train(self, total_step):
        self.logger.info("***** Running training *****")
        self.logger.info("  Num Epochs = %d" % self.args.n_epochs)
        self.logger.info("  Total train batch size  = %d" % self.args.batch_size)
        self.logger.info("  Total optimization steps = %d" % total_step)
        self.logger.info("  Running experiment for {}".format(self.task))

    def start_epoch(self, dataloader):
        self.logger.info(f"Starting epoch {self.epoch} of {self.num_epochs}")
        self.start_time = tm.time()
        self.num_steps = len(dataloader)

    def end_epoch(self):
        self.epoch += 1
        self.end_time = tm.time()

        raw_diff = self.end_time - self.start_time
        minute_diff = round(raw_diff / 60.0, 3)
        self.differences.append(minute_diff)
        avg_diff = round(np.average(self.differences), 3)

        met = round(self.best_score[self.metric] * 100, 2)
        if self.version == 'intent':
            self.logger.info(f"Best epoch is {self.best_score['epoch']} with {met}% accuracy")
        else:
            self.logger.info(f"Best epoch is {self.best_score['epoch']} with {met}% f1_score")
        self.logger.info(f"Current epoch took {minute_diff} min, average is {avg_diff} min")

        return self.early_stop()

    def early_stop(self):
        below_threshold = False
        # if self.args.debug:
        return below_threshold

        if self.epoch >= 7 and self.best_score['accuracy'] < 0.1:
            below_threshold = True
        if self.epoch >= 14 and self.best_score['accuracy'] < 0.2:
            below_threshold = True
        if self.epoch >= 21 and self.best_score['accuracy'] < 0.3:
            below_threshold = True

        if below_threshold:
            self.logger.info(f"Had to early stop at epoch {self.epoch}")
        return below_threshold

    def log_train(self, step, train_metric):
        self.global_step += 1

        if self.global_step < 100 and self.global_step % 10 == 0:
          print(self.global_step)
        if self.global_step < 1000 and self.global_step % 100 == 0:
            print(self.global_step)

        if self.global_step % self.log_interval == 0:
            current_loss = (self.tr_loss - self.logging_loss) / self.log_interval
            self.logging_loss = self.tr_loss

            step_report = f'[{step+1}/{self.num_steps}] '
            loss_report = 'Mean_loss: %.3f, ' % current_loss
            metric_report = f'{self.metric}: {train_metric}'
            print(step_report + loss_report + metric_report)

    def save_best_model(self, model, tokenizer):
        if self.do_save and self.best_score[self.metric] > 0.1:
            model_to_save = model.module if hasattr(model, 'module') else model
            if self.version == 'intent':
                learning_rate = str(self.args.learning_rate)
                accuracy = str(self.best_score['accuracy'] * 10000)[:3]
                ckpt_name = f'epoch{self.epoch}_{self.task}_lr{learning_rate}_acc{accuracy}.pt'
            elif self.version == 'augment':
                auroc = str(self.best_score['auroc'] * 1000)[:3]
                aupr = str(self.best_score['aupr'] * 1000)[:3]
                fpr_score = str(self.best_score['fpr@0.95'] * 1000)[:3]
                ckpt_name = f'epoch{self.epoch}_fpr{fpr_score}_auroc{auroc}_aupr{aupr}.pt'
            else:
                precision = str(self.best_score['precision'] * 1000)[:3]
                recall = str(self.best_score['recall'] * 1000)[:3]
                f1_score = str(self.best_score['f1_score'] * 1000)[:3]
                ckpt_name = f'source{self.source}_pre{precision}_rec{recall}_fs{f1_score}.pt'
            
            ckpt_path = os.path.join(self.save_path, ckpt_name)
            # os.makedirs(ckpt_path, exist_ok=True)
            torch.save(model_to_save.state_dict(), ckpt_path)
            # model_to_save.encoder.save_pretrained(ckpt_path)
            # tokenizer.save_pretrained(ckpt_path)
            # torch.save(model.optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
            # torch.save(model.scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))
            print(f"Saved a model at {ckpt_path}")
            self.prune_saves()

    def prune_saves(self, is_directory=False, num_keep=5):
        is_binary = self.version in ['direct', 'augment']
        # folders = glob.glob(os.path.join(self.save_path, "epoch*"))
        files = [f for f in os.listdir(self.save_path) if f.endswith('.pt')]
        # if len(folders) > num_keep:
        if len(files) > num_keep:
            # acc_and_folders = []
            scores_and_files = []
            for fname in files:
                re_str = r'fs([0-9]{3})\.pt$' if is_binary else r'acc([0-9]{3})\.pt$'
                regex_found = re.findall(re_str, fname)
                if regex_found:
                    # accuracy = int(regex_found[0])
                    score = int(regex_found[0])
                    filepath = os.path.join(self.save_path, fname)
                    # acc_and_folders.append((accuracy, fname))
                    scores_and_files.append((score, filepath))
            
            scores_and_files.sort(key=lambda tup: tup[0], reverse=True)  # largest to smallest
            # for _, folder in acc_and_folders[num_keep:]:
            for _, file in scores_and_files[num_keep:]:
                # shutil.rmtree(folder) # for recursive removal
                os.remove(file)
                # print(f'removed {folder} due to pruning')
                print(f'removed {file} due to pruning')
