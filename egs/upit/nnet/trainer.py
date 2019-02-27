#!/usr/bin/env python

# wujian@2018

import os
import sys
import time

from itertools import permutations
from collections import defaultdict

import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from utils import get_logger


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class AverageMeter(object):
    """
    A simple average meter
    """

    def __init__(self):
        self.val = defaultdict(float)
        self.cnt = defaultdict(int)

    def reset(self):
        self.val.clear()
        self.cnt.clear()

    def add(self, key, value):
        self.val[key] += value
        self.cnt[key] += 1

    def value(self, key):
        if self.cnt[key] == 0:
            return 0
        return self.val[key] / self.cnt[key]

    def sum(self, key):
        return self.val[key]

    def count(self, key):
        return self.cnt[key]


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start


class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid="0",
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=50,
                 resume=None,
                 no_impr=6):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpu = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        num_params = sum([param.nelement()
                          for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model({:.2f}M) to GPU:{}".format(
            num_params, gpuid))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.nnet.train()

        stats = AverageMeter()
        timer = SimpleTimer()

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)

            stats.add("loss", loss.item())
            loss.backward()

            progress = stats.count("loss")
            if not progress % self.logging_period:
                self.logger.info("Processed {:d} batches...".format(progress))

            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()

        return stats.value("loss"), stats.count("loss"), timer.elapsed()

    def eval(self, data_loader):
        self.nnet.eval()

        stats = AverageMeter()
        timer = SimpleTimer()

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                loss = self.compute_loss(egs)
                stats.add("loss", loss.item())

        return stats.value("loss"), stats.count("loss"), timer.elapsed()

    def run(self, train_loader, dev_loader, num_epochs=50):
        # using target device
        with th.cuda.device(self.gpu[0]):
            stats = dict()
            no_impr = 0
            # check if save is OK
            self.save_checkpoint(best=False)
            best_loss, _, _ = self.eval(dev_loader)
            self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
                self.cur_epoch, best_loss))
            while self.cur_epoch < num_epochs:
                stats[
                    "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                        self.optimizer.param_groups[0]["lr"],
                        self.cur_epoch + 1)
                tr_loss, tr_batch, tr_cost = self.train(train_loader)
                stats["tr"] = "train = {:+.4f}({:.2f}m/{:d})".format(
                    tr_loss, tr_cost / 60, tr_batch)
                cv_loss, cv_batch, cv_cost = self.eval(dev_loader)
                stats["cv"] = "dev = {:+.4f}({:.2f}m/{:d})".format(
                    cv_loss, cv_cost / 60, cv_batch)
                stats["scheduler"] = ""
                if cv_loss > best_loss:
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(
                        self.scheduler.best)
                    no_impr += 1
                else:
                    best_loss = cv_loss
                    no_impr = 0
                    self.save_checkpoint(best=True)
                self.logger.info(
                    "{title} {tr} | {cv} {scheduler}".format(**stats))
                # schedule here
                self.scheduler.step(cv_loss)
                # flush scheduler info
                sys.stdout.flush()
                # save checkpoint
                self.cur_epoch += 1
                self.save_checkpoint(best=False)
                if no_impr == self.no_impr:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    break
            self.logger.info(
                "Training for {} epoches done!".format(num_epochs))


class UpitTrainer(Trainer):
    """
    Trainer on uPIT loss
    """

    def __init__(self, *args, **kwargs):
        super(UpitTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, egs):
        """
        For permutation invariant trainer, egs contains:
            xlen: length of each utts
            feats: features that feed networks
            lx: abs(X)
            ly: [abs(Y1), abs(Y2) ...]
        """
        spk_masks = th.nn.parallel.data_parallel(
            self.nnet, egs["feats"], device_ids=self.gpu)

        num_utts = egs["xlen"].size(0)
        num_spks = len(egs["ly"])
        utt_xlen = egs["xlen"].type_as(egs["feats"])

        def loss(permute):
            loss_for_permute = []
            for s, t in enumerate(permute):
                ly = egs["ly"][t]
                lx = spk_masks[s] * egs["lx"]
                # N x T x F
                mat_loss = F.mse_loss(lx, ly, reduction="none")
                # N x 1
                utt_loss = th.sum(mat_loss, (1, 2))
                loss_for_permute.append(utt_loss)
            loss_perutt = sum(loss_for_permute) / utt_xlen
            return loss_perutt

        # O(N!), could be optimized
        # P x N
        pscore = th.stack([loss(p) for p in permutations(range(num_spks))])
        min_perutt, _ = th.min(pscore, dim=0)
        return th.sum(min_perutt) / (num_spks * num_utts)