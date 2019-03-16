# model config
num_bins = 257
feats_dim = num_bins
num_spks = 2

lstm_conf = {
    "rnn": "lstm",
    "num_layers": 3,
    "hidden_size": 600,
    "dropout": 0.5,
    "bidirectional": True
}

nnet_conf = {
    "feats_dim": feats_dim,
    "num_bins": num_bins,
    "num_spks": num_spks,
    "rnn_conf": lstm_conf,
    "non_linear": "relu",
    "dropout": 0.0
}

# trainer config
adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "logging_period": 200,  # batch number
    "clip_norm": 10,
    "min_lr": 1e-8,
    "patience": 1,
    "factor": 0.5,
    "objf": "L2"
}

train_dir = "data/2spk/train/"
dev_dir = "data/2spk/dev/"

# feature config
feats_conf = {
    "apply_log": False,
    "norm_means": True,
    "norm_vars": True,
    # linear cmvn here if apply_log = False
    "cmvn": train_dir + "mix/gcmvn.mat"
}

# PSA: mask.scp, MA: feats.scp
train_data = {
    "linear_x":
    train_dir + "mix/feats.scp",
    "linear_y":
    [train_dir + "spk{:d}/mask.scp".format(n) for n in range(1, 1 + num_spks)],
}

dev_data = {
    "linear_x":
    dev_dir + "mix/feats.scp",
    "linear_y":
    [dev_dir + "spk{:d}/mask.scp".format(n) for n in range(1, 1 + num_spks)],
}
