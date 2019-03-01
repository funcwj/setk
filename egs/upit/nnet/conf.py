# model config
num_bins = 257
feats_dim = num_bins
num_spks = 2

lstm_conf = {
    "rnn": "lstm",
    "num_layers": 3,
    "hidden_size": 896,
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
    "clip_norm": 100,
    "min_lr": 1e-8,
    "patience": 0,
    "factor": 0.5,
    "logging_period": 200  # batch number
}

# feature config
feats_conf = {
    "apply_log": False,
    "norm_means": True,
    "norm_vars": True,
    "cmvn": "data/2spk/train/mix/gcmvn.mat"
}

train_dir = "data/2spk/train/"
dev_dir = "data/2spk/dev/"

train_data = {
    "linear_x": train_dir + "mix/feats.scp",
    "linear_y": [train_dir + "spk1/mask.scp", train_dir + "spk2/mask.scp"],
}

dev_data = {
    "linear_x": dev_dir + "mix/feats.scp",
    "linear_y": [dev_dir + "spk1/mask.scp", dev_dir + "spk2/mask.scp"],
}
