#!/usr/bin/env python

# wujian@2018

import sys
import types

import argparse
import numpy as np

from pathlib import Path
from urllib import request

from libs.cluster import CacgmmTrainer
from libs.data_handler import SpectrogramReader, ScriptReader, NumpyReader, NumpyWriter
from libs.utils import get_logger, nextpow2
from libs.opts import StftParser, StrToBoolAction

pb_bss_align_url = "https://raw.githubusercontent.com/fgnt/pb_bss/master/pb_bss/permutation_alignment.py"
pb_bss_align_loc = "$SETK_ROOT/scripts/sptk/pb_perm_solver.py"

try:
    import pb_perm_solver
except ImportError:
    raise RuntimeError(
        "Import pb_perm_solver error\n " +
        f"Please download {pb_bss_align_url} to {pb_bss_align_loc}")

logger = get_logger(__name__)


def load_module(url):
    """
    Load module from url (simplest way)
    https://python3-cookbook.readthedocs.io/zh_CN/latest/c10/p11_load_modules_from_remote_machine_by_hooks.html
    """
    u = request.urlopen(url)
    source = u.read().decode("utf-8")
    mod = sys.modules.setdefault(url, types.ModuleType(url))
    code = compile(source, url, "exec")
    mod.__file__ = url
    mod.__package__ = ""
    exec(code, mod.__dict__)
    return mod


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center": args.center,
        "transpose": False
    }

    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    MaskReader = {"numpy": NumpyReader, "kaldi": ScriptReader}
    init_mask_reader = MaskReader[args.fmt](
        args.init_mask) if args.init_mask else None

    n_fft = nextpow2(
        args.frame_len) if args.round_power_of_two else args.frame_len
    # now use pb_bss
    # pb_perm_solver = load_module(pb_bss_align_url)
    aligner = pb_perm_solver.DHTVPermutationAlignment.from_stft_size(n_fft)

    num_done = 0
    with NumpyWriter(args.dst_dir) as writer:
        dst_dir = Path(args.dst_dir)
        for key, stft in spectrogram_reader:
            if not (dst_dir / f"{key}.npy").exists():
                # K x F x T
                init_mask = None
                if init_mask_reader and key in init_mask_reader:
                    init_mask = init_mask_reader[key]
                    logger.info("Using external mask to initialize cacgmm")
                # stft: N x F x T
                trainer = CacgmmTrainer(stft,
                                        args.num_classes,
                                        gamma=init_mask,
                                        cgmm_init=args.cgmm_init)
                try:
                    # EM progress
                    masks = trainer.train(args.num_epoches)
                    # align if needed
                    if not args.cgmm_init or args.num_classes != 2:
                        masks = aligner(masks)
                        logger.info(
                            "Permutation align done for each frequency")
                    num_done += 1
                    writer.write(key, masks.astype(np.float32))
                    logger.info(f"Training utterance {key} ... Done")
                except np.linalg.LinAlgError:
                    logger.warn(f"Training utterance {key} ... Failed")
            else:
                logger.info(f"Training utterance {key} ... Skip")
    logger.info(
        f"Train {num_done:d} utterances over {len(spectrogram_reader):d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speaker masks estimation using Complex Angular "
        "Central Gaussian Mixture Model (CACGMM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-channel wave scripts in kaldi format")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Where to dump estimated speech masks")
    parser.add_argument("--num-epoches",
                        type=int,
                        default=50,
                        help="Number of epoches to train Cacgmm")
    parser.add_argument("--num-classes",
                        type=int,
                        default=2,
                        help="Number of the cluster "
                        "used in cacgmm model")
    parser.add_argument("--init-mask",
                        type=str,
                        default="",
                        dest="init_mask",
                        help="Mask scripts for cacgmm initialization")
    parser.add_argument("--cgmm-init",
                        action=StrToBoolAction,
                        default=False,
                        help="For 2 classes, using the cgmm init way")
    parser.add_argument("--mask-format",
                        type=str,
                        dest="fmt",
                        default="numpy",
                        choices=["kaldi", "numpy"],
                        help="Mask storage format")
    args = parser.parse_args()
    run(args)