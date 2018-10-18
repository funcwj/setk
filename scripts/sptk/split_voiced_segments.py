#!/usr/bin/env python
# wujian@2018

import os
import argparse
import numpy as np

from libs.data_handler import WaveReader
from libs.utils import write_wav, get_logger, EPSILON

logger = get_logger(__name__)


class VoiceSpliter(object):
    def __init__(self, energy_threshold, tolerated_size):
        self.energy_threshold = energy_threshold
        self.tolerated_size = tolerated_size
        self.reset()

    def run(self, frame):
        if self.cur_step < 0:
            raise RuntimeError(
                "Seems bugs existed in VoiceSpliter's implementation")
        frame_length = frame.size
        # compute average energy per-frames
        energy = 20 * np.sum(np.log10(np.abs(frame) + EPSILON)) / frame_length
        # logger.info("Frame-ID/Energy: {}/{:.2f}".format(self.cur_frame, energy))
        if energy >= self.energy_threshold:
            if not self.voiced:
                self.cur_step += 1
                if self.cur_step == self.tolerated_size:
                    self.voiced = True
                    self.segments.append(self.cur_frame - self.tolerated_size + 1)
        else:
            if self.cur_step:
                self.cur_step -= 1
                if self.voiced and self.cur_step == 0:
                    self.voiced = False
                    self.segments.append(self.cur_frame)
        self.cur_frame += 1

    def reset(self):
        self.cur_step = 0
        self.cur_frame = 0
        self.voiced = False
        self.segments = []


def run(args):
    voice_spliter = VoiceSpliter(args.voiced_threshold, args.tolerated_size)
    wave_reader = WaveReader(args.wav_scp)
    L, S = args.frame_length, args.frame_shift
    samp_rate = args.sample_rate
    for key, wave in wave_reader:
        voice_spliter.reset()
        num_frames = (wave.size - L) // S + 1
        for idx in range(num_frames):
            voice_spliter.run(wave[idx * S:idx * S + L])
        segments = voice_spliter.segments
        if len(segments) % 2:
            segments.append(num_frames)
        logger.info("{} segments: {}".format(key, segments))
        for idx in range(len(segments) // 2):
            beg, end = segments[idx * 2:idx * 2 + 2]
            if (end - beg) * S / samp_rate < args.min_dur:
                continue
            voiced_segement = wave[beg * S:end * S]
            write_wav(
                os.path.join(args.dump_dir, "{}-{:d}-{:d}.wav".format(
                    key, beg, end)), voiced_segement, samp_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Command to split voiced segments. This command using a simple method based on energy threshold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "wav_scp", type=str, help="Input wave scripts in kaldi format")
    parser.add_argument(
        "--voiced-threshold",
        type=float,
        dest="voiced_threshold",
        default=-40.0,
        help="Energy threshold value to distinguish voiced/silenced frames(in dB)")
    parser.add_argument(
        "--frame-length",
        type=int,
        dest="frame_length",
        default=512,
        help="Number of samples per frames")
    parser.add_argument(
        "--frame-shift",
        type=int,
        dest="frame_shift",
        default=256,
        help="Number of samples for frame shift")
    parser.add_argument(
        "--dump-dir",
        type=str,
        dest="dump_dir",
        default="noise",
        help="Location to dump voiced segments")
    parser.add_argument(
        "--tolerated-size",
        type=int,
        dest="tolerated_size",
        default=10,
        help="Maximum number of frames which could be tolerated by algorithm")
    parser.add_argument(
        "--min-duration",
        type=float,
        dest="min_dur",
        default=1,
        help="Write segments out only if duration longer than this value")
    parser.add_argument(
        "--sample-rate",
        type=int,
        dest="sample_rate",
        default=16000,
        help="Sample rate of input wave files")
    args = parser.parse_args()
    run(args)
