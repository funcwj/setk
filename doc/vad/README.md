## VAD

Removing the silence from the given utterance.

### Cmd options

See `./scripts/sptk/do_vad.py -h`

### Usage

```bash
# - means stdin
echo "utt asset/utt.wav" |  ../../scripts/sptk/do_vad.py --mode 3 --fs 16000 - vad
```
The processed audio are generated under directory `vad`.

### Dependency

* [webrtcvad](https://github.com/wiseman/py-webrtcvad)
