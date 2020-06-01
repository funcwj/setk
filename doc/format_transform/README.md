## Data Format

Format tranform between Kaldi, Numpy and Matlab.

### Cmd options

See `./scripts/sptk/copy_mat_to_archive.py -h` and `./scripts/sptk/copy_archive_to_mat.py`

### Usage

```bash
# .ark to .npy
../../scripts/sptk/copy_archive_to_mat.py \
    --src-format ark \
    --dst-format npy \
    asset/egs.ark npy
# .ark to .mat
../../scripts/sptk/copy_archive_to_mat.py \
    --src-format ark \
    --dst-format mat \
    asset/egs.ark mat
# .npy to .ark
find npy -name "*.npy" | awk -F '[/.]' '{print $2"."$3"\t"$0}' \
    | ../../scripts/sptk/copy_mat_to_archive.py \
    --src npy - npy.ark
# .mat to .ark
find mat -name "*.mat" | awk -F '[/.]' '{print $2"."$3"\t"$0}' \
    | ../../scripts/sptk/copy_mat_to_archive.py \
    --src mat - mat.ark
```