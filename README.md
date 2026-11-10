# MRIEntropyMetrics

MRIEntropyMetrics calculates univariate multiscale sample entropy (MSE) for
each ROI and bivariate multiscale sample entropy (bMSE) for every ROI pair in a
preprocessed 4D fMRI image. It accepts a single NIfTI image or a BIDS-like
directory containing multiple subjects, sessions, and runs.

The tool does **not** perform preprocessing, temporal filtering, confound
regression, scrubbing, registration, or mask resampling. The supplied masks
must already be coregistered to the fMRI data.

## Installation

MRIEntropyMetrics requires Python 3.10 or newer.

```bash
git clone https://github.com/brain-complexity/MRIEntropyMetrics.git
cd MRIEntropyMetrics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy>=1.24,<2' 'pandas>=2.0' 'nibabel>=5.0' \
  'scipy>=1.10' 'joblib>=1.3' 'EntropyHub>=2.0'
```

`run_mse_bmse.sh` checks the runtime dependencies, limits BLAS/OpenMP threads,
and starts `mse_bmse_pipeline.py` with the selected Python interpreter. No
package installation is required after installing the dependencies above.

## Input requirements

- One or more preprocessed 4D `.nii` or `.nii.gz` fMRI images.
- A directory containing at least two 3D binary `.nii`/`.nii.gz` ROI masks.
- Masks and fMRI must have matching spatial shape, affine, and voxel sizes.
- Every ROI must contain at least one voxel and yield a finite, non-constant
  time series.

Mask filenames become ROI names in the output. A binary mask is selected with
`mask > 0.5`. Non-binary masks produce a warning, or an error with `--strict`.
No automatic resampling is performed.

## Quick start

### One fMRI image

Using the Bash wrapper from a clone:

```bash
./run_mse_bmse.sh \
  --fmri /data/sub-01_ses-01_task-rest_bold.nii.gz \
  --masks /data/masks \
  --output /data/results
```

For a non-BIDS filename, output identifiers can be supplied explicitly:

```bash
./run_mse_bmse.sh \
  --fmri /data/resting_state.nii.gz \
  --masks /data/masks \
  --output /data/results \
  --subject-id 01 \
  --session-id baseline \
  --run-id rest
```

### Multiple subjects and sessions

The default directory pattern is
`sub-*/ses-*/func/*_bold.nii.gz`:

```bash
./run_mse_bmse.sh \
  --input-root /data/study \
  --masks /data/masks \
  --output /data/results
```

Use `--pattern` for another layout. Quote the pattern so the shell does not
expand it:

```bash
./run_mse_bmse.sh \
  --input-root /data/study \
  --pattern 'subjects/*/sessions/*/func/*.nii.gz' \
  --masks /data/masks \
  --output /data/results
```

### Validate before calculating

```bash
./run_mse_bmse.sh \
  --input-root /data/study \
  --masks /data/masks \
  --output /data/results \
  --dry-run
```

`--dry-run` discovers files and checks NIfTI headers, mask geometry, and basic
analysis constraints without calculating metrics or creating output files.

## Parameters

The defaults follow the focused 7T reference workflow on which this tool is
based:

| Option | Meaning | Default |
|---|---|---:|
| `--m` | Embedding dimension for MSE and bMSE | `2` |
| `--mse-r` | MSE tolerance factor multiplied by ROI SD | `0.3` |
| `--bmse-r` | bMSE tolerance with EntropyHub normalization | `0.6` |
| `--a` | Number of consecutive scales, starting at 1 | `10` |
| `--jobs` | Parallel Joblib workers | `HCP_N_JOBS` or `6` |

`--r VALUE` sets the same tolerance value for both methods. It cannot be
combined with `--mse-r` or `--bmse-r`.

Useful execution options:

| Option | Effect |
|---|---|
| `--save-timeseries` | Write the extracted ROI-by-time table |
| `--strict` | Reject non-binary masks instead of warning |
| `--force` | Replace generated files for conflicting/incomplete runs |
| `--fail-fast` | Stop a batch after its first failed run |
| `--python PATH` | Select Python for the Bash wrapper |

Run `./run_mse_bmse.sh --help` for the complete command reference.

## Calculations

For each binary ROI \(R_i\), the input signal is the arithmetic mean across
all selected BOLD voxels at each time point. Actual zero-valued BOLD samples
remain valid observations.

MSE uses EntropyHub `MSEn` with `SampEn`. Its tolerance is
`mse_r × SD(ROI time series)`. bMSE is calculated for every unordered ROI pair
using EntropyHub `cMvMSEn` with `MvSampEn`, `Refined=True`, and `Norm=True`.
The complexity index (CI) is the trapezoidal integral across finite scale
values.

## Output

```text
results/
└── sub-01/
    └── ses-01/
        └── task-rest_run-1/
            ├── mse.csv
            ├── bmse_a1.csv
            ├── ...
            ├── bmse_a10.csv
            ├── bmse_CI.csv
            ├── run_manifest.json
            └── roi_timeseries.csv  # only with --save-timeseries
```

`mse.csv` contains one row per ROI and columns `a1..aN,CI`.
`bmse_a*.csv` and `bmse_CI.csv` are symmetric ROI-by-ROI matrices. Their
diagonal is `NaN`, because self-pairs are not calculated.

The JSON manifest records input fingerprints, image geometry, TR, ROI voxel
counts, effective parameters, package versions, timestamps, and warnings.
Output data are staged in a temporary run directory and the success manifest
is published last.

Complete runs with identical inputs and parameters are skipped. If existing
output is incomplete or incompatible, the tool stops unless `--force` is used.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | All selected runs succeeded or were already complete |
| `2` | Invalid command-line arguments |
| `3` | Invalid input, geometry, or conflicting output |
| `4` | At least one numerical run failed |
| `5` | Python interpreter or dependency missing (Bash wrapper) |

## Citation

If you use MRIEntropyMetrics in research, please cite this software repository
and the methods/software underlying your analysis.

The entropy calculations use [EntropyHub](https://github.com/MattWillFlood/EntropyHub).
Please also cite its methods paper:

Flood, M. W., & Grimm, B. (2021). EntropyHub: An open-source toolkit for
entropic time series analysis. *PLOS ONE*, 16(11), e0259448.
[https://doi.org/10.1371/journal.pone.0259448](https://doi.org/10.1371/journal.pone.0259448)

## License

MRIEntropyMetrics is licensed under the [MIT License](LICENSE).

EntropyHub is a separate dependency licensed under
[Apache-2.0](https://github.com/MattWillFlood/EntropyHub/blob/main/LICENSE).
The MIT License for MRIEntropyMetrics does not replace the licenses of its
dependencies.
