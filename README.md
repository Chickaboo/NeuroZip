# NeuroZip

NeuroZip is a research prototype for shared-model neural lossless compression
of arbitrary byte streams. The original V0 baseline follows
[`docs/DESIGN.md`](docs/DESIGN.md): raw bytes, a small causal GRU predictor,
deterministic integer CDFs, and a forward range coder. The experiment layer now
supports a matched-budget architecture sweep with GRU, LSTM, causal
Transformer, Mamba-Lite, Griffin-Lite, Gated DeltaNet-Lite, and Gated
DeltaNet-2-Lite predictors.

The local implementation is intentionally dependency-light. The coder, file
format, uniform predictor, CLI, and tests run with Python's standard library.
PyTorch is needed only for training and for compressing with a trained neural
model. Actual model training is configured for Kaggle GPUs in
[`notebooks/neurozip_v0_wikitext103_kaggle.ipynb`](notebooks/neurozip_v0_wikitext103_kaggle.ipynb).
The notebook clones the public GitHub repository before running the training
code, so no uploaded dataset copy is required.

## Local smoke test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
printf 'hello NeuroZip\n' > /tmp/neurozip-input.txt
PYTHONPATH=src python -m neurozip compress /tmp/neurozip-input.txt /tmp/neurozip-output.nz --predictor uniform
PYTHONPATH=src python -m neurozip decompress /tmp/neurozip-output.nz /tmp/neurozip-restored.txt
cmp /tmp/neurozip-input.txt /tmp/neurozip-restored.txt
```

For an installed editable checkout, use `pip install -e .` inside the existing
Toolbx environment. Do not layer development packages onto the Atomic Fedora
host.

## Kaggle training

Open the notebook in Kaggle with both GPUs and Internet access enabled. It
clones the public repository into `/kaggle/working`, prepares deterministic raw
WikiText-103 windows, and launches one PyTorch DistributedDataParallel process
per visible GPU. Checkpoints and metrics are written only by rank 0. The run
exports `best.pt`, `last.pt`, `metrics.jsonl`, `run_config.json`, and
`summary.json` under the run artifact directory.

The configured batch size is per GPU. With two T4s, `batch_size: 32` gives an
effective batch size of 64; the effective size is recorded in `run_config.json`.

The default preparation downloads a public mirror of the WikiText-103 raw
archive and uses deterministic 50 MiB training and 5 MiB validation windows.

The trained `best.pt` can be downloaded and used locally with any supported
architecture:

```bash
PYTHONPATH=src python -m neurozip compress input.txt output.nz --model best.pt
PYTHONPATH=src python -m neurozip decompress output.nz restored.txt --model best.pt
```

The model path is deliberately explicit: the compressed file stores the model
identifier and checksum, while the shared model itself is installed or supplied
separately.

## Architecture comparison

The Kaggle notebook reads
[`configs/architecture_sweep_wikitext103_kaggle.json`](configs/architecture_sweep_wikitext103_kaggle.json).
Every row uses the same raw-byte representation, WikiText-103 split, 2,000
training steps, per-GPU batch size, sequence length, CDF precision, and
NeuroZip range coder. Candidate configurations are within a small parameter
band around the 3M-parameter GRU baseline; exact counts are printed before
training and retained in each `run_config.json`.

After training, the notebook runs
`python -m neurozip.experiments.architecture_benchmark` on a deterministic
held-out prefix of `validation.raw`. The benchmark records training and
validation loss/BPB, model/checkpoint size, parameter count, training and
codec throughput, peak memory, full-stream and payload BPB, compression ratio,
generation samples, and both byte and SHA-256 equality. A failed exact round
trip is marked `FAILED` and cannot win the recommendation. The generated
`comparison.md`, `comparison.csv`, and `comparison.json` are archived with the
Kaggle run.

Mamba, Griffin, and Gated DeltaNet entries are intentionally labeled `-Lite`:
they are small pure-PyTorch reference variants with explicit streaming state,
not claims of optimized reproductions of the full research CUDA kernels. This
keeps the comparison runnable and makes the implementation boundary visible in
the speed tradeoff.
