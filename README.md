# NeuroZip

NeuroZip is a research prototype for shared-model neural lossless compression
of arbitrary byte streams. V0 follows [`docs/DESIGN.md`](docs/DESIGN.md): raw
bytes, a small causal GRU predictor, deterministic integer CDFs, and a forward
range coder.

The local implementation is intentionally dependency-light. The coder, file
format, uniform predictor, CLI, and tests run with Python's standard library.
PyTorch is needed only for training and for compressing with a trained GRU.
Actual model training is configured for Kaggle GPU in
[`notebooks/neurozip_v0_wikitext103_kaggle.ipynb`](notebooks/neurozip_v0_wikitext103_kaggle.ipynb).

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

Upload the repository as a Kaggle Dataset, attach it to a GPU notebook, and run
the notebook. It discovers the repository under `/kaggle/input`, copies it to
`/kaggle/working`, prepares deterministic raw WikiText-103 windows, and calls
the repository's `neurozip.train` module. The run exports `best.pt`,
`last.pt`, `metrics.jsonl`, `run_config.json`, and a zipped artifact bundle.

The trained `best.pt` can be downloaded and used locally with:

```bash
PYTHONPATH=src python -m neurozip compress input.txt output.nz --model best.pt
PYTHONPATH=src python -m neurozip decompress output.nz restored.txt --model best.pt
```

The model path is deliberately explicit: the compressed file stores the model
identifier and checksum, while the shared model itself is installed or supplied
separately.
