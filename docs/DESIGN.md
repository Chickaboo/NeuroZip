# NeuroZip: research and design plan

Status: research/design phase only; no main compressor has been implemented yet.

Decision date: 2026-08-14

## Executive decision

V0 will be a **shared-model, causal byte predictor** followed by an integer
range coder. The first learned predictor will be a small two-layer GRU with a
256-way byte output and approximately 3--4M parameters. It will be trained
with truncated backpropagation through time, but it will retain its recurrent
state while compressing a file. The model is a research baseline, not a claim
that GRUs are the eventual best architecture.

The first implementation will separate four choices that are often conflated:

1. **Source representation:** raw bytes, values 0--255. This handles arbitrary
   input and makes the lossless invariant unambiguous.
2. **Model representation:** continuous hidden state in the GRU. It is not a
   binary or ternary source representation.
3. **Model precision:** floating-point training and a deterministic CPU
   reference inference profile first; BF16/FP16/FP8/INT8/INT4/ternary/binary
   model variants are later experiments.
4. **Entropy coding:** binary integer range coding with an integer CDF. The
   fact that the final file is binary does not imply bit-level neural input.

The shared model is installed once and referenced by a model identifier in the
file. Model bytes will be reported separately and also amortized into an
effective BPB calculation. Per-file adaptation, online training, copy
matches, and larger sequence models are follow-up hypotheses rather than V0
assumptions.

## 1. What current work says

The basic causal construction is well established: a predictor assigns a
probability to the next symbol and an arithmetic/range coder turns those
probabilities into a lossless stream. DeepZip demonstrated this with an RNN
and explicitly called out the need for encoder/decoder symmetry and the cost
of storing model weights ([paper](https://arxiv.org/abs/1811.08162)). DZip
explored adaptive and semi-adaptive training, but also showed the practical
cost of training during compression ([paper](https://arxiv.org/abs/1911.03572)).

More recent work shows both the opportunity and the cost of scaling the
predictor:

- *Language Modeling Is Compression* established a useful two-part-code
  accounting rule: model bits count, and a large language model is not a fair
  compressor if its weights are omitted. It found that simple tokenizers can
  be surprisingly competitive and that compressing FP16 weights was not a
  free win ([ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/3cbf627fa24fb6cb576e04e689b9428b-Paper-Conference.pdf),
  [official code](https://github.com/google-deepmind/language_modeling_is_compression)).
- NNCP is an important practical reference: a compiled Transformer compressor
  reports enwik8 results around 1.19 bpb, comparable to CMIX in its published
  table, while emphasizing optimized inference rather than a research-only
  Python implementation ([project and results](https://www.bellard.org/nncp/)).
- LLMZip and FineZip show that large LMs, online memorization, and dynamic
  contexts can improve ratios, but their reported compute and model sizes are
  unsuitable as the first individual-researcher baseline
  ([LLMZip](https://arxiv.org/abs/2306.04050),
  [FineZip](https://arxiv.org/abs/2409.17141)).
- L3TC uses RWKV plus a specialized tokenizer and reports strong gains with
  substantially fewer parameters than some learned comparators
  ([AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/33446),
  [code](https://github.com/alipay/L3TC-leveraging-rwkv-for-learned-lossless-text-compression.git)).
- Two 2026 preprints are especially important prior art for future work. Nacrith
  combines a 135M Transformer with online n-gram/context predictors, adaptive
  biasing, and high-precision arithmetic coding
  ([paper](https://arxiv.org/abs/2602.19626)); StateSMix combines an online
  Mamba-style state model with sparse n-gram mixing in a small C/AVX2 system
  ([paper](https://arxiv.org/abs/2605.02904),
  [code](https://github.com/robtacconelli/StateSMix)). Their preprint numbers
  are useful hypotheses, not measurements reproduced here. Any future NeuroZip
  hybrid must distinguish itself from these mechanisms.

Byte-level modeling is also now a credible architectural direction. MEGABYTE
uses local byte models plus a global patch model for long contexts
([paper](https://arxiv.org/abs/2305.07185)); MambaByte applies a selective SSM
directly to bytes ([paper](https://arxiv.org/abs/2401.13660),
[code](https://github.com/jxiw/MambaByte)); and BLT learns dynamic byte patches
around a latent Transformer ([paper](https://arxiv.org/abs/2412.09871),
[code](https://github.com/facebookresearch/blt)). These motivate later
ablations, but their boundary logic and inference machinery are too much risk
for the first correctness-focused implementation.

## 2. Candidate representations and architecture families

| Candidate | Potential benefit | Main risk | Decision |
|---|---|---|---|
| Raw UTF-8 bytes | Universal, exact, 256-way alphabet, no tokenizer metadata | Longer sequence than text tokens | **V0** |
| Unicode code points | Natural linguistic units | Not byte-preserving without an encoding layer; invalid UTF-8 needs special handling | Later, only with a reversible byte wrapper |
| Bits | Fine probability resolution and simple binary alphabet | 8x sequence length and more recurrent steps | Later ablation |
| Binary/ternary radix source | Tests the radix hypothesis directly | Changes sequence length without a demonstrated information benefit | Do not assume; later ablation |
| Reversible subwords/BPE | Fewer prediction steps and larger semantic units | Vocabulary/CDF cost, rare strings, code/markup behavior, tokenizer metadata | Later reversible comparison |
| Learned/dynamic patches | Can allocate capacity to high-entropy spans; supported by BLT/MEGABYTE | Encoder and decoder must agree on boundaries; extra model and format state | Later |
| 256-way GRU/LSTM | Small, causal, portable, easy to stream and train in parallel over windows | Sequential inference and finite context | **First learned model** |
| Tiny causal Transformer | Stronger parallel training and explicit long context | KV/state cost and slower CPU decoding | First architecture ablation |
| RWKV/Mamba/other SSM | Long context with bounded recurrent state; promising recent results | Custom kernels, implementation complexity, prior-art overlap | Second-wave ablation |
| Hierarchical byte/patch model | Long context at manageable local sequence length | More moving parts and coding-boundary risk | Later |
| Neural plus copy/context mixer | May capture repetitions classical predictors find | Must prove complementarity rather than duplicate LZ/context mixing | Later hybrid experiment |

The initial model will use a byte embedding, two GRU layers of roughly 512
hidden units, and a linear 256-way output head. The training window can be
4096 bytes with truncated state for batching; compression itself carries state
from the beginning to the end of each file and resets at the file boundary.
The interface will expose `predict(state, previous_byte)` so the predictor can
later be replaced by a Transformer, SSM, patch model, or mixer without changing
the coder or file-format code.

## 3. Why GRU bytes are V0

This choice is deliberately falsifiable:

- It is the smallest model family that tests whether a reusable learned
  probability distribution exploits useful structure beyond an order-`n` byte
  model.
- A 256-symbol alphabet avoids the large-vocabulary CDF and tokenizer issues
  seen in larger language-model compressors.
- Raw bytes preserve arbitrary binary/text input exactly. Unicode normalization,
  capitalization, whitespace, line endings, nulls, and invalid UTF-8 are never
  silently changed.
- A recurrent state maps naturally to one causal probability per byte and a
  forward FIFO range coder. Training can still be batched over independent
  windows, which is practical on this machine.
- A portable reference implementation is easier to audit than a custom
  attention/SSM kernel. This reduces the chance that an apparent compression
  gain is actually a coder or synchronization bug.

This is not a claim that a GRU has the best BPB. Transformer, SSM, bit-level,
reversible-token, patch, and neural/classical-hybrid alternatives remain
explicit challenges to V0. A small neural model that does not beat a simple
byte n-gram on strict held-out data will be a useful negative result, not a
reason to hide the choice.

## 4. Entropy coding and deterministic decoding

### V0 coder

Use a 64-bit integer range coder with a forward/FIFO stream. Quantize each
model distribution to an integer CDF with total frequency `2^20` initially;
measure `2^16`, `2^20`, and `2^24` as a coder-precision ablation. The CDF
construction must have a specified deterministic tie-break rule, give every
byte at least one count, and satisfy:

```text
cdf[0] = 0
cdf[256] = total
cdf[i] < cdf[i+1]
```

The coder will consume only integer frequencies and integer boundaries. It
will never call a floating-point probability function. ANS/rANS is a later
throughput comparison; ANS is attractive for speed but has reverse/stack
ordering that complicates a first forward autoregressive implementation. The
[`constriction`](https://docs.rs/constriction/latest/constriction/index.html)
library is a candidate for a tested production implementation because it
provides range coding, ANS, fixed-point distributions, and Python/Rust APIs.

### Probability accounting

Every experiment will report all three quantities:

1. float-model cross entropy: `-sum(log2 p(symbol))`;
2. quantized-CDF expected cost, including finite-precision rounding;
3. actual range-stream bits, including termination/padding, then the complete
   file size including metadata and checksum.

The gap between these quantities is a direct test for probability quantization,
coder inefficiency, padding, and file-format mistakes. `payload BPB` excludes
the shared model; `effective BPB` adds the serialized model cost amortized over
the stated corpus or file size.

The benchmark will also report compressed bytes, compression ratio
`original_bytes / compressed_bytes`, percentage reduction
`100 * (1 - compressed_bytes / original_bytes)`, encode/decode throughput,
peak CPU RAM and GPU VRAM when applicable, parameter count, serialized model
bytes, training wall time/CPU-hours, metadata/checksum/padding bytes, and
entropy-coder overhead. Rates will be shown per domain, byte-weighted, and as a
macro-average so a large web corpus cannot hide a code or structured-text
regression.

### Model-precision plan

Do not quantize before the floating-point baseline is trusted. Then compare
FP32, BF16/FP16, INT8, INT4, and—only if the evidence justifies the added
engineering—FP8, ternary, and binary weight variants. Start with post-training
quantization for speed, then use quantization-aware training for the best
candidate. Measure BPB, model bytes, encode/decode throughput, peak RAM/VRAM,
reproducibility, and implementation complexity together. Quantized embeddings
and output heads must be reported separately because they can affect the
256-way probability distribution disproportionately. BitNet is relevant prior
art for ternary **weights**, but it is not evidence that a ternary source radix
will help.

### Determinism contract

V0 will support a named inference profile rather than pretending arbitrary
floating-point implementations are identical. The profile will specify:

- model artifact hash, architecture, weights, byte order, BOS behavior, and
  state-reset rule;
- deterministic CPU inference with dropout and randomness disabled;
- fixed operation order and no nondeterministic GPU/fused kernels;
- the exact CDF quantization, minimum-count, rounding, and tie-breaking rules;
- integer range-coder arithmetic, renormalization, termination, and padding;
- a decoder refusal path for unknown model/profile IDs.

Encoder and decoder will share golden probability/CDF vectors and bitstream
fixtures. A future portable profile should use fixed-point or integer neural
inference (and be tested across x86/ARM); until then, the reference binary is
the supported decoder for the V0 profile. Online adaptation is deferred until
its update state, rounding, and any transmitted information can be specified
in exact bits.

## 5. Shared model versus per-file model

V0 is shared-model compression. The file contains a compact model ID/hash and
the data stream, not the model weights. This makes small-file economics
unfavorable, so benchmark file sizes from 1 KiB through at least 100 MiB and
report the break-even size. The model registry will publish:

- serialized model bytes and format;
- parameter count and precision;
- model hash and inference profile;
- training data/version and intended domain;
- effective BPB at each evaluated file/corpus size.

Per-file training or online state is a separate experiment. Its accounting
must include every transmitted weight, update, seed, codebook, and adaptation
parameter. The recent DZip, FineZip, Nacrith, and StateSMix work shows why
adaptation can help, but it does not remove this accounting requirement.

The eventual minimal format is planned as:

```text
magic | version | model/profile id | flags/CDF precision |
original byte length | stream length | checksum | range-coded payload
```

The format will remain versioned and reject incompatible models. A raw/store
fallback for tiny or incompressible inputs is desirable, but should be added
only after the basic stream is correct.

## 6. Classical baselines

The primary conventional baselines will be real command-line outputs,
including their container overhead, with exact versions/settings recorded:

| Baseline | Initial setting |
|---|---|
| gzip/DEFLATE 1.14 | `gzip -9 -n -c` |
| Zstandard 1.5.7 | `zstd -19 -q -c`; also `zstd --ultra -22 -q -c` |
| Brotli | `brotli -q 11 -c`, record toolbox version and window setting |
| XZ/LZMA2 5.8.2 | `xz -9e -c` |
| 7-Zip/LZMA2 | `7z a -t7z -mx=9 -m0=lzma2 -mmt=1`, with archive timestamps disabled where supported |
| bzip2 | `bzip2 -9 -c` |

Single-thread settings will be used for throughput comparisons unless a
separate parallel-throughput table is clearly labeled. The slower context
mixing references CMIX v21, ZPAQ, and PAQ8PX are valuable high-ratio controls,
but CMIX recommends at least 32 GiB RAM and may not be practical on this
30-GiB machine. They will be optional, run on smaller fixtures first, and
never omitted from the report without saying why. CMIX's official project page
is [here](https://www.byronknoll.com/cmix.html); the Hutter Prize page is useful
for historical enwik9 context but is not a substitute for our reproducible
measurements ([Hutter Prize](https://hutter1.net/prize/index.htm)).

Neural comparison will distinguish three evidence classes: measurements
reproduced by NeuroZip, measurements reproduced from an official release such
as NNCP or LMIC, and numbers reported by authors that we did not reproduce.
Published systems will not be reimplemented wholesale merely to produce a
headline table; the first fair comparison is a common frozen corpus, common
file-size accounting, and the small byte n-gram/GRU controls.

## 7. Dataset research and licensing

The full FineWeb-family corpora are research sources, not download targets for
this workstation. The current cards report approximately:

| Dataset | Current scale and useful properties | Licensing/provenance |
|---|---|---|
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | More than 18.5T English web tokens; current card metadata is about 54.8 TB of Parquet and 52.45B rows. Official samples include roughly 10B tokens/27.6 GB, 100B/277.4 GB, and 350B/388 GB. Cleaned and deduplicated Common Crawl. | ODC-By 1.0 plus Common Crawl terms; review source terms before redistribution. |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | About 1.3T tokens for score >=3 and 5.4T for score >=2; education-quality scoring applied to FineWeb. Card metadata is about 5.84 TB and 3.50B rows, with small sample configurations. | ODC-By 1.0 plus Common Crawl terms; classifier quality is not the same as compression usefulness. |
| [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | Roughly 3T tokens, 475M documents, 1,733 language categories; English subset is about 1.19T tokens/1.71 TB. Documents are substantially longer, with extraction/OCR artifacts and some code switching. HF metadata is about 5.38 TB; the card's narrative estimate differs, so use the larger value for planning. | ODC-By 1.0 plus Common Crawl terms; PDF extraction provenance matters. |
| [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) | More than 350B educational PDF tokens in 69 languages; top 10% selected by a Qwen3-235B-A22B-Instruct-2507 quality score. It has tens of millions of rows but no equivalent small official sample, so stream and cap it. | ODC-By 1.0 plus Common Crawl terms. |
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | More than 3T multilingual words across 1,868 language-script pairs and about 20.2 TB; relevant only if multilingual compression becomes a goal. | ODC-By 1.0 plus Common Crawl terms. |
| [Dolma](https://huggingface.co/datasets/allenai/dolma) | About 3T tokens across web, academic text, code, books, and encyclopedic material. The v1.6 sample is about 16.4 GB and roughly 10B tokens, making it a practical later mixture source. | ODC-By 1.0 on the dataset card, with source-specific/third-party terms to audit. |
| [PG19](https://github.com/google-deepmind/pg19) | 28,602 train books, 50 validation, 100 test; about 1.97B train tokens. Long public-domain books with minimal preprocessing. | Public-domain source material; repository is Apache-2.0. |
| [CodeParrot GitHub code](https://huggingface.co/datasets/codeparrot/github-code) | About 115M files, 32 languages, roughly 1 TB uncompressed/300 GB compressed. Useful only after license filtering. | Heterogeneous repository licenses, not one blanket license; retain per-file license metadata. |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 161,443 annotated messages in 35 languages and more than 10k conversation trees; small enough for controlled conversational tests. | Apache-2.0 card; verify individual content policy before redistribution. |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | About 3.0 GB in the HF metadata; filtered, mostly English synthetic dialogue. | MIT card; treat as artificial conversational distribution, not natural-web evidence. |

The license labels above do not waive source-level obligations. The data
manifest will retain dataset revision, source/configuration, URL or document
identifier when available, and license/provenance fields. FinePDFs' published
pipeline is [here](https://github.com/huggingface/finepdfs); it requires much
more compute than this project should spend during sampling.

## 8. Sampling and mixture experiments

### Hardware-aware sampling plan

Use Hugging Face Parquet streaming and deterministic reservoir/priority
sampling rather than downloading full corpora. The HF streaming documentation
supports streaming Parquet row groups with bounded memory and disk
([docs](https://huggingface.co/docs/hub/en/datasets-streaming)). The initial
experiment budget is 100--250M **raw UTF-8 bytes per training mixture**, with a
10--25M-byte validation stream. Scale to 1B and then 1--10B bytes only if the
first result justifies it.

For every source:

1. pin the dataset revision/configuration and field;
2. hash each document deterministically and use source-stratified reservoir or
   priority sampling;
3. balance length buckets so short web pages do not dominate by row count;
4. cap individual documents while keeping their document boundary in one
   split; never split train/validation/test by contiguous chunks of one source
   document;
5. interleave sources by a byte budget, not by GPT/BPE token counts;
6. preserve raw text for audit and encode it as UTF-8 without normalization;
7. cache only the selected sample plus a manifest, not the upstream corpus.

At current card estimates, even FineWeb's official 10B-token sample is about
27.6 GB and the 100B sample is about 277.4 GB before any extracted/cache
expansion. The 350B sample is larger than the free disk budget. The first run
will therefore stream a small deterministic slice; it will not download any
of these samples in full. Dolma's 16.4-GB sample is a later, feasible local
cache, not a prerequisite.

### Candidate mixtures

All percentages below are **raw UTF-8 byte shares** under the same total byte
budget. They are hypotheses to measure, not assertions about data quality.

| ID | Mixture |
|---|---|
| M0 | 100% FineWeb |
| M1 | 100% FineWeb-Edu (score >=3) |
| M2 | 75% FineWeb + 25% FineWeb-Edu |
| M3 | 50% FineWeb + 25% FineWeb-Edu + 15% FinePDFs + 10% FinePDFs-Edu |
| M4 | 40% FineWeb + 20% FineWeb-Edu + 15% long-form PDF/books + 15% permissively licensed code/JSON/XML/config + 10% OASST/dialogue |
| M5 | Dolma source-balanced: 40% web, 20% academic/reference/books, 20% code/structured, 10% education, 10% dialogue |

The first training run uses M2 because it tests broad web versus educational
quality without adding PDF extraction or licensing confounders. Mixture
experiments then hold total bytes, architecture, optimizer, steps, seed, and
validation protocol constant. Selection is based on macro-average held-out
BPB, byte-weighted BPB, per-domain BPB, throughput, and confidence intervals;
not on a language-model benchmark score.

## 9. Fixed evaluation corpus

The benchmark will have a versioned manifest and immutable hashes. The suite
will report both per-domain and macro-average results:

1. **General English prose:** Canterbury `alice29`, `asyoulik`, plus a small
   public-domain prose set.
2. **Educational/technical prose:** Canterbury `lcet10`, held-out technical
   documents, and a separately sampled educational set.
3. **Long form:** PG19 validation/test books and selected long PDF-derived
   documents, keeping full-document boundaries.
4. **Source code:** Canterbury `fields.c` and `grammar.lsp`, plus a
   license-filtered held-out CodeParrot slice.
5. **Structured text:** Canterbury `cp.html`, JSON/XML/config/log fixtures,
   and selected structured documents from enwik9/Silesia.
6. **Conversational/informal:** held-out OASST1 and, separately labeled,
   UltraChat 200k conversations.
7. **Mixed domain:** a fixed bundle with equal document-domain quotas.
8. **Adversarial/sanity:** empty, one-byte, repetitive, random, null-byte,
   long-line, unusual-Unicode, and mixed text/binary fixtures. These test
   correctness and overhead, not natural-language claims.
9. **Post-training OOD:** a small document-level WET sample from the official
   current [Common Crawl latest crawl](https://commoncrawl.org/latest-crawl),
   such as `CC-MAIN-2026-30`, acquired only after the training manifest is
   frozen. This is the cleanest practical check against accidental reuse of
   FineWeb-era pages.

Canterbury is a useful legacy suite but is small and old; enwik8/enwik9 are
recognized compression references but have substantial contamination risk.
They remain labeled reference/legacy results, while the postdated and
document-level suites carry the main generalization claim. Canterbury's file
descriptions and sizes are documented [here](https://www.corpus.canterbury.ac.nz/descriptions/).

## 10. Contamination and leakage controls

- Split at document, book, URL, repository, or conversation-tree level. Never
  put pieces of one document in train and test.
- Exclude exact SHA-256 matches for every frozen benchmark and its known public
  mirrors from training manifests.
- Run cross-source near-duplicate checks using normalized-character n-gram
  MinHash/SimHash; record candidate overlaps and the removal threshold.
- Keep training source revisions and crawl dates explicit. The final OOD sample
  must be postdated relative to the pinned training snapshot, rather than
  merely being another random web sample.
- Do not train on PG19 validation/test, Canterbury, enwik8/enwik9, OASST test
  documents, or the CodeParrot evaluation repositories.
- Publish a manifest containing hashes, source IDs, split, crawl/dump date,
  byte count, and license/provenance. If a benchmark cannot be decontaminated,
  label it contaminated/reference rather than discarding the result silently.

## 11. Hardware and environment inspection

Inspection on 2026-08-14 found:

- Fedora Linux 44 Kinoite, an atomic OSTree host; no host package layering is
  planned.
- Existing `fedora-toolbox-44` container is available and `toolbox enter`
  works. Future compilers, Python environments, and native dependencies belong
  there.
- AMD Ryzen 9 7940HS, 8 cores/16 threads, AVX2 and AVX-512 flags; about 30 GiB
  RAM with roughly 20 GiB available during inspection and 8 GiB swap.
- About 341 GiB free on the project filesystem. This is enough for selected
  caches and checkpoints, not full FineWeb/FinePDF corpora.
- Integrated AMD Radeon 780M (`amdgpu`), approximately 1 GiB reported VRAM and
  shared-memory aperture; no NVIDIA/CUDA or ROCm toolchain was exposed.
  V0 is therefore CPU-first. GPU experiments require a separate hardware and
  software validation, not an assumption.
- Host/toolbox Python is 3.14.6. PyTorch, NumPy, Transformers, Datasets,
  Hugging Face Hub, pytest, and Hypothesis were not installed at inspection;
  no project environment or lockfile exists yet. Dependencies will be added
  inside the existing toolbox only after the lightweight coder design is
  settled.
- Available baseline tools include gzip 1.14, Zstandard 1.5.7, XZ 5.8.2, and
  7-Zip on the host; Brotli is available in the toolbox. Exact versions will be
  captured by the benchmark runner.

The repository currently has no usable Git history. Experiment records should
still reserve a `git_commit` field; initialize/version the project before the
first reproducible run rather than fabricating a commit ID.

## 12. Smallest falsifiable MVP

The MVP has two gates.

**Gate A: coder and invariant.** Implement a byte histogram/order-2 predictor,
the integer range coder, a minimal versioned stream, and an exact decoder.
Test empty files, every byte value, random/repetitive data, long lines, UTF-8,
nulls, truncation, corruption, and unknown model IDs. This gate can falsify
the file-format/coder assumptions without ML dependencies.

**Gate B: learned shared predictor.** Train the 2x512 GRU on a deterministic
100M-byte M2 stream, with a 10M-byte validation stream and a fixed seed. On the
frozen suite, compare:

- order-2 and order-4 byte predictors;
- the GRU's float cross entropy, quantized-CDF estimate, and actual range bits;
- gzip -9, Zstandard -19, Brotli q11, and XZ -9e;
- payload BPB and effective BPB at each file size;
- encode/decode speed, peak RSS, model size, and exact round-trip behavior.

The first benchmark should sweep only CDF precision (`2^16`, `2^20`, `2^24`)
and one small model-size control. Do not mix architecture, data mixture, and
quantization changes in this run.

## 13. Experiment record

Each JSONL/CSV row must include at least:

```text
experiment_id, git_commit, config_hash, seed, dataset_revision,
dataset_manifest_hash, mixture, train_bytes, validation_bytes, steps,
architecture, parameters, model_precision, cdf_precision,
validation_loss, predicted_float_bpb, predicted_quantized_bpb,
actual_payload_bpb, actual_effective_bpb, compressed_bytes,
encode_bytes_per_second, decode_bytes_per_second, peak_rss_bytes,
peak_vram_bytes, model_bytes, metadata_bytes, coder_overhead_bits,
train_time_seconds, train_cpu_hours, notes
```

Predicted and actual rates must never be collapsed into one “BPB” field.

## 14. Expected risks

- **Coder desynchronization:** a CDF rounding, state-update, or termination
  mismatch makes the entire stream undecodable. Golden vectors and corrupted
  stream tests are a release gate.
- **Floating-point drift:** encoder and decoder can disagree across kernels or
  platforms. V0 therefore has a named CPU profile; portable integer inference
  is a follow-up requirement, not an untested promise.
- **Model overhead:** a shared model can lose on small files even when payload
  BPB is good. Effective BPB and break-even file size must be first-class
  results.
- **Sequential speed:** byte-by-byte GRU inference may be too slow. Measure it
  before increasing model size; then test a Transformer/SSM or batching strategy
  rather than assuming more parameters solve the problem.
- **Distribution mismatch:** a clean educational or PDF mixture may hurt code,
  markup, informal text, or ordinary web text. Report every domain and use
  macro averages.
- **Contamination:** FineWeb/Common Crawl overlap with legacy corpora can make
  results look much better than true OOD performance. Hash, near-deduplicate,
  date-separate, and retain manifests.
- **Quantization damage:** lower precision can worsen log loss even if model
  storage and arithmetic get cheaper. Treat precision as an experiment.
- **Classical ceiling:** Zstandard, LZMA2, CMIX, and PAQ-style context mixing
  may remain better on important domains. A neural gain is still useful only
  if its compute/model-size tradeoff is clear.
- **Licensing and extraction artifacts:** PDF/code data can carry source-level
  terms, OCR errors, and duplicated boilerplate. Keep provenance and do not
  silently redistribute unfiltered caches.

## 15. Scaling and stopping criteria

Scale the project beyond the MVP only if all of the following are true:

1. Encoder and decoder pass deterministic golden vectors and property/fuzz
   tests, including rejection of corrupted or incompatible streams.
2. The learned predictor improves actual payload BPB over the order-`n` byte
   control on at least three benchmark domains, with the improvement surviving
   a second seed and the postdated OOD set.
3. The coder gap from quantized expected cost to actual payload is small and
   explained (initial target: below 0.05--0.10 bpb on medium/large files).
4. The shared model is economically plausible for the intended file sizes;
   model bytes, metadata, and checksum are reported rather than hidden.
5. CPU throughput and memory are useful for an individual workstation. An
   initial target is at least tens of KiB/s for both encode and decode with a
   model under roughly 20 MB, while recording the actual tradeoff rather than
   treating these targets as scientific laws.

If V0 fails, the next controlled tests are a tiny Transformer and an SSM
predictor. If both fail against the byte n-gram control on strict OOD data,
stop scaling model size and investigate representation, data mixture, or a
classical/neural context mixer instead. If gains occur only on contaminated
legacy corpora, do not scale. If a small model gives a reliable OOD gain but
does not beat Zstandard or CMIX, the project can still be worthwhile as a
compression-to-compute study; the claim must then be framed accurately.

## 16. Planned repository boundaries

The implementation should grow around these replaceable boundaries:

```text
src/neurozip/data          streaming, sampling, manifests, deduplication
src/neurozip/representation byte/bit/token interfaces
src/neurozip/models        GRU, Transformer, SSM, patch, mixer predictors
src/neurozip/coding        CDF quantization, range coder, later ANS
src/neurozip/format        versioned header, checksum, model registry
src/neurozip/compress      encoder orchestration
src/neurozip/decompress    decoder orchestration
src/neurozip/eval          BPB, predicted-vs-actual, corpus manifests
src/neurozip/bench         classical/neural timing and memory runners
tests                      round-trip, corruption, deterministic fixtures
configs                    pinned experiment configurations
docs                       research log, design decisions, result reports
```

The eventual CLI should stay small: `neurozip compress INPUT OUTPUT`,
`neurozip decompress INPUT OUTPUT`, `neurozip benchmark ...`, and an inspection
command for model/profile and stream metadata. CLI polish is deliberately
after coder correctness and the first falsifiable benchmark.

No architecture-specific implementation should be allowed to leak into the
coder or file format. That is the main protection against preserving a bad
initial hypothesis.
