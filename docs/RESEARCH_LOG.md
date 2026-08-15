# NeuroZip research log

Access date for this initial log: 2026-08-14. “Influence” records whether a
source changed the V0 decision, is a planned ablation, or is only background.
Reported numbers from preprints are marked as author claims and have not been
reproduced in this repository.

| Source | Major idea | Relevance | Influence |
|---|---|---|---|
| [DeepZip](https://arxiv.org/abs/1811.08162) | RNN next-symbol probabilities plus arithmetic coding; exact causal symmetry; model storage matters. | Foundational neural lossless construction. | **V0:** causal predictor, shared-model accounting, deterministic mirror. |
| [DZip](https://arxiv.org/abs/1911.03572) | Adaptive/semi-adaptive neural compression without pretraining data. | Shows online training can help but costs time and state. | Per-file/adaptive follow-up; not V0. |
| [Language Modeling Is Compression](https://proceedings.iclr.cc/paper_files/paper/2024/file/3cbf627fa24fb6cb576e04e689b9428b-Paper-Conference.pdf) | Two-part model/data accounting; autoregressive arithmetic coding; tokenization and model-size tradeoffs. | Prevents misleading model-free comparisons. | **V0:** effective BPB and simple reversible representation first. |
| [Official LMIC code](https://github.com/google-deepmind/language_modeling_is_compression) | Reproducible enwik8 language-model compression experiments. | Future reproduction/control. | Benchmark infrastructure reference. |
| [NNCP](https://www.bellard.org/nncp/) | Optimized Transformer compressor with strong enwik8/enwik9 results. | Practical learned-compressor reference. | Comparison target; not copied as V0. |
| [LLMZip](https://arxiv.org/abs/2306.04050) | LLaMA-scale predictor with entropy coding. | Demonstrates large-model ratio/latency/model-overhead tradeoff. | Reject large LM for MVP. |
| [FineZip](https://arxiv.org/abs/2409.17141) | LLM plus online memorization and dynamic context. | Important adaptive prior art. | Future adaptation comparison; no novelty claim. |
| [L3TC](https://ojs.aaai.org/index.php/AAAI/article/view/33446) | RWKV, outlier-aware tokenizer, high-rank reparameterization. | Strong small(er) learned text-compression precedent. | SSM/RWKV ablation and prior-art boundary. |
| [Nacrith](https://arxiv.org/abs/2602.19626) | 135M Transformer plus n-gram predictors, adaptive bias head, high-precision CDF, hybrid format. | Closely resembles likely future hybrid design. | Explicitly reserved as prior art; do not claim these mechanisms as novel. |
| [StateSMix](https://arxiv.org/abs/2605.02904) | Mamba-style online state model plus sparse n-gram mixing in C/AVX2. | Relevant low-memory adaptive comparator. | Reproduce later; V0 remains shared/offline. |
| [MEGABYTE](https://arxiv.org/abs/2305.07185) | Hierarchical local byte/global patch Transformer. | Candidate long-context architecture. | Later patch ablation. |
| [MambaByte](https://arxiv.org/abs/2401.13660) | Token-free selective SSM over raw bytes. | Candidate efficient long-context byte model. | Later SSM ablation. |
| [BLT](https://arxiv.org/abs/2412.09871) | Dynamic entropy-based byte patches with local encoders/decoders and latent Transformer. | Candidate learned segmentation. | Later; too complex for V0. |
| [ByT5](https://arxiv.org/abs/2105.13626) | Direct UTF-8 byte modeling without a tokenizer. | Representation evidence. | Supports byte baseline, not a compression design by itself. |
| [BitNet](https://arxiv.org/abs/2504.12285) and [bitnet.cpp](https://arxiv.org/abs/2502.11880) | Ternary weights and efficient integer inference. | Distinguishes model quantization from source radix. | Quantization ablation after FP baseline. |
| [Constriction](https://docs.rs/constriction/latest/constriction/index.html) | Tested range/ANS coders and fixed-point distributions. | Candidate production coder and reference. | **V0:** integer range-coder design; later library comparison. |
| [ANS original work](https://arxiv.org/abs/0902.0271) | Asymmetric numeral systems approach arithmetic-coding rates. | Candidate alternative entropy coder. | Later rANS throughput/ordering ablation. |
| [torchac documentation](https://pypi.org/project/torchac/) | Finite-precision CDF monotonicity and quantization constraints. | Practical warning about probability-to-CDF conversion. | **V0:** explicit CDF invariants and precision sweep. |
| [CMIX](https://www.byronknoll.com/cmix.html) | High-ratio context mixing with substantial CPU/RAM cost. | Strong classical/text-specific reference. | Required optional baseline; hardware-limited. |
| [ZPAQ documentation](https://manpages.debian.org/unstable/zpaq/zpaq.1.en.html) | Adaptive context models, mixing, SSE, arithmetic coding, and LZ/BWT components. | Classical hybrid/context-mixing reference. | Later baseline if available. |
| [gzip manual](https://www.gnu.org/software/gzip/manual/gzip.html) | Current DEFLATE command/settings and default behavior. | Reproducible conventional baseline. | `gzip -9 -n -c`. |
| [Zstandard CLI documentation](https://chromium.googlesource.com/external/github.com/facebook/zstd/+/refs/heads/upstream/cmake_root/programs/README.md) | Compression levels and benchmark/CLI behavior. | Fast/strong modern baseline. | `-19` and `--ultra -22`; record version. |
| [Brotli CLI](https://github.com/google/brotli/blob/master/c/tools/brotli.md) | Quality 0--11, window settings, RFC 7932 stream. | Strong text/web baseline. | `-q 11`, record toolbox version. |
| [7-Zip SDK](https://www.7-zip.org/sdk.html) | LZMA/LZMA2/XZ implementations and SDK. | High-ratio conventional baseline. | `7z` LZMA2 level 9, metadata controlled. |
| [HF streaming docs](https://huggingface.co/docs/hub/en/datasets-streaming) | Stream Parquet datasets with bounded memory/disk. | Required for FineWeb/FinePDF scale. | **V0 data plan:** stream deterministic byte-budget samples. |
| [FineWeb card](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | >18.5T tokens, Common Crawl cleaning/dedup, official 10B/100B/350B samples. | Broad web candidate and size/licensing reference. | M0/M2 source; never download full corpus. |
| [FineWeb-Edu card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1.3T high-score tokens and 5.4T score-2 tokens from FineWeb. | Tests quality-filtered versus broad web statistics. | M1/M2/M3 candidates. |
| [FinePDFs card](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | ~3T tokens, long PDF documents, extraction artifacts, multilingual content. | Tests long-form and PDF-specific structure. | M3/M4 candidate; stream only. |
| [FinePDFs-Edu card](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) | 350B+ educational PDF tokens, quality top 10%, 69 languages. | Tests long educational documents. | M3 candidate; capped stream. |
| [FineWeb2 card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Multilingual FineWeb-style corpus with 1,868 language-script pairs. | Future multilingual extension. | Not in English-first V0. |
| [Dolma card](https://huggingface.co/datasets/allenai/dolma) | ~3T mixed web/academic/code/books data; practical 10B-token sample. | Broad alternative mixture with explicit source categories. | M5 candidate. |
| [PG19](https://github.com/google-deepmind/pg19) | Long public-domain books and fixed train/validation/test splits. | Long-context evaluation and training option. | Frozen long-form test; never train on val/test. |
| [CodeParrot](https://huggingface.co/datasets/codeparrot/github-code) | 115M files/32 languages, large code corpus with per-file metadata. | Code/structured diversity. | License-filtered held-out slice only. |
| [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 161k annotated messages, 35 languages. | Small conversational evaluation source. | Held-out conversational domain. |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Filtered mostly-English synthetic conversations. | Artificial/informal stress test. | Separate labeled domain, not core training evidence. |
| [Canterbury corpus](https://www.corpus.canterbury.ac.nz/descriptions/) | Small recognized prose/code/HTML/technical benchmark with published sizes. | Reproducible legacy control. | Frozen reference; contamination caveat. |
| [Hutter Prize](https://hutter1.net/prize/index.htm) | enwik9 convention and historical compression records. | Context for CMIX/PAQ and enwik9. | Legacy comparison only; not a clean OOD claim. |
| [Common Crawl latest crawl](https://commoncrawl.org/latest-crawl) | Current crawl index including 2026 snapshots. | Post-training web OOD test. | Acquire a small sample after training manifest freeze. |

## Decisions not yet made

The following remain intentionally open and will be decided by measurements:

- whether a Transformer, RWKV/Mamba/SSM, patch model, or GRU gives the best
  BPB-per-compute;
- whether FineWeb-Edu, FinePDFs, code/structured data, or broader source
  diversity improves unseen-domain compression;
- whether bit-level or reversible subword modeling beats raw bytes after coder
  and metadata costs;
- whether INT8/INT4/ternary weights preserve BPB sufficiently to justify model
  distribution savings;
- whether copy candidates/context mixing complement the neural probabilities;
- whether rANS is materially faster without making the causal file format less
  reliable.

