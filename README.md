# Drug Safety LLM Fine-Tuning

![tests](https://github.com/S-Harshni/Drug-Safety-LLM-FineTuning/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.12-blue)
![peft](https://img.shields.io/badge/fine--tuning-LoRA_(PEFT)-ee4c2c)
![model](https://img.shields.io/badge/model-Qwen2.5--0.5B-6b46c1)
![license](https://img.shields.io/badge/license-MIT-green)

When is it worth fine-tuning a small language model instead of prompting a larger one? This project answers that on a real pharmacovigilance task: finding **adverse drug events** in sentences from medical case reports. A 0.5-billion-parameter open-source model is fine-tuned with **LoRA** on a laptop and compared with prompting, with a model six times larger, and with a classical baseline.

**Live demo:** https://s-harshni.github.io/Drug-Safety-LLM-FineTuning/

![Results](docs/img/detection.png)

## The two tasks

| Task | Input | Output | Data |
| --- | --- | --- | --- |
| **Detection** | A sentence from a case report | Does it describe an adverse drug event? | 20,895 distinct sentences, 20.4% positive |
| **Extraction** | A sentence that does | Every (drug, adverse effect) pair in it | 4,271 sentences, 1.55 pairs each on average |

Data: [ADE Corpus V2](https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2) (Gurulingappa et al., 2012), sentences from MEDLINE case reports. Repeated sentences are merged before splitting, and the split (70% train, 10% validation, 20% test) is decided by a hash of the sentence, so no sentence is on both sides.

## Results

### Detection

Every method is scored on the same 2,000 held-out sentences. Thresholds are chosen on validation sentences, never on the test set.

| Method | F1 | 95% interval | Precision | Recall | Time per sentence |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Qwen2.5-0.5B-Instruct + LoRA, 4,000 training sentences** | 81.7% | 78.8%–84.6% | 81.0% | 82.4% | 37 ms |
| TF-IDF + logistic regression (classical baseline) | 72.2% | 68.7%–75.7% | 70.6% | 73.8% | 0.05 ms |
| Qwen2.5-0.5B-Instruct + LoRA, 1,000 training sentences | 72.1% | 68.7%–75.4% | 65.5% | 80.2% | 38 ms |
| qwen2.5:3b (6× larger), 6 examples in the prompt | 71.2% | 64.0%–77.8% | 67.5% | 75.4% | 263 ms |
| Qwen2.5-0.5B-Instruct + LoRA, 250 training sentences | 60.9% | 57.2%–64.6% | 50.7% | 76.0% | 52 ms |
| qwen2.5:3b (6× larger), no examples | 49.0% | 38.5%–58.4% | 84.4% | 34.5% | 248 ms |
| Qwen2.5-0.5B-Instruct, no examples in the prompt | 48.5% | 45.0%–51.9% | 35.9% | 74.5% | 37 ms |
| Qwen2.5-0.5B-Instruct, 6 examples in the prompt | 41.2% | 38.1%–44.2% | 27.2% | 84.7% | 159 ms |

- **Fine-tuning is the best method here.** LoRA on 4,000 sentences reaches **81.7% F1** (95% interval 78.8%–84.6%), against 71.2% for the best prompting result and 72.2% for the classical baseline trained on all 14,599 sentences. The intervals do not overlap.
- **But it needs enough data.** F1 by number of training sentences: 60.9% with 250, 72.1% with 1,000, 81.7% with 4,000. With 250 sentences it is below the classical baseline, with 1,000 it matches it, and the lead comes at 4,000.
- **It trains 0.22% of the weights**: 1,081,344 of 495,114,112 parameters, in 25 minutes on a laptop.
- **Prompting the small model is not enough.** Without training it scores 48.5%, and adding examples to the prompt made it worse (41.2%).
- **A model six times larger, with examples in the prompt, gets close to the baseline** (71.2%) but takes 263 ms per sentence against 37 ms for the fine-tuned small model.

### Adapter size

Three LoRA ranks on the same 1,000 training sentences, scored on the same 2,000 test sentences.

| Rank | Trained parameters | Share of the model | F1 | ROC AUC |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 270,336 | 0.05% | 63.9% | 0.880 |
| 8 | 1,081,344 | 0.22% | 72.1% | 0.929 |
| 32 | 4,325,376 | 0.87% | 72.0% | 0.941 |

Rank 2 is too small (63.9%). Rank 8 and rank 32 give the same F1 (72.1% and 72.0%), so four times the parameters bought nothing at this data size; more labelled sentences helped far more than a wider adapter. Rank 8 is used for every other run.

### Extraction

Scored on 400 held-out sentences. A pair counts only if both the drug and the effect match the annotation exactly.

| Method | Pair F1 | Precision | Recall | Sentences fully right | Time per sentence |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Qwen2.5-0.5B-Instruct + LoRA, 2,993 training sentences** | 72.4% | 72.5% | 72.2% | 67.0% | 206 ms |
| qwen2.5:3b (6× larger), 4 examples in the prompt | 64.5% | 68.9% | 60.7% | 52.8% | 958 ms |
| qwen2.5:3b (6× larger), no examples | 59.9% | 64.8% | 55.7% | 50.7% | 1109 ms |
| Qwen2.5-0.5B-Instruct, 4 examples in the prompt | 49.6% | 46.4% | 53.4% | 37.5% | 510 ms |

The fine-tuned 0.5B model reaches **72.4%** pair F1 and gets 67.0% of sentences fully right, against 64.5% and 52.8% for the best prompted model. Exact matching is strict ("skin rash" against an annotation of "rash" is a miss), so part of the gain is the fine-tuned model learning the annotation style as well as the task.

![Extraction](docs/img/extraction.png)

### Real-world check against FDA data

Do the pairs the fine-tuned model writes out match what happens after a drug is on the market? Each (drug, effect) pair
from the first 120 held-out sentences is checked against **20,692,690 reports in FAERS**, the FDA Adverse Event Reporting
System that drug-safety teams monitor, and against the drug's **FDA label** ([`faers_signals.py`](faers_signals.py),
[`faers.py`](src/drugsafety/faers.py), data from openFDA). A pair is a **signal** when the effect is reported with the drug
disproportionately often: proportional reporting ratio (PRR) >= 2, chi-squared >= 4 and at least 3 reports (Evans et al., 2001).
The same check runs on the annotators' pairs and on shuffled pairs (each model pair given an effect from another sentence).

| Pairs | Checked | FAERS signal | Effect on the FDA label | Median PRR |
| --- | ---: | ---: | ---: | ---: |
| **Fine-tuned model** | 100 of 194 | **54.0%** | **52.4%** | 2.56 |
| Human annotators | 98 of 191 | 56.1% | 51.8% | 2.79 |
| Shuffled pairs (chance) | 102 of 194 | 22.6% | 20.5% | 1.13 |

- **The model's output behaves like expert annotation in real-world data**, and is confirmed about 2.4 times as often as chance pairs.
- Known effects it recovered include ethambutol and optic neuropathy (PRR 114), metformin and lactic acidosis (PRR 73) and
  methotrexate and leukoencephalopathy (PRR 15).
- **20 of its pairs are FAERS signals whose effect the label does not mention**: the queue a safety reviewer would start with.
  Some are wording gaps (the gentamicin label says nephrotoxicity, not renal failure) and some are confounding by indication
  (furosemide is given for heart failure); a signal is an association, not proof of cause.
- "Checked" means the drug is in FAERS and the effect maps to a MedDRA term. Mapping uses the same small normalisation for
  every set (British spelling, plurals, modifiers such as "severe", everyday words such as fever to pyrexia, and common
  abbreviations such as 5-FU). Every openFDA response is cached in `data/faers_cache.json.gz`, so `python faers_signals.py --offline`
  reproduces the table without the API.

## How it works

- **Base model:** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), frozen.
- **LoRA:** low-rank adapters on the attention projections (q_proj, k_proj, v_proj, o_proj) through the PEFT library ([`finetune.py`](src/drugsafety/finetune.py)).
- **Training:** 2 epochs, AdamW, warm-up and linear decay, gradient clipping. The loss is computed on the answer tokens only. Each optimiser step accumulates gradients over micro-batches so back-propagation fits in laptop memory.
- **Reading the answer:** detection compares the logits of "yes" and "no" in one forward pass, which gives a score that can be thresholded; extraction generates `drug | effect` lines with greedy decoding.
- **Same instructions for every model** ([`prompts.py`](src/drugsafety/prompts.py)): the fine-tuned model, the untrained one and the larger prompted one, so the comparison is about the method.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py               # downloads the corpus; baseline, prompting the small model, all LoRA runs (resumable)
ollama pull qwen2.5:3b && python eval_prompting.py    # the larger prompted model
python demo_examples.py     # runs the adapters on sentences written for the demo page
python build_demo.py        # docs/data.json for the demo page
python faers_signals.py     # real-world check against FDA data (cached; --offline needs no API)
pytest -q                   # 13 tests
```

Tested on an Apple-silicon laptop (mps); it also runs on CUDA or, slowly, on CPU.

## Tests

13 tests, run in CI: the hash split and de-duplication, every metric on a hand-worked case, threshold selection, the bootstrap interval, prompt construction and parsing, that a fresh LoRA adapter changes nothing and trains only adapter weights, that training labels cover only the answer tokens, the PRR on a hand-worked 2x2 table, MedDRA spelling and label matching, and the offline cache.

## Limitations

- FAERS reports are voluntary and unverified, so counts reflect reporting, not incidence; a drug is matched by name in any role on a report. Without an openFDA key the check covers 120 of the 400 held-out sentences (1,000 requests a day).

- The public corpus has no document ids, so sentences from the same case report can fall on both sides of the split. That favours every trained method, the baseline included.
- One random seed and one base model; the 95% intervals show how much the test sample alone moves a score.
- The larger model was prompted on 500 detection sentences (the fine-tuned model on 2,000) to keep the run time reasonable.
- Extraction is scored by exact match, which under-rates answers that are right but worded differently.
- This is a research exercise on published literature. It is not a medical device and makes no clinical claims.

## Data

Gurulingappa, H., et al. (2012). *Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports.* Journal of Biomedical Informatics. The corpus is downloaded at run time and not redistributed here.

## Author

[S Harshni](https://github.com/S-Harshni)
