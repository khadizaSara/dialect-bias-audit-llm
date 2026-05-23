# Dialect Bias Audit in Medical LLMs

A project for CSE449 — Parallel, Distributed & High Performance Computing.

---

## What this is

Medical AI systems are increasingly used as the first point of contact for patients — symptom checkers, triage chatbots, health assistants. These systems are trained mostly on Standard American English. The question I wanted to ask was simple: Does the AI respond differently when a patient describes the same symptom in a different dialect?

To answer that, I built a parallel inference pipeline that sends medical queries to a locally-hosted LLM (Llama 3) and collects urgency scores. The dataset covers four dialect groups — Standard English, African American Vernacular English (AAVE), Non-native English, and Informal/Slang. Each of the 15 symptoms was written in all four dialects, keeping the medical content identical and only changing the language style. That design choice is what makes the comparison meaningful.

The parallel computing side came from a real need: running hundreds of queries through a local LLM one by one is slow. Using Python's `concurrent.futures.ThreadPoolExecutor`, I tested 1, 2, 4, and 8 parallel workers and compared performance against serial execution.

---

## What I found

Standard English scored highest or joint-highest in both dataset versions. AAVE scored 8.33 in both V1 and V2 — unchanged even after I revised all 15 AAVE queries with deeper grammatical features in V2. Non-native English and Informal also showed variation across versions. For high-urgency symptoms like chest pain, panic attack, and sepsis, the dialect gap showed up as a consistent 1-point difference, which in a real triage context is not a small thing.

On the HPC side, parallel inference peaked at 1.39x speedup with 2 workers, then completely plateaued at 4 and 8 workers. This is Amdahl's Law in practice — Ollama processes one request at a time internally, so no matter how many threads are dispatching queries, the bottleneck is the inference engine itself.

---

## Folder structure

```
bias_audit_v1/          first version of the dataset and results
bias_audit_v2/          revised dataset (deeper AAVE features) and results
analysis/               colab notebook, charts
docs/                   proposal, report, presentation
```

---

## How to run it

You need Ollama installed (ollama.com) and Llama 3 pulled:

```bash
ollama pull llama3
```

Install the Python dependencies:

```bash
pip install requests pandas matplotlib
```

Then from either project folder:

```bash
python3 serial_inference.py
python3 parallel_inference.py
```

The parallel script tests 1, 2, 4, and 8 workers and prints the time for each. Results are saved as CSV files.

---

## Dataset

I wrote the queries manually rather than pulling from an existing dataset. The reason is that pre-existing datasets don't let you control for symptom severity across dialects — you can't isolate dialect as the only variable. Writing them myself meant every dialect version of a symptom was medically equivalent, just linguistically different.

There are two versions. V1 used surface-level dialect markers. V2 revised all 15 AAVE queries to include deeper grammatical structures like habitual *be*, aspectual *done*, double negatives, and *been* for ongoing past states. The other three dialects stayed the same between versions.

15 symptoms × 4 dialects = 60 queries per version, 120 total.

---

## Tools used

- Llama 3 8B via Ollama
- Python — concurrent.futures, requests, pandas, matplotlib
- Google Colab for analysis
- Apple M2 MacBook Air
