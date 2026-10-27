# Soft Foreign Keys: Linking Drug Reviews to Standardized Drug Data

Patients do not write drug names the way a drug database stores them. A review says "Xanax",
"glucobay" or "dolo650"; the reference table says "alprazolam", "acarbose", "paracetamol". An exact
join between the two finds almost nothing. This project builds a **soft foreign key**: a semantic
match from the brand or common name in a review to the standardized generic name in a clean
PostgreSQL table, so the two datasets can be queried together.

Course project for CSCI 5707 Database Systems, University of Minnesota, Fall 2025, with Mohith Siva
Sai Bayana, Abhay Shashidhara and Sahitya Avadhanam.

## Why a soft key

An exact join between the two tables barely works. Of the 3,107 reviews (502 distinct drug names as
patients typed them), an exact match on the generic-name column links **4.9%** of reviews, or 6.6%
of the distinct names; allowing substring matches in either direction only reaches 14.7%. The rest
are brand names (Xanax, Lyrica, Vyvanse, Prilosec), salts and spellings the reference table does
not use. (Measured with pandas on the two files in `data/`.)

## What it does

1. **Load** two tables into PostgreSQL: `generic_drugs` (21,362 rows: generic name, package, dosage
   form, strength, manufacturer; 1,507 distinct generic names) and `drug_reviews` (3,107 patient
   reviews from the Druglib.com dataset: drug name as the patient typed it, rating, effectiveness,
   side effects, condition and three free-text review fields).
2. **Embed** every distinct generic name with `all-MiniLM-L6-v2` (384 dimensions) and store the
   vectors in a Pinecone index.
3. **Match** each review's drug name in three stages:
   - an LLM (Llama 3.1 8B Instruct via the Hugging Face Inference API) proposes a generic name
     with a justification;
   - the proposal is embedded and searched against the Pinecone index, keeping candidates above
     0.5 cosine similarity;
   - the LLM picks the best candidate from that short list and the mapping is accepted only if a
     validation pass scores it above 0.5. If the drug is missing from the reference table, the
     answer falls back to the model's own knowledge and says so.
4. **Unify**: the matched `standard_name` acts as the foreign key to join reviews with every dosage
   form of the generic, written to a denormalized `unified_drug_data` table (66,129 rows in the
   project run) for fast querying.
5. **Ask**: a small RAG chatbot answers "what is this drug, and what do patients say about it?"
   from the unified table plus the LLM.

The ER diagram is in `docs/er_diagram.jpg`.

## Repository

```
soft_keys_pipeline.py                 the full pipeline as a script (exported from the notebook)
notebooks/soft_keys_pipeline.ipynb    the same pipeline as a notebook
notebooks/embedding_experiments.ipynb early experiments: cosine vs Euclidean similarity, Pinecone top-k queries
data/generic_drugs.csv                standardized generic-drug reference table
data/drug_reviews.csv                 cleaned, lower-cased review table loaded into PostgreSQL
data/drugLibTrain_raw.tsv             the raw Druglib.com training split the reviews came from
docs/er_diagram.jpg                   schema
```

## Running it

You need a local PostgreSQL database, a Pinecone account and a Hugging Face token with Inference
permission. Keys are read from the environment, never from the code.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in DB_PASSWORD, PINECONE_API_KEY, HF_TOKEN
export $(cat .env | xargs)
python soft_keys_pipeline.py
```

Create the database and load `data/generic_drugs.csv` and `data/drug_reviews.csv` into tables
named `generic_drugs` and `drug_reviews` first (the pipeline reads them with `SELECT *`). The
review table needed manual cleanup before PostgreSQL accepted it: some review text contained
characters and words the bulk loader rejected.

## What we learned

- Embedding similarity alone is not enough. Scores between 0.50 and 0.60 were often wrong
  (`lisdexamfetamine` sits closest to `levetiracetam`), which is why the LLM validation stage
  exists.
- Some drugs in the reviews simply are not in the reference table. Saying "not in our database,
  here is the model's best guess" is better than a confident wrong join.
- A vector index is a complement to the relational database, not a replacement: the relational
  schema still holds the facts, the index only supplies the join key.

## Data sources

- Reviews: Druglib.com drug review dataset (UCI Machine Learning Repository), training split.
- Generic drugs: a public generic-medicine reference table (generic name, dosage form, strength,
  manufacturer).
