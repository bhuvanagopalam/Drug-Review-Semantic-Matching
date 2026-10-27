# -*- coding: utf-8 -*-
"""Soft foreign keys: link messy drug reviews to a standardized generic-drug table.

Exported from the project notebook (notebooks/soft_keys_pipeline.ipynb) and cleaned of credentials.

### Cell 1 – Imports and global display options

This cell imports all the Python libraries we need for the project and sets a few
global options to make the output easier to read in a notebook. In particular,
we import:
- `psycopg2` to connect to PostgreSQL,
- `pandas` and `numpy` for data wrangling,
- `sentence_transformers` and `sklearn` for embeddings and similarity,
- `pinecone` and `huggingface_hub` for the RAG / LLM part.

We also set pandas display options so that long text columns and tables are not truncated.
No data is loaded yet in this cell – it just prepares the environment.
"""

import os

import psycopg2
import pandas as pd
from psycopg2.extras import execute_values

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient

pd.set_option("display.max_colwidth", 200)

print("Imports loaded.")

"""### Cell 2 – Connect to the local PostgreSQL database

This cell opens a connection to the local PostgreSQL instance where our
project tables live. The actual credentials (host, database name, user,
password, port) are taken from variables so that we don't hard-code
secrets in the notebook.

The output of this cell should just confirm that the connection was created.
All later SQL operations reuse this connection.

"""

# Cell 2 – PostgreSQL connection (edit password if needed)
DB_HOST = 'localhost'
DB_PORT = 5432
DB_NAME = 'dbproject'      # your project database
DB_USER = 'postgres'       # your postgres user

# Set DB_PASSWORD, PINECONE_API_KEY and HF_TOKEN in the environment (see .env.example)
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')

conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
)
print('Connected to PostgreSQL db:', DB_NAME)

"""### Cell 3 – Load generic_drugs and drug_reviews from PostgreSQL

This cell reads the two main project tables directly from PostgreSQL:

- `generic_drugs`: the clean reference table with one row per generic drug form
- `drug_reviews`: the messy user review table with brand/common names and text

We use `pd.read_sql` to pull the full tables into pandas DataFrames and quickly
inspect the shape and first few rows to sanity-check that the connection and tables
are working.

"""

SQL_GENERIC = "SELECT * FROM generic_drugs;"
SQL_REVIEWS = "SELECT * FROM drug_reviews;"

df_generic = pd.read_sql(SQL_GENERIC, conn)
df_reviews = pd.read_sql(SQL_REVIEWS, conn)

print("generic_drugs rows:", len(df_generic))
print("drug_reviews rows:", len(df_reviews))

display(df_generic.head())
display(df_reviews.head())

"""### Cell 4 – Basic text cleaning and normalization

This cell standardizes the key text fields we will later match on. For example,
we lower-case the generic drug names and the messy `urldrugname` / `review_drugname`
columns, strip spaces, and optionally remove trivial punctuation.

The goal is not to "fix" all spelling errors, but to make obvious matches like
"Glucobay" vs "glucobay" line up before we use embeddings and Pinecone for fuzzier matching.


"""

# --- Configure column names used for soft-keying ---
GENERIC_NAME_COL    = "genericname"    # clean generic name in generic_drugs
REVIEW_DRUGNAME_COL = "urldrugname"    # messy drug name in drug_reviews
REVIEW_RATING_COL   = "rating"         # numeric rating column

# Build ONE combined free-text review column inside pandas (still from SQL)
TEXT_COLS = ["benefitsreview", "sideeffectsreview", "commentsreview"]

for c in TEXT_COLS:
    if c not in df_reviews.columns:
        raise ValueError(
            f"Expected column '{c}' in drug_reviews table. "
            f"Actual columns: {df_reviews.columns.tolist()}"
        )

df_reviews["review"] = (
    df_reviews[TEXT_COLS]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
)

REVIEW_TEXT_COL = "review"  # new combined text column

# --- Build canonical lists for embeddings ---

generic_names = (
    df_generic[GENERIC_NAME_COL]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .unique()
)

review_drugnames = (
    df_reviews[REVIEW_DRUGNAME_COL]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .unique()
)

print("Unique generic names:", len(generic_names))
print("Unique review drug names:", len(review_drugnames))
print("Sample generic names:", list(generic_names[:10]))
print("Sample review drug names:", list(review_drugnames[:10]))

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("Loaded embedding model:", EMBED_MODEL_NAME)

# ---- Pinecone API key directly in the cell ----
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")

if not PINECONE_API_KEY:
    raise SystemExit("Set PINECONE_API_KEY in the environment before running.")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "softkeys-index"

existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,        # all-MiniLM-L6-v2 output dim
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Created Pinecone index:", index_name)
else:
    print("Using existing Pinecone index:", index_name)

softkey_index = pc.Index(index_name)

"""Build embeddings for generic names and index them in Pinecone (RAG prep)

This cell prepares the vector search layer used for soft-key matching:

1. Load a small sentence-transformer model (`all-MiniLM-L6-v2`).
2. Embed every unique generic drug name into a vector.
3. Connect to Pinecone using the API key already set in the environment.
4. Create (or reuse) an index named `softkeys-index`.
5. Upsert one vector per generic drug, with metadata containing the `genericname`.

Later, when a messy brand/common name arrives, we embed it and query this
index to get the top-k candidate generic names.

"""

generic_embs = embed_model.encode(
    list(generic_names),
    convert_to_numpy=True,
    show_progress_bar=True,
)

print("generic_embs shape:", generic_embs.shape)

vectors = []
for i, (name, emb) in enumerate(zip(generic_names, generic_embs)):
    vec_id = f"gen-{i}"
    metadata = {"genericname": name}
    vectors.append((vec_id, emb.tolist(), metadata))

batch_size = 200
for i in range(0, len(vectors), batch_size):
    softkey_index.upsert(vectors=vectors[i:i+batch_size])

print("Upserted", len(vectors), "generic vectors into Pinecone.")

SIM_THRESHOLD = 0.5   # you can tune this

def get_similar_generics_with_threshold(query_name: str,
                                        similarity_threshold: float = SIM_THRESHOLD,
                                        top_k: int = 50):
    """
    Given a messy/common name, embed it, query Pinecone,
    and return candidates with score >= threshold.
    """
    # 1) embed query
    query_emb = embed_model.encode([query_name.lower()],
                                   convert_to_numpy=True)[0].tolist()

    # 2) Pinecone query
    res = softkey_index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True,
    )

    similar = []
    for match in res["matches"]:
        score = match["score"]
        if score >= similarity_threshold:
            word = match["metadata"].get("genericname", "N/A")
            similar.append((word, score))

    # Sort by score descending just in case
    similar.sort(key=lambda x: x[1], reverse=True)
    return similar

# quick smoke test
for test_word in ["glucobay", "elavil", "vasotec", "ortho-tri-cyclen"]:
    print(f"\nQuery = {test_word}")
    print(get_similar_generics_with_threshold(test_word)[:5])

def map_review_row(row):
    messy = str(row[REVIEW_DRUGNAME_COL]).strip().lower()
    if not messy:
        return pd.Series({"mapped_generic": None, "softkey_score": None})

    candidates = get_similar_generics_with_threshold(messy, similarity_threshold=SIM_THRESHOLD, top_k=20)
    if not candidates:
        return pd.Series({"mapped_generic": None, "softkey_score": None})

    best_generic, best_score = candidates[0]
    return pd.Series({"mapped_generic": best_generic, "softkey_score": best_score})

mapped = df_reviews.apply(map_review_row, axis=1)
df_reviews = pd.concat([df_reviews, mapped], axis=1)

df_reviews[["urldrugname", "mapped_generic", "softkey_score"]].head(10)

TEXT_COLS = ["benefitsreview", "sideeffectsreview", "commentsreview"]

for c in TEXT_COLS:
    if c not in df_reviews.columns:
        print(f"WARNING: {c} not in df_reviews; adjust TEXT_COLS if needed.")

df_reviews["review_text"] = (
    df_reviews[TEXT_COLS]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
)

df_reviews[["urldrugname", "mapped_generic", "review_text"]].head()

df_unified = df_reviews.merge(
    df_generic,
    left_on="mapped_generic",
    right_on=GENERIC_NAME_COL,
    how="left",
    suffixes=("_review", "_generic"),
)

print("Unified rows:", len(df_unified))
df_unified.head()

""" Create (or recreate) the unified_drug_data table in PostgreSQL

Here we define the unified table schema in PostgreSQL that will hold the "soft key"
mapping between messy review names and clean generic drugs.

The high-level steps:
1. Drop any existing object named `unified_drug_data` (view or table).
2. Create a fresh table with:
   - review-level information (drug name, rating, condition, review text, etc.)
   - mapped generic name and soft-key score
   - generic metadata from `generic_drugs` (dose, strength, manufacturer, ...)

This is the table our RAG pipeline and chatbot will later query.

"""

TABLE_NAME = "unified_drug_data"

cur = conn.cursor()
cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE;")

# You can tune schema; here we keep it fairly generic with text columns
create_sql = f"""
CREATE TABLE {TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    urldrugname TEXT,
    mapped_generic TEXT,
    softkey_score DOUBLE PRECISION,
    review_text TEXT,
    rating DOUBLE PRECISION,
    condition TEXT,
    effectiveness TEXT,
    sideeffects TEXT,
    -- generic metadata from generic_drugs
    genericname TEXT,
    packagemark TEXT,
    dosagetype TEXT,
    strength TEXT,
    manufacturer TEXT
);
"""
cur.execute(create_sql)
conn.commit()
print(f"Created table {TABLE_NAME}.")

# Prepare rows to insert (subset of columns we created above)
cols_for_sql = [
    "urldrugname",
    "mapped_generic",
    "softkey_score",
    "review_text",
    "rating",
    "condition",
    "effectiveness",
    "sideeffects",
    "genericname",      # from generic_drugs
    "packagemark",
    "dosagetype",
    "strength",
    "manufacturer",
]

rows = df_unified[cols_for_sql].where(pd.notnull(df_unified), None).to_records(index=False).tolist()

insert_sql = f"""
INSERT INTO {TABLE_NAME} (
    urldrugname, mapped_generic, softkey_score, review_text,
    rating, condition, effectiveness, sideeffects,
    genericname, packagemark, dosagetype, strength, manufacturer
) VALUES %s;
"""

cur = conn.cursor()
execute_values(cur, insert_sql, rows)
conn.commit()
cur.close()

print(f"Inserted {len(rows)} rows into {TABLE_NAME}.")

HF_TOKEN = os.environ.get("HF_TOKEN", "")  # fine-grained token with Inference permission
LLM_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

hf_client = InferenceClient(api_key=HF_TOKEN)

print("HF client ready with model:", LLM_MODEL_ID)

def validate_drug_mapping(common_name: str, chosen_drug: str, justification: str) -> float:
    """
    Ask LLaMA to give a confidence 0–1 that chosen_drug is the correct
    generic for common_name, based on justification.
    """
    validation_prompt = f"""
You are a strict medical AI assistant specializing in drug data.

You are given:
- Common/brand drug name: "{common_name}"
- Proposed standardized drug name: "{chosen_drug}"
- Justification for this mapping: "{justification}"

Your task:
1. Decide how likely it is that the proposed standardized drug is the correct primary active ingredient for the given common/brand name.
2. Return a single confidence score between 0 and 1.

Output format:
Confidence: <number between 0 and 1>
"""

    completion = hf_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "system", "content": validation_prompt}],
    )

    response = completion.choices[0].message.content.strip()

    # Expected:  "Confidence: 0.87"
    try:
        if ":" in response:
            score_str = response.split(":", 1)[1].strip()
        else:
            score_str = response.strip()
        conf = float(score_str)
    except Exception:
        conf = 0.0

    conf = max(0.0, min(1.0, conf))
    return conf

"""Soft-key mapping: from messy review name to generic name

This cell defines the core helper functions for the "soft foreign key":

1. `get_similar_generics_with_threshold`:
   - embeds a messy common/brand name,
   - queries Pinecone,
   - returns all candidate generic names with similarity above a given threshold.

2. `map_review_row`:
   - called on each review row,
   - uses the function above to choose the best generic candidate (if any),
   - returns both the chosen generic and the similarity score.

These functions are used to populate the `mapped_generic` and `softkey_score`
columns in the unified table.
"""

common = "xanax"
cands = get_similar_generics_with_threshold(common, similarity_threshold=SIM_THRESHOLD)

if cands:
    best_generic, best_score = cands[0]
    just = f"Soft foreign key via embedding + Pinecone; cosine score = {best_score:.3f}."
    conf = validate_drug_mapping(common, best_generic, just)
    print("Common:", common)
    print("Mapped generic:", best_generic)
    print("Cosine score:", best_score)
    print("LLM confidence:", conf)
else:
    print("No candidate above threshold for", common)

# Cell A - Export unified_drug_data to CSV (for download / backup)

import pandas as pd

UNIFIED_TABLE = "unified_drug_data"   # change only if your table name differs

sql_unified = f"SELECT * FROM {UNIFIED_TABLE};"

df_unified = pd.read_sql(sql_unified, conn)
print(f"Rows in {UNIFIED_TABLE}:", len(df_unified))

# Save to CSV in the current folder
csv_path = f"{UNIFIED_TABLE}.csv"
df_unified.to_csv(csv_path, index=False)
print(f"Saved unified table to: {csv_path}")

# Optional: peek at the first few rows
df_unified.head()

# Cell B - Helper to fetch context rows from unified_drug_data

def get_unified_rows_for_generic_or_brand(name: str, limit: int = 50) -> pd.DataFrame:
    """
    Look up rows in unified_drug_data matching this name as:
      - mapped_generic
      - genericname
      - urldrugname (common/brand from reviews)
    Case-insensitive, returns up to `limit` rows.
    """
    name = name.strip().lower()
    sql = f"""
        SELECT *
        FROM {UNIFIED_TABLE}
        WHERE LOWER(mapped_generic) = %s
           OR LOWER(genericname)   = %s
           OR LOWER(urldrugname)   = %s
        LIMIT %s;
    """
    return pd.read_sql(sql, conn, params=[name, name, name, limit])

# Cell - LLaMA helper: choose_generic_with_llm

SYSTEM_PROMPT = """
You are a careful medical assistant helping map messy common/brand drug
names to standardized generic drug names from a database.

You will be given:
- A common/brand drug name
- A short list of candidate generic names from the database

Your job:
1. Pick the single best generic name if one clearly matches.
2. If none look correct, return 'none'.

Always follow the output format exactly:
Drug Name : <generic-name-or-none> ##### Justification : <short reason>
""".strip()


def choose_generic_with_llm(common_name: str, top_k: int = 10):
    """
    1) Get top_k similar generic candidates from Pinecone.
    2) Ask LLaMA to pick the best generic (or 'none').
    3) Return (chosen_generic_name_or_None, justification, raw_candidates).
    """
    # This assumes you already have:
    #   - embed_model  (SentenceTransformer)
    #   - softkey_index (Pinecone index)
    #   - a helper get_top_similar_generics()  OR we define it here.

    # ---- if you DON'T already have get_top_similar_generics, use this: ----
    def get_top_similar_generics(query: str, top_n: int = 10):
        """Return list of (generic_name, score) from Pinecone."""
        q_emb = embed_model.encode([query.lower()], convert_to_numpy=True)[0]
        res = softkey_index.query(vector=q_emb.tolist(),
                                  top_k=top_n,
                                  include_metadata=True)
        out = []
        for m in res["matches"]:
            name = m["metadata"].get("genericname", "").strip()
            score = m["score"]
            out.append((name, score))
        return out

    candidates = get_top_similar_generics(common_name, top_n=top_k)
    candidate_names = [w for (w, _) in candidates]

    if not candidate_names:
        return None, "No candidates from Pinecone.", []

    candidate_block = "\n".join(f"- {w}" for w in candidate_names)

    user_prompt = (
        f"Common/brand name: '{common_name}'\n\n"
        f"Candidate generic names from database:\n{candidate_block}\n\n"
        "Follow the format exactly:\n"
        "Drug Name : <generic-name-or-none> ##### Justification : <reason>"
    )

    completion = hf_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = completion.choices[0].message.content.strip()

    # Simple parsing: "Drug Name : X ##### Justification : Y"
    chosen = None
    justification = content

    if "##### " in content:
        drug_part, just_part = content.split("#####", 1)
        chosen_raw = (
            drug_part.replace("Drug Name", "")
            .replace(":", "")
            .strip()
            .strip("'\"")
        )
        justification = (
            just_part.replace("Justification", "")
            .replace(":", "")
            .strip()
        )

        if chosen_raw and chosen_raw.lower() not in {"none", "no match", "no generic"}:
            chosen = chosen_raw

    return chosen, justification, candidates

def drug_rag_chatbot(common_name: str, top_k: int = 10) -> str:
    ...
    common_name = common_name.strip()

    # 1) Use your existing LLM-based mapper
    generic_name, gen_justification, _candidates = choose_generic_with_llm(
        common_name, top_k=top_k
    )
    ...
    if not generic_name:
        return f"No suitable generic drug found for '{common_name}'."

# Cell C - RAG chatbot over unified_drug_data

def drug_rag_chatbot(common_name: str, top_k: int = 10) -> str:
    """
    High-level flow:
      1) Use your existing choose_generic_with_llm() to map common -> generic.
      2) Look into unified_drug_data for that generic (or brand name).
      3) If rows exist, pass a compact summary of them as context to LLaMA.
      4) If no rows exist, ask LLaMA to answer from its own knowledge, and
         clearly say that the project table had no matching rows.
    """

    common_name = common_name.strip()

    # 1) Use your existing LLM-based mapper
    generic_name, gen_justification, _candidates = choose_generic_with_llm(
        common_name, top_k=top_k
    )

    if not generic_name or generic_name.lower() in {"none", "unknown", "n/a"}:
        generic_name = "unknown"

    # 2) Pull context from the unified table
    df_ctx = get_unified_rows_for_generic_or_brand(generic_name, limit=50)
    in_table = not df_ctx.empty

    # Build a small, structured context string from the table (if we have rows)
    if in_table:
        num_rows = len(df_ctx)

        # Aggregate a few things to keep prompt small
        example_conditions = (
            df_ctx["condition"]
            .dropna()
            .astype(str)
            .str.lower()
            .value_counts()
            .head(5)
            .to_dict()
        )

        example_side = (
            df_ctx["sideeffects"]
            .dropna()
            .astype(str)
            .str.lower()
            .value_counts()
            .head(5)
            .to_dict()
        )

        avg_rating = None
        if "rating" in df_ctx.columns:
            try:
                avg_rating = float(df_ctx["rating"].dropna().astype(float).mean())
            except Exception:
                avg_rating = None

        context_lines = [f"Rows in unified_drug_data for generic '{generic_name}': {num_rows}"]

        if avg_rating is not None:
            context_lines.append(f"Average user rating: {avg_rating:.2f}")

        if example_conditions:
            cond_str = "; ".join(f"{k} (count={v})" for k, v in example_conditions.items())
            context_lines.append(f"Common reported conditions: {cond_str}")

        if example_side:
            side_str = "; ".join(f"{k} (count={v})" for k, v in example_side.items())
            context_lines.append(f"Common side effects mentioned: {side_str}")

        table_context = "\n".join(context_lines)
    else:
        table_context = (
            f"No rows were found in {UNIFIED_TABLE} for generic name '{generic_name}'. "
            "The answer below is based only on the model's general medical knowledge."
        )

    # 3) Build the prompt for LLaMA

    user_prompt = f"""
A user entered the common/brand drug name: "{common_name}".

The system attempted to map it to a generic drug name using vector search + LLaMA:
- Mapped generic name: "{generic_name}"
- Justification from that step: "{gen_justification}"

Project database context (from the unified_drug_data table):
{table_context}

Now, in 3–5 simple sentences, explain to the user:

1. What generic name you mapped "{common_name}" to (or say clearly if it is unknown).
2. Whether this generic appears in the project's unified_drug_data table or not.
3. In plain, non-technical language, what this medicine is usually used for.
4. Clearly state that this is educational information only and NOT medical advice.

Avoid long lists; keep it friendly and easy to understand.
"""

    completion = hf_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful but careful medical explainer bot for a student "
                    "database project. Use the table context faithfully; if it is weak, "
                    "do not over-claim. Always remind the user this is not medical advice."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = completion.choices[0].message.content.strip()
    return answer

# Quick manual test (optional)
get_unified_rows_for_generic_or_brand("xanax").head()

# Cell C - RAG chatbot over unified_drug_data

def drug_rag_chatbot(common_name: str, top_k: int = 10) -> str:
    """
    High-level flow:
      1) Use your existing choose_generic_with_llm() to map common -> generic.
      2) Look into unified_drug_data for that generic (or brand name).
      3) If rows exist, pass a compact summary of them as context to LLaMA.
      4) If no rows exist, ask LLaMA to answer from its own knowledge, and
         clearly say that the project table had no matching rows.
    """

    common_name = common_name.strip()

    # 1) Use your existing LLM-based mapper
    generic_name, gen_justification, _candidates = choose_generic_with_llm(
        common_name, top_k=top_k
    )

    if not generic_name or generic_name.lower() in {"none", "unknown", "n/a"}:
        generic_name = "unknown"

    # 2) Pull context from the unified table
    df_ctx = get_unified_rows_for_generic_or_brand(generic_name, limit=50)
    in_table = not df_ctx.empty

    # Build a small, structured context string from the table (if we have rows)
    if in_table:
        num_rows = len(df_ctx)

        # Aggregate a few things to keep prompt small
        example_conditions = (
            df_ctx["condition"]
            .dropna()
            .astype(str)
            .str.lower()
            .value_counts()
            .head(5)
            .to_dict()
        )

        example_side = (
            df_ctx["sideeffects"]
            .dropna()
            .astype(str)
            .str.lower()
            .value_counts()
            .head(5)
            .to_dict()
        )

        avg_rating = None
        if "rating" in df_ctx.columns:
            try:
                avg_rating = float(df_ctx["rating"].dropna().astype(float).mean())
            except Exception:
                avg_rating = None

        context_lines = [f"Rows in unified_drug_data for generic '{generic_name}': {num_rows}"]

        if avg_rating is not None:
            context_lines.append(f"Average user rating: {avg_rating:.2f}")

        if example_conditions:
            cond_str = "; ".join(f"{k} (count={v})" for k, v in example_conditions.items())
            context_lines.append(f"Common reported conditions: {cond_str}")

        if example_side:
            side_str = "; ".join(f"{k} (count={v})" for k, v in example_side.items())
            context_lines.append(f"Common side effects mentioned: {side_str}")

        table_context = "\n".join(context_lines)
    else:
        table_context = (
            f"No rows were found in {UNIFIED_TABLE} for generic name '{generic_name}'. "
            "The answer below is based only on the model's general medical knowledge."
        )

    # 3) Build the prompt for LLaMA

    user_prompt = f"""
A user entered the common/brand drug name: "{common_name}".

The system attempted to map it to a generic drug name using vector search + LLaMA:
- Mapped generic name: "{generic_name}"
- Justification from that step: "{gen_justification}"

Project database context (from the unified_drug_data table):
{table_context}

Now, in 3–5 simple sentences, explain to the user:

1. What generic name you mapped "{common_name}" to (or say clearly if it is unknown).
2. Whether this generic appears in the project's unified_drug_data table or not.
3. In plain, non-technical language, what this medicine is usually used for.
4. Clearly state that this is educational information only and NOT medical advice.

Avoid long lists; keep it friendly and easy to understand.
"""

    completion = hf_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful but careful medical explainer bot for a student "
                    "database project. Use the table context faithfully; if it is weak, "
                    "do not over-claim. Always remind the user this is not medical advice."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = completion.choices[0].message.content.strip()
    return answer

# Helper: let LLaMA guess the generic name from its own knowledge,
# even if Pinecone / DB don't help.

def guess_generic_from_llm(common_name: str):
    """
    Use the LLM's own medical knowledge (not the database) to guess
    the most likely generic / active ingredient for a common/brand name.

    Returns:
        (generic_name_or_None, justification_text)
    """

    prompt = f"""
You are a careful medical assistant.

A user typed this common/brand drug name:
"{common_name}"

Your tasks:
1. From your own medical knowledge (do NOT assume a table), guess the most likely
   generic / active ingredient name for this drug.
2. If you truly do not recognize the name, say 'unknown'.
3. Be conservative: if you are not reasonably sure, prefer 'unknown'.

Output format (one line):
Generic : <generic-name-or-unknown> ##### Justification : <short reason>
"""

    completion = hf_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": "You are a cautious medical assistant. When in doubt, say 'unknown'.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = completion.choices[0].message.content.strip()

    # Parse "Generic : X ##### Justification : Y"
    generic = None
    justification = content

    if "##### " in content:
        generic_part, just_part = content.split("#####", 1)
        generic_raw = (
            generic_part.replace("Generic", "")
            .replace(":", "")
            .strip()
            .strip("'\"")
        )
        justification = (
            just_part.replace("Justification", "")
            .replace(":", "")
            .strip()
        )

        if generic_raw and generic_raw.lower() not in {"unknown", "none", "no match"}:
            generic = generic_raw

    return generic, justification

def drug_rag_chatbot(common_name: str, top_k: int = 10) -> str:
    """
    High-level RAG-style chatbot:

    1) Try to map common/brand name -> generic using Pinecone + LLaMA.
    2) If that fails, ask LLaMA to guess the generic directly from its own knowledge.
    3) Look into unified_drug_data for that generic (or brand) to pull context.
    4) Let LLaMA answer using:
       - table context if available, plus
       - its own medical knowledge if needed.
    """

    common_name = common_name.strip()

    # ---------------------------------------------------------
    # 1) First attempt: use Pinecone + LLaMA mapping
    # ---------------------------------------------------------
    generic_name, gen_justification, _candidates = choose_generic_with_llm(
        common_name, top_k=top_k
    )

    source_info = "Mapped via Pinecone + LLaMA."
    fallback_justification = None

    # ---------------------------------------------------------
    # 2) If that failed / returned 'none' or 'unknown', let LLaMA guess
    # ---------------------------------------------------------
    if not generic_name or generic_name.lower() in {"none", "unknown", "n/a"}:
        guessed_generic, guess_just = guess_generic_from_llm(common_name)
        fallback_justification = guess_just

        if guessed_generic and guessed_generic.lower() not in {"unknown", "none"}:
            generic_name = guessed_generic
            gen_justification = (
                f"Guessed by LLaMA without DB candidates: {guess_just}"
            )
            source_info = "Guessed purely from LLaMA's own medical knowledge."
        else:
            generic_name = "unknown"
            gen_justification = (
                f"No reliable mapping found even after LLaMA guess. "
                f"Justification: {guess_just}"
            )
            source_info = (
                "No reliable generic mapping found; explanation is based only on "
                "LLaMA's general medical knowledge, not on the project table."
            )

    # ---------------------------------------------------------
    # 3) Pull context from unified_drug_data (if we have any generic)
    # ---------------------------------------------------------
    if generic_name.lower() != "unknown":
        df_ctx = get_unified_rows_for_generic_or_brand(generic_name, limit=50)
        in_table = not df_ctx.empty
    else:
        df_ctx = None
        in_table = False

    # Build a compact context string from the table
    if in_table:
        num_rows = len(df_ctx)

        example_conditions = (
            df_ctx["condition"]
            .dropna()
            .astype(str)
            .str.lower()
            .value_counts()
            .head(5)
            .to_dict()
            if "condition" in df_ctx.columns
            else {}
        )

        example_ratings = (
            df_ctx["rating"]
            .dropna()
            .astype(float)
            .describe()
            .to_dict()
            if "rating" in df_ctx.columns
            else {}
        )

        context_summary = f"""
Project table 'unified_drug_data' has {num_rows} rows for generic '{generic_name}'.

Most common conditions in these rows:
{example_conditions}

Basic rating statistics (user-reported):
{example_ratings}
""".strip()
    else:
        context_summary = (
            f"The generic name '{generic_name}' does NOT appear in the "
            f"'unified_drug_data' table for this project, or no generic was found. "
            f"Any explanation you give should be based only on your general "
            f"medical knowledge, not on the project data."
        )

    # ---------------------------------------------------------
    # 4) Final answer prompt to LLaMA
    # ---------------------------------------------------------
    user_prompt = f"""
The user typed the common/brand drug name: "{common_name}".

You (the system) ended up with this generic name: "{generic_name}".
How we got it:
- {source_info}
- Detailed mapping justification: "{gen_justification}"

Project table context:
{context_summary}

Now, in 3–6 simple sentences, explain to the user:

1. What generic name you are using for "{common_name}" (if any).
2. Whether that name came from the project's table + Pinecone or from your own knowledge only.
3. In plain, non-technical language, what the medicine is usually used for.
4. Mention gently that this is educational information only, not medical advice, and
   they must ask a doctor or pharmacist before changing any medication.
"""

    completion = hf_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful but cautious medical explainer bot for a "
                    "student database project. Never give treatment instructions; "
                    "just describe typical uses and explain uncertainty honestly."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = completion.choices[0].message.content.strip()
    return answer

# Ask once, answer once
common_name = input("Enter a common/brand drug name: ").strip()
answer = drug_rag_chatbot(common_name)
print("\n" + "=" * 60)
print(answer)

def drug_rag_chatbot(common_name: str, top_k: int = 10) -> str:
    """
    High-level chatbot:

    1) Map common/brand name -> generic name (manual overrides + LLM+Pinecone).
    2) Look into unified_drug_data for that generic/brand.
    3) If rows exist, build a compact summary for the LLM.
    4) Ask the LLM to answer in simple language, without talking about
       Pinecone, embeddings, or 'our project'.
    """

    common_name = common_name.strip()

    # --- 1) Map to generic name (with overrides) ---
    generic_name, gen_justification, _candidates = smart_map_common_to_generic(
        common_name, top_k=top_k
    )

    if not generic_name or generic_name.lower() in {"none", "unknown", "n/a"}:
        generic_name = "unknown"

    # --- 2) Pull context from the unified table (generic OR brand) ---
    df_ctx = get_unified_rows_for_generic_or_brand(generic_name, limit=50)
    in_table = not df_ctx.empty

    # --- 3) Build a tiny structured context summary (only if we have rows) ---
    if in_table:
        num_rows = len(df_ctx)

        # Example: what conditions does this appear for?
        example_conditions = (
            df_ctx["condition"]
            .dropna()
            .astype(str)
            .str.lower()
            .value_counts()
            .head(5)
            .to_dict()
        )

        # Simple rating stats
        rating_series = (
            df_ctx["rating"]
            .dropna()
            .astype(float)
        )
        if not rating_series.empty:
            avg_rating = round(rating_series.mean(), 2)
        else:
            avg_rating = "n/a"

        context_lines = [
            f"Number of rows in table: {num_rows}",
            f"Approx. average rating: {avg_rating}",
            f"Most common conditions (top 5): {example_conditions}",
        ]
        db_context = "\n".join(context_lines)
        db_status = "found_rows"
    else:
        db_context = "The table has no matching rows for this medicine."
        db_status = "no_rows"

    # --- 4) Build a MUCH simpler prompt for the LLM ---
    if generic_name == "unknown":
        # Model could not confidently map the name; ask it to do its best
        user_prompt = f"""
The user asked about a medicine called "{common_name}".

You could not reliably map it to a generic active ingredient from the table.

Using only your general medical knowledge:
- If you recognize "{common_name}", explain briefly what it is usually used for.
- If you do NOT recognize it, clearly say that you are not sure what it is.
- Do NOT invent details.

Answer in 2–4 short sentences, in simple language,
and end with: "This is not medical advice."
"""
    else:
        # Normal path: we have a generic name and some table context
        user_prompt = f"""
The user asked about the medicine "{common_name}".

You inferred that its generic (active ingredient) is "{generic_name}".
Short reason: {gen_justification or "no extra justification."}

From the database we know:
{db_context}

In 3–5 short sentences, in very simple language, please:
1) Confirm the mapping from "{common_name}" to "{generic_name}".
2) Say what this medicine is usually used for.
3) Optionally mention one or two very common side effects (if you know them).
4) Always end with: "This is not medical advice."
"""

    # --- 5) Call the LLM (same client/model as before) ---
    completion = hf_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": "You are a concise medical explainer. Be clear and avoid technical jargon.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    answer = completion.choices[0].message.content.strip()
    return answer

common_name = input("Enter a common/brand drug name: ").strip()
answer = drug_rag_chatbot(common_name)
print("\n" + "=" * 60)
print(answer)