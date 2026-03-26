import os
import faiss
import numpy as np
import hashlib
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://reverse.onechats.top/v1"
)

PDF_FOLDER = "PDF"
VECTOR_DB_PATH = "paper_vector_db"
METADATA_PATH = "paper_metadata.npy"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_and_chunk_pdfs():
    texts, metas, hashes = [], [], []
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    if not pdfs:
        print("No PDF files found.")
        return texts, metas, hashes

    existing_meta = (
        np.load(METADATA_PATH, allow_pickle=True).tolist()
        if os.path.exists(METADATA_PATH)
        else []
    )
    existing_hashes = {m["hash"] for m in existing_meta}

    print(f"Processing {len(pdfs)} PDF(s)...")

    for pdf in pdfs:
        reader = PdfReader(os.path.join(PDF_FOLDER, pdf))
        for p, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if not text:
                continue

            words = text.split()
            step = CHUNK_SIZE - CHUNK_OVERLAP

            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i + CHUNK_SIZE])
                if len(chunk) < 80:
                    continue

                h = hash_text(chunk)
                if h in existing_hashes:
                    continue

                texts.append(chunk)
                metas.append({
                    "file_name": pdf,
                    "page": p,
                    "chunk_text": chunk
                })
                hashes.append(h)

        print(f"Finished {pdf} ({len(reader.pages)} pages)")

    print(f"Loaded {len(texts)} new chunks.")
    return texts, metas, hashes


def embed_texts(texts, batch=10):
    vectors = []
    total_tokens = 0

    print(f"Embedding {len(texts)} chunks...")

    for i in range(0, len(texts), batch):
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts[i:i + batch]
            )

            vectors.extend([x.embedding for x in resp.data])

            if hasattr(resp, "usage"):
                total_tokens += resp.usage.total_tokens

        except Exception as e:
            print(f"Embedding error at batch {i // batch}: {e}")

    vectors = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)

    return vectors, total_tokens



def update_index(new_vecs, new_meta, new_hashes):
    index = (
        faiss.read_index(VECTOR_DB_PATH)
        if os.path.exists(VECTOR_DB_PATH)
        else faiss.IndexFlatIP(EMBED_DIM)
    )

    existing_meta = (
        np.load(METADATA_PATH, allow_pickle=True).tolist()
        if os.path.exists(METADATA_PATH)
        else []
    )
    existing_hashes = {m["hash"] for m in existing_meta}

    add_vecs, add_meta = [], []

    for i, h in enumerate(new_hashes):
        if h not in existing_hashes:
            add_vecs.append(new_vecs[i])
            add_meta.append({**new_meta[i], "hash": h})

    if not add_vecs:
        print("Index already up to date.")
        return

    index.add(np.array(add_vecs, dtype=np.float32))
    faiss.write_index(index, VECTOR_DB_PATH)

    existing_meta.extend(add_meta)
    np.save(METADATA_PATH, existing_meta, allow_pickle=True)

    print(f"Added {len(add_vecs)} vectors to index.")


def main():
    print("=== PDF → Vector DB Update ===")

    texts, meta, hashes = load_and_chunk_pdfs()
    if not texts:
        return

    vecs, total_tokens = embed_texts(texts)
    print("\n=== Embedding Token Usage ===")
    print(f"Total Tokens : {total_tokens}")

    update_index(vecs, meta, hashes)


if __name__ == "__main__":
    main()
