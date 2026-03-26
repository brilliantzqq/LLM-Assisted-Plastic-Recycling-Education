import os, time, json, faiss, datetime, textwrap
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ================= CONFIG =================
load_dotenv()
CFG = {
    "VECTOR_DB": "paper_vector_db",
    "META": "paper_metadata.npy",
    "SAVE": "./qa_records",
    "EMB_MODEL": "text-embedding-3-small",
    "GEN_MODEL": "qwen/qwen3-max",
#"deepseek-v3","deepseek-ai/DeepSeek-R1","gemini-2.5-flash","gpt-4o","qwen/qwen3-max","gpt-5"
    "TOP_K": 50,
    "SIM_TH": 0.2
}

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
os.makedirs(CFG["SAVE"], exist_ok=True)

# ================= CORE =================
def load_db():
    return (
        faiss.read_index(CFG["VECTOR_DB"]),
        np.load(CFG["META"], allow_pickle=True).tolist()
    )

def embed_query(question_text):
    start = time.time()
    vec = np.array(
        client.embeddings.create(
            model=CFG["EMB_MODEL"],
            input=question_text
        ).data[0].embedding,
        dtype=np.float32
    )[None, :]
    faiss.normalize_L2(vec)
    return vec, round(time.time() - start, 3)

def retrieve_chunks(index, metadata, query_vec):
    start = time.time()
    distances, indices = index.search(query_vec, CFG["TOP_K"])
    chunks = [
        {
            "file": metadata[i]["file_name"],
            "page": metadata[i]["page"],
            "text": metadata[i]["chunk_text"],
            "sim": round(1 - distances[0][r], 3)
        }
        for r, i in enumerate(indices[0])
        if 1 - distances[0][r] >= CFG["SIM_TH"]
    ]
    return chunks, round(time.time() - start, 3)

def ask_model(question_text, chunks):
    context = "\n".join(
        f"[{i+1}] {c['file']} p{c['page']} s={c['sim']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an educator and an expert in plastic recycling. "
                "Answer strictly based on the provided paper chunks. "
                "Synthesize information from multiple chunks. "
                "Ensure clarity, logical structure, and student-friendly explanations. "
                "Only cite relevant literature with the corresponding RETRIEVED CHUNKS numbers."
            )
        },
        {
            "role": "user",
            "content": f"Question:\n{question_text}\n\nPaper chunks:\n{context}"
        }
    ]

    print(f"\n===== MODEL ANSWER ({CFG['GEN_MODEL']}) =====\n")
    start = time.time()
    response = client.chat.completions.create(
        model=CFG["GEN_MODEL"],
        messages=messages,
        temperature=0.2
    )
    elapsed = round(time.time() - start, 3)

    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens

    print(f"[{CFG['GEN_MODEL']}] {elapsed}s | {tokens} tokens\n")
    print(textwrap.fill(answer, 100))
    print("-" * 70)

    return answer, {"time": elapsed, "tokens": tokens}

def save_record(question, chunks, answer, timing):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{CFG['SAVE']}/single_{ts}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Question:\n{question}\n\n=== Retrieved Chunks ===\n")
        for i, c in enumerate(chunks):
            f.write(f"[{i+1}] {c['file']} p{c['page']} sim={c['sim']}\n")

        f.write(f"\n=== Model Answer ({CFG['GEN_MODEL']}) ===\n")
        f.write(textwrap.fill(answer, 100))

        f.write("\n\n=== Timing ===\n")
        f.write(json.dumps(timing, indent=2))

    print(f"\n[Saved] {path}\n")

# ================= MAIN =================
def main():
    index, metadata = load_db()
    print(f"Single-LLM RAG System ({CFG['GEN_MODEL']}) | type 'quit' to exit")

    while True:
        q = input("\nQ> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue

        q_vec, t_embed = embed_query(q)
        chunks, t_ret = retrieve_chunks(index, metadata, q_vec)

        print("\n=== RETRIEVED CHUNKS ===")
        for i, c in enumerate(chunks):
            print(f"[{i+1}] {c['file']} p{c['page']} sim={c['sim']}")
        print("-" * 60)

        answer, gen_stats = ask_model(q, chunks)

        save_record(
            q,
            chunks,
            answer,
            {
                "embed": t_embed,
                "retrieve": t_ret,
                "generation": gen_stats
            }
        )

if __name__ == "__main__":
    main()
