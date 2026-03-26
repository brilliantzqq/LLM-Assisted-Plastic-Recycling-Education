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
    "GEN_MODELS": [
        "gpt-5",
        "deepseek-v3",
        "deepseek-ai/DeepSeek-R1",
        "gemini-2.5-flash",
        "gpt-4o",
        "qwen/qwen3-max"
    ],
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
    start_time = time.time()
    query_vector = np.array(
        client.embeddings.create(
            model=CFG["EMB_MODEL"], input=question_text
        ).data[0].embedding,
        dtype=np.float32
    )[None, :]
    faiss.normalize_L2(query_vector)
    return query_vector, round(time.time() - start_time, 3)

def retrieve_chunks(vector_index, metadata, query_vector):
    start_time = time.time()
    distances, indices = vector_index.search(query_vector, CFG["TOP_K"])
    retrieved_chunks = [
        {
            "file": metadata[index]["file_name"],
            "page": metadata[index]["page"],
            "text": metadata[index]["chunk_text"],
            "sim": round(1 - distances[0][rank], 3)
        }
        for rank, index in enumerate(indices[0])
        if 1 - distances[0][rank] >= CFG["SIM_TH"]
    ]
    return retrieved_chunks, round(time.time() - start_time, 3)

def ask_all_models(question_text, retrieved_chunks):
    context_text = "\n".join(
        f"[{i+1}] {c['file']} p{c['page']} s={c['sim']}\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    messages = [
        {
            "role": "system",
            "content": (
                        "You are an educator and an expert in plastic recycling "
                        "Prioritize using original text information to ensure accuracy "
                        "Ensure explanations are clear, logically structured, and accessible to student learners. "
                        "Synthesize information from multiple chunks"
                        "Only cite relevant literature with the corresponding RETRIEVED CHUNKS numbers."
            )
        },
        {"role": "user", "content": f"Question:\n{question_text}\n\nPaper chunks:\n{context_text}"}
    ]

    model_answers, generation_stats = {}, {}
    print("\n===== MULTI-LLM ANSWERS =====\n")

    for model_name in CFG["GEN_MODELS"]:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.2
        )
        elapsed_time = round(time.time() - start_time, 3)
        token_usage = response.usage.total_tokens
        answer_text = response.choices[0].message.content

        model_answers[model_name] = answer_text
        generation_stats[model_name] = {"time": elapsed_time, "tokens": token_usage}

        print(f"[{model_name}] {elapsed_time}s | {token_usage} tokens")
        print(textwrap.fill(answer_text, 100))
        print("-" * 70)

    return model_answers, generation_stats

def judge_answers(question_text, retrieved_chunks, model_answers):
    evidence_text = "\n".join(
        f"[{i+1}] {c['file']} p{c['page']}\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    answers_text = "\n\n".join(
        f"Model: {model}\n{answer}"
        for model, answer in model_answers.items()
    )

    judge_prompt = f"""
You must fill the following JSON template with 1-10.
DO NOT add, remove, rename, or reorder any fields.
DO NOT omit any item.
JSON TEMPLATE (fill numbers only):
{{
  "evaluations": [
    {{
      "model": "deepseek-v3",
      "faithfulness": X,
      "coverage": X,
      "relevance": X,
      "clarity": X
    }},
    {{
      "model": "deepseek-ai/DeepSeek-R1",
      "faithfulness": X,
      "coverage": X,
      "relevance": X,
      "clarity": X
    }},
    {{
      "model": "gemini-2.5-flash",
      "faithfulness": X,
      "coverage": X,
      "relevance": X,
      "clarity": X
    }},
    {{
      "model": "gpt-4o",
      "faithfulness": X,
      "coverage": X,
      "relevance": X,
      "clarity": X
    }},
    {{
      "model": "gpt-5",
      "faithfulness": X,
      "coverage": X,
      "relevance": X,
      "clarity": X
    }},
    {{
      "model": "qwen/qwen3-max",
      "faithfulness": X,
      "coverage": X,
      "relevance": X,
      "clarity": X
    }}
  ]
}}

SCORING DEFINITIONS:
- faithfulness: whether the answer is based on evidence
- coverage: whether the content of the answer is complete
- relevance: whether the answer directly addresses the question
- clarity: whether the answer is well-structured and easy to understand

Question:
{question_text}

Evidence:
{evidence_text}

Model answers:
{answers_text}

Return ONLY the JSON object.
"""

    print("\n===== MULTI-JUDGE =====\n")
    judge_results, judge_stats = {}, {}

    for judge_model in CFG["GEN_MODELS"]:
        print(f"[Judge: {judge_model}]")
        start_time = time.time()
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        )
        elapsed_time = round(time.time() - start_time, 3)
        token_usage = response.usage.total_tokens
        judge_output = response.choices[0].message.content

        print(judge_output)
        print(f"({elapsed_time}s | {token_usage} tokens)")
        print("-" * 60)

        judge_results[judge_model] = judge_output
        judge_stats[judge_model] = {"time": elapsed_time, "tokens": token_usage}

    return judge_results, judge_stats

def save_record(question_text, retrieved_chunks, model_answers, judge_results, timing_info):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"{CFG['SAVE']}/multi_{timestamp}.txt"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Question:\n{question_text}\n\n=== Retrieved Chunks ===\n")
        for i, c in enumerate(retrieved_chunks):
            file.write(f"[{i+1}] {c['file']} p{c['page']} sim={c['sim']}\n")

        file.write("\n=== Model Answers ===\n")
        for model_name, answer in model_answers.items():
            file.write(f"\n[{model_name}]\n{textwrap.fill(answer,100)}\n")

        file.write("\n=== Multi-Judge ===\n")
        for judge_model, output in judge_results.items():
            file.write(f"\n[Judge: {judge_model}]\n{output}\n")

        file.write("\n\n=== Timing ===\n" + json.dumps(timing_info, indent=2))

    print(f"\n[Saved] {file_path}\n")

# ================= MAIN =================
def main():
    vector_index, metadata = load_db()
    print("Multi-LLM RAG System | type 'quit' to exit")

    while True:
        question_text = input("\nQ> ").strip()
        if question_text.lower() in ("quit", "exit"):
            break
        if not question_text:
            continue

        query_vector, embedding_time = embed_query(question_text)
        retrieved_chunks, retrieval_time = retrieve_chunks(
            vector_index, metadata, query_vector
        )

        print("\n=== RETRIEVED CHUNKS ===")
        for i, c in enumerate(retrieved_chunks):
            print(f"[{i+1}] {c['file']} p{c['page']} sim={c['sim']}")
        print("-" * 60)

        model_answers, generation_stats = ask_all_models(
            question_text, retrieved_chunks
        )
        judge_results, judge_stats = judge_answers(
            question_text, retrieved_chunks, model_answers
        )

        save_record(
            question_text,
            retrieved_chunks,
            model_answers,
            judge_results,
            {
                "embed": embedding_time,
                "retrieve": retrieval_time,
                "generation": generation_stats,
                "judge": judge_stats
            }
        )

if __name__ == "__main__":
    main()
