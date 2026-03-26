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
    "TOP_K": 20,
    "SIM_TH": 0.35
}

RETRIEVAL_QUERY = "PET glycolysis mild conditions ethylene glycol"

QUESTION = """
You are a university polymer chemistry lab instructor.

TASK: Adapt a PET glycolysis experiment FROM THE PROVIDED REFERENCES into a 2-hour undergraduate teaching laboratory lesson.

Basic Requirements (must follow)

Target audience: undergraduate students
Total duration: ~2 hours
Use only common teaching-lab chemicals and equipment
Reaction conditions must be safe, mild, and controllable
Difficulty suitable for students with basic organic/polymer chemistry knowledge

Write the output using EXACTLY these section headings

1. Experiment Title

2. Teaching Objectives

3. Background Theory

4. Materials and Reagents

5. Experimental Procedure

6. Time Allocation

7. Safety Considerations

8. Expected Observations and Discussion Points

9. Post-Lab Questions

Style Rules

Use formal instructional language.
Do not write a research paper.
Do not include irrelevant content.
The result should look like a ready-to-use undergraduate lab handout.
"""

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

def embed_query(query_text):
    start_time = time.time()
    query_vector = np.array(
        client.embeddings.create(
            model=CFG["EMB_MODEL"], input=query_text
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

    grounding_rules = """
You MUST base the experiment ONLY on the reference materials.

MANDATORY RULES:
1. Every experimental condition (temperature, time, catalyst, ratios, equipment) MUST be supported by the references.
2. You MUST cite supporting chunk numbers in square brackets like [1], [3].
3. Do NOT use outside knowledge.
"""

    messages = [
        {
            "role": "user",
            "content": f"""{grounding_rules}

Question:
{question_text}

Reference Materials:
{context_text}
"""
        }
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
You are evaluating the QUALITY OF EXPERIMENTAL DESIGN proposed by different models.

You must fill the following JSON template with integers from 1-10.
DO NOT add, remove, rename, or reorder any fields.

JSON TEMPLATE:
{{
  "evaluations": [
    {{"model": "deepseek-v3", "feasibility": X, "safety": X, "clarity_of_procedure": X}},
    {{"model": "deepseek-ai/DeepSeek-R1", "feasibility": X, "safety": X, "clarity_of_procedure": X}},
    {{"model": "gemini-2.5-flash", "feasibility": X, "safety": X, "clarity_of_procedure": X}},
    {{"model": "gpt-4o", "feasibility": X, "safety": X, "clarity_of_procedure": X}},
    {{"model": "gpt-5", "feasibility": X, "safety": X, "clarity_of_procedure": X}},
    {{"model": "qwen/qwen3-max", "feasibility": X, "safety": X, "clarity_of_procedure": X}}
  ]
}}

SCORING DEFINITIONS:
- feasibility: realistic under lab constraints
- safety: avoids risks
- clarity_of_procedure: clear and reproducible

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
        file.write(f"Question:\n{question_text}\n")
        file.write(f"Retrieval Query:\n{RETRIEVAL_QUERY}\n\n=== Retrieved Chunks ===\n")
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
    print("Multi-LLM RAG System | Auto Run Mode\n")

    question_text = QUESTION.strip()
    print(f"Question (for generation): {question_text}\n")
    print(f"Retrieval query (for vector search): {RETRIEVAL_QUERY}\n")

    query_vector, embedding_time = embed_query(RETRIEVAL_QUERY)
    retrieved_chunks, retrieval_time = retrieve_chunks(
        vector_index, metadata, query_vector
    )

    print("\n=== RETRIEVED CHUNKS ===")
    for i, c in enumerate(retrieved_chunks):
        print(f"[{i+1}] {c['file']} p{c['page']} sim={c['sim']}")
    print("-" * 60)

    model_answers, generation_stats = ask_all_models(question_text, retrieved_chunks)
    judge_results, judge_stats = judge_answers(question_text, retrieved_chunks, model_answers)

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
