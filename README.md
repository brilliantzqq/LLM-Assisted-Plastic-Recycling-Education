# LLM-Assisted-Plastic-Recycling-Education
Codes for "Integrating Retrieval-Augmented Large Language Models into Plastic Recycling Education" (no more update after 26th, March, 2026)

## Usage

### 1. Environment Variables

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_base_url   
```

---

### 2. Data Preparation

Place your PDF files in the `PDF/` folder:

```bash
project/
├── PDF/
│   ├── paper1.pdf
│   ├── paper2.pdf
```

Then build the vector database:

```bash
python build_vector_db.py
```

---

### 3. Core Configuration

Key parameters can be modified in each script via the `CFG` dictionary:

```python
CFG = {
    "VECTOR_DB": "paper_vector_db",   # FAISS index path
    "META": "paper_metadata.npy",     # metadata storage
    "EMB_MODEL": "text-embedding-3-small",
    "GEN_MODELS": [...],              # multi-LLM list
    "TOP_K": 20,                      # retrieval depth
    "SIM_TH": 0.35                    # similarity threshold
}
```

---

### 4. Running Modes

**Single-model QA**
```bash
python qa_single_model.py
```

**Multi-model QA + cross-evaluation**
```bash
python qa_cross_evaluation.py
```

**Automated experiment generation**
```bash
python rag_multi_llm_exp.py
```

## build_vector_db.py
- `build_vector_db.py` builds a vector database from PDF documents. It reads files from the `PDF/` directory, splits the text into chunks, and generates embeddings to store in a FAISS index along with metadata.
- A hashing-based deduplication mechanism is used to avoid duplicate entries in the database.
- To use this script, place your PDF files in the `PDF/` folder, configure the API environment variables, and run the script to build the vector database.

## qa_single_model.py
- `qa_single_model.py` implements a single-LLM retrieval-augmented generation (RAG) pipeline. It retrieves relevant document chunks based on the query and uses them as context for answer generation.
- The model is instructed to generate answers grounded in retrieved evidence, with clear explanations and proper citation.
- This script is suitable for baseline testing and single-model QA tasks.

## qa_cross_evaluation.py
- `qa_cross_evaluation.py` implements a multi-LLM question answering and cross-evaluation framework. Multiple models generate answers to the same query.
- Each model also acts as a judge to evaluate all generated answers based on faithfulness, coverage, relevance, and clarity.
- The script automatically saves retrieved content, model outputs, evaluation results, and runtime statistics for further analysis.

## rag_multi_llm_exp.py
- `rag_multi_llm_exp.py` is designed for automated experimental task generation using a multi-LLM RAG framework.
- It retrieves relevant literature based on a predefined query and prompts multiple models to generate structured outputs (e.g., teaching lab design).
- A multi-model evaluation mechanism is applied to assess the quality of generated results in terms of feasibility, safety, and clarity.
- This script is suitable for testing LLM performance in complex, structured educational tasks.
