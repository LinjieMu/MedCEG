# MedCEG: Reinforcing Verifiable Medical Reasoning with Critical Evidence Graph

[![huggingface](https://img.shields.io/badge/huggingface-available-yellow)](https://huggingface.co/ICLRAnonymous/MedCEG) [![arXiv](https://img.shields.io/badge/arXiv-2512.XXXXX-b31b1b.svg)](#)

**MedCEG** is a framework that augments medical language models with clinically valid reasoning pathways by explicitly supervising the reasoning process through a Critical Evidence Graph (CEG). 

## 🚀 Model Architecture & Pipeline

Our approach consists of two main stages:
1.  **Cold-Start**: We transform structured evidence graphs into natural language to teach the model logical dependencies.
2.  **Graph-guided Reinforcement Learning**: We use a Critical Evidence Graph (CEG) to provide dense, process-oriented rewards. 

![MedCEG Pipeline](images/pipeline.png)
*Figure: Overview of the MedCEG pipeline including Data Preparation and Graph-guided RL.* 

## 🛠️ Training with VeRL

We utilize [**VeRL**](https://github.com/volcengine/verl) (Volcengine Reinforcement Learning) for the RLHF/RLAIF stage. Our specific **Process Reward** functions provided in the `code/reward` directory.

### 1. Start the Embedding Service

To ensure efficient calculation of semantic similarity during training without reloading the model repeatedly, we use a standalone FastAPI server for embeddings. This server manages a pool of `BGE-LARGE-EN-V1.5` models across available GPUs.

**Setup:** Update the `BGE_PATH` variable in `code/server/EmbeddingServer.py` to point to your local model path.

**Launch:**

```Bash
python code/server/EmbeddingServer.py
```

*Note: Ensure this server is accessible to your training nodes (default port 8000).*

### 2. Integration with VeRL

The files in `code/reward/` contain the custom reward functions tailored for MedCEG.

- **`GraphReward.py`**: Implements the core logic for calculating the composite reward: `Node Coverage`, `Structural Correctness`, and `Chain Completeness`.
- **`GraphMCQ.py`** & **`GraphOpenendQuestion.py`**: These scripts contain the `compute_score()` function, which acts as the primary interface for the reward model.

The reward function evaluates:

1. **Format Reward**: Checks for proper `<think>...</think>` tag usage.
2. **Accuracy Reward**: Checks the final answer against Ground Truth.
3. **Process Reward**: Extracts the graph from the `<think>` trace and compares it against the Ground Truth CEG using the Embedding Server.

## 💻 Inference

The `Inference.py` script allows you to test the model. Our model weights are open-sourced at [HuggingFace](https://huggingface.co/LinjieMu/MedCEG).

```Python
import transformers
import torch

# 1. Load the model and tokenizer
model_id = "ICLRAnonymous/MedCEG"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 2. Define the question
question = "A 78-year-old Caucasian woman presented with..."
suffix = "\nPut your final answer in \\boxed{}."
messages = [{"role": "user", "content": question + suffix}]

# 3. Generate
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids, max_new_tokens=8196, do_sample=False)
decoded_response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(decoded_response)
```

## 📊 Data Description

`DataExample.jsonl` contains examples of our constructed data. Key fields include:

| **Key**                         | **Description**                                              |
| ------------------------------- | ------------------------------------------------------------ |
| `question`                      | The input clinical question.                                 |
| `answer`                        | The ground truth answer.                                     |
| `reasoning_content`             | The rewritten "thinking" process based on the CEG.           |
| `graph/triplets`                | The full Evidence Graph (EG).                                |
| `graph/core_reasoning_subgraph` | The **Critical Evidence Graph (CEG)** used for reward calculation. |

## 🏆 Experimental Results

### Main Results

<<<<<<< HEAD
MedCEG achieves state-of-the-art performance across multiple medical benchmarks, showing significant improvements in both accuracy and reasoning quality.

![main_results](./images/main-results.png)

*Table 1: Comprehensive performance comparison across ID and OOD benchmarks.* 

### Reasoning Process Quality

We evaluated the reasoning process across five dimensions: Logical Coherence, Factual Accuracy, Evidence Faithfulness, Interpretability & Clarity, and Information Utilization. MedCEG significantly outperforms baselines in producing clinically sound reasoning.

<img src="./images/process.png" alt="process_results" style="zoom:50%; display: block; margin: 0 auto;" />



*Figure 2: Multi-dimensional evaluation of the reasoning process.* 

## 📁 File Structure

```text
│   📄 ReadMe.md
│───📁 code
│   └───📄 DataExample.jsonl
├───📁 code
    ├───📁 evaluation
    │       📄 ProcessEvaluation.py   # Evaluate reasoning process precision
    │
    ├───📁 reward                     # Core Reward Logic for VeRL
    │       📄 graph_extract.py       # Extract triplets from reasoning text via LLM
    │       📄 GraphReward.py         # Calculate graph-based rewards (Node, Struct, Chain)
    │       📄 GraphMCQ.py            # Reward entry point for Multiple Choice Questions
    │       📄 GraphOpenendQuestion.py# Reward entry point for Open-ended Questions
    │       📄 TripletsRecall.py      # Utility for calculating triplet recall
    │
    └───📁 server
            📄 EmbeddingServer.py     # FastAPI server for BGE embeddings
```

## 
=======
| Benchmark File          | Correct | Total | Accuracy |
| ----------------------- | ------- | ----- | -------- |
| `DiagArena.jsonl`       | 459     | 915   | 50.16    |
| `MedBullets-5op.jsonl`  | 392     | 616   | 63.64    |
| `MedCase.jsonl`         | 283     | 897   | 31.55    |
| `MedQA.jsonl`           | 960     | 1273  | 75.41    |
| `MMLU-health.jsonl`     | 884     | 1089  | 81.18    |
| `MMLU-Pro-Health.jsonl` | 509     | 818   | 62.22    |
>>>>>>> 8fd9342115042447242b2a9b2e6a732086185818
