# Supplementary Material for MedCEG: Reinforcing Verifiable Medical Reasoning with Critical Evidence Graph

[![huggingface](https://img.shields.io/badge/huggingface-available-yellow)](https://huggingface.co/ICLRAnonymous/MedCEG)

## 📁 File Structure

```
│   📄 DataExample.jsonl
│   📄 ReadMe.md
│
├───📁 code
│       📄 EmbeddingServer.py
│       📄 GraphExtract.py
│       📄 GraphReward.py
│       📄 Inference.py
│       📄 ProcessEvaluation.py
│
└───📁 results
        📄 DiagArena.jsonl
        📄 MedBullets-5op.jsonl
        📄 MedCase.jsonl
        📄 MedQA.jsonl
        📄 MMLU-health.jsonl
        📄 MMLU-Pro-Health.jsonl
```

## 💻 Code Description

The `code` directory contains the scripts used for inference and training.

- **🚀 `Inference.py`** This is the main script for running inference with our model. To test your own questions, simply replace the `question` variable within the script. **Please do not modify the `suffix` variable** to ensure proper model performance. Our model weights are open-sourced at https://huggingface.co/ICLRAnonymous/MedCEG. You can either download them directly and replace the path, or run the following code directly.

  ```
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
  
  # 2. Define the question and create the message prompt
  # Replace this with your own question
  question = "A 78-year-old Caucasian woman presented with..."
  suffix = "\nPut your final answer in \\boxed{}."
  messages = [
      {"role": "user", "content": question + suffix},
  ]
  
  # 3. Apply chat template and tokenize the input
  input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(model.device)
  
  # 4. Generate the response
  outputs = model.generate(
      input_ids,
      max_new_tokens=8196,
      do_sample=False,
  )
  
  # 5. Decode and print the output
  response_ids = outputs[0][input_ids.shape[-1]:]
  decoded_response = tokenizer.decode(response_ids, skip_special_tokens=True)
  print(decoded_response)
  ```

- **🌐 `EmbeddingServer.py`** This script launches a local server for the `BGE-LARGE-EN-V1.5` model, which is used to generate embeddings for reward calculation.

- **🧠 `GraphExtract.py`** This script was used during the Reinforcement Learning (RL) training process. It leverages the `GPT-OSS-120B` model to extract knowledge graphs from text.

- **💰 `GraphReward.py`** This script is responsible for calculating the reward signal during RL training. Please note that some core functionalities in this file have been replaced with comments. **The complete source code will be made available upon the publication of our paper.**

- **🧪 `ProcessEvaluation.py`** This script is used for evaluating the model's reasoning process. It reads the .jsonl files generated during inference, and compares it against the scoring standard to calculate process precision scores for each benchmark.

## 📊 Data Description

- **`DataExample.jsonl`** This file provides 12 examples of our constructed data to illustrate the format. Each JSON object in the file contains the following fields:

| Key                             | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `data_source`                   | The original source of the data (e.g., "MedCase").           |
| `index`                         | A unique identifier for the data point.                      |
| `question`                      | The input question for the model.                            |
| `answer`                        | The ground truth or standard answer.                         |
| `reasoning_content`             | The rewritten "thinking" process of the model, based on the Critical Evidence Graph (CEG). |
| `content`                       | The final response.                                          |
| `graph/triplets`                | The full Evidence Graph (EG) extracted from the reasoning process. |
| `graph/core_reasoning_subgraph` | The Critical Evidence Graph (CEG) which represents the most crucial reasoning steps. |

## 🏆 Results

The `results` directory contains the raw output files from our model's evaluation on several benchmarks. The performance summary is as follows:

| Benchmark File          | Correct | Total | Accuracy |
| ----------------------- | ------- | ----- | -------- |
| `DiagArena.jsonl`       | 459     | 915   | 50.16    |
| `MedBullets-5op.jsonl`  | 392     | 616   | 63.64    |
| `MedCase.jsonl`         | 283     | 897   | 31.55    |
| `MedQA.jsonl`           | 960     | 1273  | 75.41    |
| `MMLU-health.jsonl`     | 884     | 1089  | 81.18    |
| `MMLU-Pro-Health.jsonl` | 509     | 818   | 62.22    |