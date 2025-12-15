import json
import os
import random
import re
from typing import List, Tuple

from openai import OpenAI
from rich import print

from verl.utils.reward_score.graph_extract import generate_reasoning_graph
from verl.utils.reward_score.graph_reward import calculate_total_reward

BASE_MODE_SERIES = os.environ["BASE_MODE_SERIES"]
Graph = List[Tuple[str, str, str]]
BASE_URL = os.environ.get("REWARD_MODEL_URL", "http://XXX.XXX.XXX:8000/v1")
client = OpenAI(base_url=BASE_URL, api_key="Not_Needed")
model_name = "default"

def compute_process_similarity(
    process_str: str, 
    gt_graph: Graph, 
    threshold_entity: float,
    threshold_relation: float,
    reward_weights: dict[str, float],
    debug: bool = False
):
    generated_graph = generate_reasoning_graph(process_str)
    
    if debug:
        print(f"Generated graph from process string: \n{generated_graph}")

    if not generated_graph:
        return 0.0
    
    similarity = calculate_total_reward(
        generated_graph["triplets"], gt_graph, reward_weights,
        entity_threshold=threshold_entity, relation_threshold=threshold_relation
    )
    
    if debug:
        print(f"{similarity=}")
    return 1.0 if similarity["total_reward"] >= 0.8 else similarity["total_reward"]

def format_reward_qwen3(predict_str: str) -> float:
    pattern = re.compile(r"^<think>.*</think>.*$", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str.strip())
    return 1.0 if match_result else 0.0

def format_reward_llama3(predict_str: str) -> float:
    pattern = re.compile(r"^## Thinking\n\n.*\n\n## Final Response\n\n.*$", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str.strip())
    return 1.0 if match_result else 0.0

def format_reward(predict_str: str) -> float:
    if BASE_MODE_SERIES == "qwen3":
        return format_reward_qwen3(predict_str)
    elif BASE_MODE_SERIES == "llama3.1":
        return format_reward_llama3(predict_str)
    else:
        raise ValueError(f"Unsupported BASE_MODE_SERIES for reasoning extraction: {BASE_MODE_SERIES}")

def llm_judge(extracted_answer, ground_truth, max_tries=3):
    prompt = f"""Is our predicted diagnosis correct (yes/no)?
Predicted diagnosis: {extracted_answer}
True diagnosis: {ground_truth}
Your response should be a word, "yes" or "no"."""
    for _ in range(max_tries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                reasoning_effort="medium",
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            content = response.choices[0].message.content
            if not content: continue
            normalized_content = content.replace("*","").replace("\"","").strip().lower()
            if normalized_content in ["yes", "yes.", "[yes]", "[yes.]", "answer: [yes]", "answer: [yes.]", "answer:[yes]", "answer:[yes.]"]:
                return True
            elif normalized_content in ["no", "no.", "[no]", "[no.]", "answer: [no]", "answer: [no.]", "answer:[no]", "answer:[no.]"]:
                return False
            raise ValueError(f"Err Content: {content}")
        except Exception as e:
            print(e)
    return False

def acc_reward(predict_str: str, ground_truth: str) -> float:
    predict_str = predict_str.replace("\\text{", "")
    matches = re.findall(r'\\boxed\{(.*?)\}', predict_str, re.DOTALL)
    if not matches:
        return 0.0

    extracted_answer = matches[-1].strip()
    if len(extracted_answer) == 0:
        return 0.0

    if llm_judge(extracted_answer, ground_truth):
        return 1.0
    else:
        return 0.0

def compute_score(predict_str: str, ground_truth: str, extra_info: dict, debug: bool = False) -> float:
    fmt_score = format_reward(predict_str)
    acc_score = acc_reward(predict_str, ground_truth)

    reasoning_content = []
    if BASE_MODE_SERIES == "qwen3":
        reasoning_content = re.findall(r'<think>(.*?)</think>', predict_str, re.DOTALL)
    elif BASE_MODE_SERIES == "llama3.1" or BASE_MODE_SERIES == "llama3":
        # Extract content between "## Thinking" and "## Final Response"
        reasoning_content = re.findall(r'## Thinking\n\n(.*?)\n\n## Final Response', predict_str, re.DOTALL)
    else:
        raise ValueError(f"Unsupported BASE_MODE_SERIES for reasoning extraction: {BASE_MODE_SERIES}")

    if not reasoning_content:
        think_score = 0.0
    else:
        think_score = compute_process_similarity(reasoning_content[-1], 
                                                 extra_info["core_reasoning_subgraph"],
                                                 threshold_entity=0.7,
                                                 threshold_relation=0.6,
                                                 reward_weights={"node": 0.5, "structure": 0.3, "chain": 0.2},
                                                 debug=debug
                                                )
    
    
    if random.random() * 64 < 1:
        print(f"Question: {extra_info.get('question', 'N/A')}\nPrediction: {predict_str}\nAnswer: {ground_truth}\nReward: Format - {fmt_score}, Acc - {acc_score}")
    if debug:
        print(f"Prediction: {predict_str}\nAnswer: {ground_truth}\nReward: Format - {fmt_score}, Acc - {acc_score}")
        
    return 0.2 * fmt_score + 0.6 * acc_score + 0.2 * think_score

