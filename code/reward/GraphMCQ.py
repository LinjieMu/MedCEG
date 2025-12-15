import json
import random
import re
from typing import List, Tuple
import os
from rich import print

from verl.utils.reward_score.graph_extract import generate_reasoning_graph
from verl.utils.reward_score.graph_reward import calculate_total_reward


Graph = List[Tuple[str, str, str]]

BASE_MODE_SERIES = os.environ["BASE_MODE_SERIES"]

def compute_process_similarity(process_str: str, 
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


def acc_reward(predict_str: str, ground_truth: str) -> float:
    predict_str = predict_str.replace("\\text{", "")
    matches = re.findall(r'\\boxed\{(.*?)\}', predict_str, re.DOTALL)
    if not matches:
        return 0.0

    extracted_answer = matches[-1].strip()
    if len(extracted_answer) == 0:
        return 0.0

    if ground_truth.strip()[0] == extracted_answer[0]:
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
        
    return 0.2 * fmt_score + 0.6 * acc_score + 0.2 * think_score