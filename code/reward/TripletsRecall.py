import sys
from itertools import product
from typing import Any, List, Set, Tuple

import numpy as np
import requests
import torch
from rich import print
from sentence_transformers import util

API_URL = "http://127.0.0.1:8000/encode"

def get_embeddings_from_api(texts: List[str]) -> np.ndarray | None:
    if not texts:
        return np.array([])
        
    try:
        response = requests.post(API_URL, json={"texts": texts})
        response.raise_for_status() 
        
        data = response.json()
        return np.array(data["embeddings"], dtype=np.float32)

    except requests.exceptions.RequestException as e:
        print(f"[bold red]API Call Failed:[/bold red] Could not connect to the server at {API_URL}. Error: {e}", file=sys.stderr)
        print("Please ensure the FastAPI model server is running.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[bold red]An unexpected error occurred:[/bold red] {e}", file=sys.stderr)
        return None

Triple = Tuple[Any, Any, Any]
Graph = List[Triple]

def check_graph_format(graph: Any, debug: bool = False) -> bool:
    if not isinstance(graph, list):
        if debug:
            print("[bold red]Error:[/bold red] Graph is not a List.")
        return False
    for item in graph:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            if debug:
                print(f"[bold red]Error:[/bold red] Non-triple element found: {item}")
            return False
    return True

def get_unique_elements(graph: Graph) -> Set[Any]:
    elements = set()
    for s, p, o in graph:
        elements.add(s)
        elements.add(p)
        elements.add(o)
    return elements

def calculate_recall(
    generated_graph: Graph,
    gt_graph: Graph,
    threshold1: float = 0.75,
    threshold2: float = 0.6,
    debug: bool = False,
) -> float:
    if debug: print("\n--- [DEBUG] Starting Format Check ---")
    if not check_graph_format(generated_graph, debug) or not check_graph_format(gt_graph, debug):
        if debug: print("[bold red]Error:[/bold red] Input graph format is incorrect. Aborting.")
        return 0.0
    if debug: print("[green]Format check passed.[/green]")

    if not gt_graph: return 1.0 if not generated_graph else 0.0
    if not generated_graph: return 0.0

    if debug: print("\n--- [DEBUG] Starting Element Encoding via API ---")
    gen_elements_list = list(get_unique_elements(generated_graph))
    gt_elements_list = list(get_unique_elements(gt_graph))
    
    if not gen_elements_list or not gt_elements_list:
        if debug: print("[yellow]Warning:[/yellow] One of the graphs has no unique elements to encode.")
        return 0.0

    all_elements = gen_elements_list + gt_elements_list
    all_elements_str = [str(e) for e in all_elements]
    
    all_embeddings_np = get_embeddings_from_api(all_elements_str)
    
    if all_embeddings_np is None:
        print("[bold red]CRITICAL: Failed to retrieve embeddings from API. Aborting calculation.[/bold red]", file=sys.stderr)
        return 0.0

    len_gen = len(gen_elements_list)
    gen_embeddings = torch.from_numpy(all_embeddings_np[:len_gen])
    gt_embeddings = torch.from_numpy(all_embeddings_np[len_gen:])
    
    similarity_matrix = util.cos_sim(gen_embeddings, gt_embeddings).cpu().numpy()

    gen_to_gt_mappings = {gen_elem: [] for gen_elem in gen_elements_list}
    if debug: print(f"\n--- [DEBUG] Element Mapping Details (Threshold >= {threshold1}) ---")
        
    for i, gen_elem in enumerate(gen_elements_list):
        similar_gt_indices = np.where(similarity_matrix[i] >= threshold1)[0]
        if len(similar_gt_indices) > 0:
            for idx in similar_gt_indices:
                gen_to_gt_mappings[gen_elem].append(gt_elements_list[idx])
    
    if debug: print(f"\n--- [DEBUG] Recall Analysis (Jaccard Threshold >= {threshold2}) ---")
    recalled_gt_triples = set()
    gt_graph_unique = list(set(map(tuple, gt_graph)))
    for i, gt_triple in enumerate(gt_graph_unique):
        gt_triple_set = set(gt_triple)
        best_jaccard_for_this_gt = -1.0
        best_gen_triple_match = None
        best_candidate_for_this_gt = None
        if debug:
            print("\n" + "-" * 20)
            print(f"[bold]Analyzing GT Triple #{i+1}:[/bold] {gt_triple}")
        for gen_triple in generated_graph:
            s, p, o = gen_triple
            s_mappings, p_mappings, o_mappings = gen_to_gt_mappings.get(s, []), gen_to_gt_mappings.get(p, []), gen_to_gt_mappings.get(o, [])
            if not (s_mappings and p_mappings and o_mappings):
                continue
            candidate_triples = product(s_mappings, p_mappings, o_mappings)
            for candidate in candidate_triples:
                candidate_set = set(candidate)
                intersection_size = len(candidate_set.intersection(gt_triple_set))
                union_size = len(candidate_set.union(gt_triple_set))
                jaccard = intersection_size / union_size if union_size > 0 else 1.0
                if jaccard > best_jaccard_for_this_gt:
                    best_jaccard_for_this_gt, best_gen_triple_match, best_candidate_for_this_gt = jaccard, gen_triple, candidate
        is_recalled = best_jaccard_for_this_gt >= threshold2
        if is_recalled:
            recalled_gt_triples.add(gt_triple)
        if debug:
            if best_gen_triple_match:
                print(f"   -> Best match in generated graph: {best_gen_triple_match}")
                print(f"     (Mapped to candidate: {best_candidate_for_this_gt})")
                print(f"   -> [cyan]Best Jaccard Similarity:[/cyan] {best_jaccard_for_this_gt:.4f}")
            else:
                print("   -> No fully mappable triple found in the generated graph.")
            if is_recalled:
                print("   -> [bold green]Final Status: RECALLED[/bold green]")
            else:
                print(f"   -> [bold red]Final Status: FAILED[/bold red] (Best Jaccard Sim <= {threshold2})")
    
    recall = len(recalled_gt_triples) / len(gt_graph_unique) if gt_graph_unique else 1.0
    if debug:
        print("\n" + "="*50)
        print("--- [DEBUG] Final Summary ---")
        print(f"Unique GT triples recalled: {len(recalled_gt_triples)}")
        print(f"Total unique GT triples: {len(gt_graph_unique)}")
        print(f"[bold magenta]Final Recall Rate: {recall:.4f}[/bold magenta]")
        print("="*50)
    return recall


def calculate_average_max_jaccard_similarity(
    generated_graph: Graph,
    gt_graph: Graph,
    threshold1: float = 0.75,
    debug: bool = False,
) -> float:
    if not check_graph_format(generated_graph, debug) or not check_graph_format(gt_graph, debug):
        return 0.0

    if not gt_graph or not generated_graph:
        return 0.0

    gen_elements_list = list(get_unique_elements(generated_graph))
    gt_elements_list = list(get_unique_elements(gt_graph))

    if not gen_elements_list or not gt_elements_list:
        return 0.0
    
    all_elements = gen_elements_list + gt_elements_list
    all_elements_str = [str(e) for e in all_elements]
    
    all_embeddings_np = get_embeddings_from_api(all_elements_str)
    
    if all_embeddings_np is None:
        print("[bold red]CRITICAL: Failed to retrieve embeddings from API. Aborting calculation.[/bold red]", file=sys.stderr)
        return 0.0
    
    len_gen = len(gen_elements_list)
    gen_embeddings = torch.from_numpy(all_embeddings_np[:len_gen])
    gt_embeddings = torch.from_numpy(all_embeddings_np[len_gen:])

    similarity_matrix = util.cos_sim(gen_embeddings, gt_embeddings).cpu().numpy()
    gen_to_gt_mappings = {gen_elem: [] for gen_elem in gen_elements_list}
    for i, gen_elem in enumerate(gen_elements_list):
        similar_gt_indices = np.where(similarity_matrix[i] >= threshold1)[0]
        if len(similar_gt_indices) > 0:
            for idx in similar_gt_indices:
                gen_to_gt_mappings[gen_elem].append(gt_elements_list[idx])

    total_max_jaccard = 0.0
    gt_graph_unique = list(set(map(tuple, gt_graph)))
    
    for gt_triple in gt_graph_unique:
        gt_triple_set = set(gt_triple)
        max_jaccard_for_gt = 0.0
        for gen_triple in generated_graph:
            s, p, o = gen_triple
            s_mappings = gen_to_gt_mappings.get(s, [])
            p_mappings = gen_to_gt_mappings.get(p, [])
            o_mappings = gen_to_gt_mappings.get(o, [])
            
            if not (s_mappings and p_mappings and o_mappings):
                continue

            candidate_triples = product(s_mappings, p_mappings, o_mappings)
            for candidate in candidate_triples:
                candidate_set = set(candidate)
                intersection_size = len(candidate_set.intersection(gt_triple_set))
                union_size = len(candidate_set.union(gt_triple_set))
                jaccard = intersection_size / union_size if union_size > 0 else 1.0
                if jaccard > max_jaccard_for_gt:
                    max_jaccard_for_gt = jaccard
        
        total_max_jaccard += max_jaccard_for_gt
        
    average_max_jaccard = total_max_jaccard / len(gt_graph_unique) if gt_graph_unique else 1.0
    return average_max_jaccard