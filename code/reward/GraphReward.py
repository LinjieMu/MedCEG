import sys
from typing import Any, List, Set, Tuple, Dict, Optional
from collections import defaultdict, deque

import numpy as np
import requests
import torch
from rich import print
from sentence_transformers import util

API_URL = "http://127.0.0.1:8000/encode"

Triple = Tuple[Any, Any, Any]
Graph = List[Triple]
AdjacencyList = Dict[Any, List[Any]]

def get_embeddings_from_api(texts: List[str]) -> np.ndarray | None:
    if not texts:
        return np.array([])
    try:
        response = requests.post(API_URL, json={"texts": texts})
        response.raise_for_status()
        data = response.json()
        return np.array(data["embeddings"], dtype=np.float32)
    except requests.exceptions.RequestException as e:
        print(f"[bold red]API Call Failed:[/bold red] Could not connect to server {API_URL}. Error: {e}", file=sys.stderr)
        print("Please ensure the FastAPI model server is running.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[bold red]An unexpected error occurred:[/bold red] {e}", file=sys.stderr)
        return None

def check_graph_format(graph: Any, debug: bool = False) -> bool:
    if not isinstance(graph, list):
        if debug: print("[bold red]Error:[/bold red] Graph is not a List.")
        return False
    for item in graph:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            if debug: print(f"[bold red]Error:[/bold red] Non-triple element found: {item}")
            return False
    return True

def get_unique_elements(graph: Graph) -> Set[Any]:
    elements = set()
    for s, p, o in graph:
        elements.update([s, p, o])
    return elements

def build_graph_structures(graph: Graph) -> Tuple[AdjacencyList, Set[Any], Set[Tuple[Any, Any]], Set[Triple]]:
    adj = defaultdict(list)
    nodes = set()
    edges = set()
    triples = set()
    for s, p, o in graph:
        if s != o:
            adj[s].append(o)
        nodes.add(s)
        nodes.add(o)
        edges.add((s, o))
        triples.add((s, p, o))
    return adj, nodes, edges, triples

def _setup_graph_comparison(
    generated_graph: Graph,
    gt_graph: Graph,
    entity_threshold: float,
    relation_threshold: float,
) -> Optional[Dict[str, Any]]:
    if not check_graph_format(generated_graph) or not check_graph_format(gt_graph): return None
    if not generated_graph or not gt_graph: return None

    gen_elements_list = list(get_unique_elements(generated_graph))
    gt_elements_list = list(get_unique_elements(gt_graph))
    
    if not gen_elements_list or not gt_elements_list: return None

    all_elements = gen_elements_list + gt_elements_list
    all_elements_str = [str(e) for e in all_elements]
    all_embeddings_np = get_embeddings_from_api(all_elements_str)
    
    if all_embeddings_np is None:
        print("[bold red]Error:[/bold red] Failed to retrieve embeddings from API. Aborting calculation.", file=sys.stderr)
        return None

    len_gen = len(gen_elements_list)
    gen_embeddings = torch.from_numpy(all_embeddings_np[:len_gen])
    gt_embeddings = torch.from_numpy(all_embeddings_np[len_gen:])
    similarity_matrix = util.cos_sim(gen_embeddings, gt_embeddings).numpy()

    gt_relations_set = {p for _, p, _ in gt_graph}
    threshold_matrix = np.full(similarity_matrix.shape, entity_threshold)
    gt_relation_indices = [i for i, elem in enumerate(gt_elements_list) if elem in gt_relations_set]
    threshold_matrix[:, gt_relation_indices] = relation_threshold
    indices = np.argwhere(similarity_matrix >= threshold_matrix)
    
    gt_to_gen_mappings = defaultdict(list)
    for i, j in indices:
        gen_elem, gt_elem = gen_elements_list[i], gt_elements_list[j]
        gt_to_gen_mappings[gt_elem].append(gen_elem)

    gt_adj, gt_nodes, _, gt_triples = build_graph_structures(gt_graph)
    _, gen_nodes, _, gen_triples = build_graph_structures(generated_graph)

    return {
        "gt_nodes": gt_nodes, "gt_triples": gt_triples, "gt_adj": gt_adj,
        "gen_nodes": gen_nodes, "gen_triples": gen_triples,
        "gen_elements_list": gen_elements_list, "gt_elements_list": gt_elements_list,
        "similarity_matrix": torch.from_numpy(similarity_matrix),
        "gt_to_gen_mappings": gt_to_gen_mappings,
    }

def _compute_node_coverage(setup_data: Dict[str, Any]) -> float:
    gt_nodes, gen_nodes = setup_data["gt_nodes"], setup_data["gen_nodes"]
    gt_elements_list, gen_elements_list = setup_data["gt_elements_list"], setup_data["gen_elements_list"]
    similarity_matrix = setup_data["similarity_matrix"]

    if not gt_nodes or not gen_nodes: return 0.0
    total_max_similarity = 0.0
    try:
        gen_node_indices = [gen_elements_list.index(node) for node in gen_nodes if node in gen_elements_list]
    except ValueError: return 0.0
    if not gen_node_indices: return 0.0

    for gt_node in gt_nodes:
        try:
            gt_idx = gt_elements_list.index(gt_node)
            sim_with_gen_nodes = similarity_matrix[gen_node_indices, gt_idx]
            max_sim = torch.max(sim_with_gen_nodes).item() if len(sim_with_gen_nodes) > 0 else 0.0
            total_max_similarity += max_sim
        except ValueError: continue
    return total_max_similarity / len(gt_nodes) if gt_nodes else 1.0

def _compute_structural_correctness(setup_data: Dict[str, Any]) -> float:
    gt_triples, gen_triples = setup_data["gt_triples"], setup_data["gen_triples"]
    gt_to_gen_mappings = setup_data["gt_to_gen_mappings"]

    if not gt_triples: return 1.0
    recalled_triples_count = 0
    for s, p, o in gt_triples:
        s_map, p_map, o_map = gt_to_gen_mappings.get(s,[]), gt_to_gen_mappings.get(p,[]), gt_to_gen_mappings.get(o,[])
        if not all([s_map, p_map, o_map]): continue
        if any((s_m, p_m, o_m) in gen_triples for s_m in s_map for p_m in p_map for o_m in o_map):
            recalled_triples_count += 1
    return recalled_triples_count / len(gt_triples) if gt_triples else 1.0

def _compute_chain_completeness(setup_data: Dict[str, Any]) -> float:
    gt_triples, gen_triples = setup_data["gt_triples"], setup_data["gen_triples"]
    gt_to_gen_mappings = setup_data["gt_to_gen_mappings"]
    
    total_gt_triples = len(gt_triples)
    if total_gt_triples == 0:
        return 1.0

    recalled_gt_triples = set()
    for s, p, o in gt_triples:
        s_map, p_map, o_map = gt_to_gen_mappings.get(s,[]), gt_to_gen_mappings.get(p,[]), gt_to_gen_mappings.get(o,[])
        if not all([s_map, p_map, o_map]): continue
        if any((s_m, p_m, o_m) in gen_triples for s_m in s_map for p_m in p_map for o_m in o_map):
            recalled_gt_triples.add((s, p, o))
    
    if not recalled_gt_triples:
        return 0.0

    adj = defaultdict(set)
    recalled_nodes = set()
    for s, p, o in recalled_gt_triples:
        adj[s].add(o)
        adj[o].add(s)
        recalled_nodes.add(s)
        recalled_nodes.add(o)

    visited = set()
    max_triples_in_component = 0
    for node in recalled_nodes:
        if node not in visited:
            component_nodes = set()
            q = deque([node])
            visited.add(node)
            while q:
                curr = q.popleft()
                component_nodes.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            
            triples_count = sum(1 for s, p, o in recalled_gt_triples if s in component_nodes and o in component_nodes)
            
            if triples_count > max_triples_in_component:
                max_triples_in_component = triples_count

    return max_triples_in_component / total_gt_triples

def calculate_total_reward(
    generated_graph: Graph,
    gt_graph: Graph,
    weights: Dict[str, float],
    entity_threshold: float = 0.7,
    relation_threshold: float = 0.6,
) -> Dict[str, float]:
    setup_data = _setup_graph_comparison(generated_graph, gt_graph, entity_threshold, relation_threshold)
    results = {"node_coverage": 0.0, "structural_correctness": 0.0, "chain_completeness": 0.0, "total_reward": 0.0}
    if not setup_data: return results

    r_node = _compute_node_coverage(setup_data)
    r_struct = _compute_structural_correctness(setup_data)
    r_chain = _compute_chain_completeness(setup_data)
    
    total = (weights.get("node", 0.0) * r_node + weights.get("structure", 0.0) * r_struct + weights.get("chain", 0.0) * r_chain)
    
    results.update({"node_coverage": r_node, "structural_correctness": r_struct, "chain_completeness": r_chain, "total_reward": total})
    return results

def calculate_node_coverage_reward(generated_graph: Graph, gt_graph: Graph) -> float:
    setup_data = _setup_graph_comparison(generated_graph, gt_graph, -1.0, -1.0)
    return _compute_node_coverage(setup_data) if setup_data else 0.0

def calculate_structural_correctness_reward(generated_graph: Graph, gt_graph: Graph, entity_threshold: float = 0.7, relation_threshold: float = 0.4) -> float:
    setup_data = _setup_graph_comparison(generated_graph, gt_graph, entity_threshold, relation_threshold)
    return _compute_structural_correctness(setup_data) if setup_data else 0.0

def calculate_chain_completeness_reward(generated_graph: Graph, gt_graph: Graph, entity_threshold: float = 0.7, relation_threshold: float = 0.4) -> float:
    setup_data = _setup_graph_comparison(generated_graph, gt_graph, entity_threshold, relation_threshold)
    return _compute_chain_completeness(setup_data) if setup_data else 0.0
