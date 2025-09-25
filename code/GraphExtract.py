import json
import os

from openai import OpenAI
from rich import print

BASE_URL = os.environ.get("REWARD_MODEL_URL")
client = OpenAI(base_url=BASE_URL,api_key="Not_Needed")
model_name = "./models/gpt-oss-120b"


def get_knowledge_graph_prompt(reasoning_text: str) -> str:
    """
    Generates a single, comprehensive prompt for extracting and refining a medical knowledge graph.

    Args:
        reasoning_text: The medical reasoning process text to be analyzed.

    Returns:
        A formatted prompt string for the language model.
    """
    return f"""
You are an expert medical knowledge engineer. Your task is to analyze a medical reasoning process and convert it into a highly structured and refined knowledge graph. You must extract, standardize, and logically connect all key medical concepts to produce a final, clean JSON output.

### Processing Rules:

#### 1. **Entity Construction: Standardization & Purity (CRITICAL)**
- **Extract & Standardize**: Identify all medical entities (diseases, symptoms, tests, drugs, biomarkers). Immediately unify all synonymous, near-synonymous, or abbreviated entities (e.g., "the tumor," "the patient's mass") into the **single most medically precise and complete standard name** found in the source text (e.g., "invasive ductal carcinoma").
- **Enforce Purity**: Entities MUST be core nouns or noun phrases only. All modifiers (adjectives, states, locations) must be handled in one of two ways:
    - **Splitting (Preferred)**: Decompose a complex entity into multiple, atomic triplets.
        - **INCORRECT**: `["patient", "has symptom", "hard mass in upper outer quadrant of left breast"]`
        - **CORRECT**: `[["patient", "has symptom", "breast mass"], ["breast mass", "has texture", "hard"], ["breast mass", "has location", "upper outer quadrant of left breast"]]`
    - **Modifier Integration**: Move the modifier into the relationship phrase.
        - **INCORRECT**: `["HER2/neu positivity", "associated with", "poor prognosis"]`
        - **CORRECT**: `["HER2/neu receptor", "positivity is associated with", "poor prognosis"]`

#### 2. **Relationship Definition**
- Use clear, directional verb phrases (e.g., "causes," "is treated with," "is diagnosed by," "is positive for").
- Explicitly mark negative relationships (e.g., "rules out," "is not associated with").
- Retain hypothetical relationships from the reasoning process.
- Convert time information into medical temporal expressions (e.g., "acute," "chronic").
- Quantify probabilistic statements with risk levels (e.g., "high risk of").

#### 3. **Bridging Inference for Logical Gaps**
- Review the reasoning flow to ensure logical completeness.
- Add missing, but clinically essential, procedural steps that connect key events. A common failure is jumping from a symptom to a diagnosis without stating the intervening test.
- **Example**: If the text implies a diagnosis was made from a mass, you must add bridging triplets like `["patient", "undergoes", "biopsy"]` and `["biopsy", "was performed on", "mass"]`.


#### 4. **Output Format**
Your output must be a single JSON object with keys: "triplets".

### Correct Output Example
```json
{{
  "triplets": [
    ["patient", "has age", "65 years"],
    ["patient", "has gender", "female"],
    ["patient", "has symptom", "mass"],
    ["mass", "has texture", "hard"],
    ["mass", "is", "palpable"],
    ["mass", "located in", "left breast"],
    ["patient", "undergoes", "biopsy"],
    ["biopsy", "was performed on", "mass"],
    ["biopsy", "confirms diagnosis of", "invasive ductal carcinoma"],
    ["invasive ductal carcinoma", "is positive for", "HER2/neu receptor"],
    ["HER2/neu receptor", "positivity predicts", "a poor prognosis"],
    ["invasive ductal carcinoma", "is treated with", "Trastuzumab"]
  ]
}}
```

# Reasoning process
{reasoning_text}
"""


def process_one_item(item, max_tries=3):
    """Calls the API to process a single item, including retry logic."""
    for attempt in range(max_tries):
        try:
            reasoning_content = item
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": get_knowledge_graph_prompt(reasoning_content)},
                ],
                reasoning_effort="medium", 
            )
            content = response.choices[0].message.content

            json_str = content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[-1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
            
            js = json.loads(json_str.strip())

            if "triplets" in js:
                return js
            else:
                print(f"[bold yellow]Warning:[/bold yellow] 'triplets' key not found in response for item {item[:30]}.")
        except Exception as e:
            print(f"[bold red]Error processing item {item[:30]} (Attempt {attempt + 1}/{max_tries}):[/bold red] {e}. Retrying...")
    print(f"[bold red]Failed to process item {item[:30]} after {max_tries} retries.[/bold red]")
    return None


def clean_graph_data(graph):
    """Checks and cleans the graph data to ensure the triplet format is correct."""
    if "triplets" in graph:
        graph["triplets"] = [t for t in graph["triplets"] if isinstance(t, list) and len(t) == 3]
    return graph


def generate_reasoning_graph(item: str):
    graph = process_one_item(item)
    if not graph:
        return  
    return clean_graph_data(graph)
