from openai import OpenAI
from rich import print
import json
import re
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.rich import tqdm


# --- Configuration ---
QWEN3_235B_BASE_URL = ""
DEEPSEEK_V3_BASE_URL = ""
DEEPSEEK_R1_BASE_URL = ""
GEMINI_PRO_BASE_URL = ""

# List of files to process
FILE_LIST = ["..."]
API_KEY = "sk-..."
GPT_KEY = "sk-..."

MAX_TRIES = 3
MAX_WORKERS = 48
TARGET_MODEL = "gpt-5-high"
# --- Client Initialization ---
try:
    client = OpenAI(
        api_key=GPT_KEY,
        base_url=GEMINI_PRO_BASE_URL,
    )
    MODEL_NAME = client.models.list().data[0].id
    MODEL_NAME = TARGET_MODEL
    print(f"[bold green]Successfully connected to model: {MODEL_NAME}[/bold green]")
except Exception as e:
    print(f"[bold red]Error connecting to API endpoint: {e}[/bold red]")
    print("Please ensure the API server is running and accessible.")
    exit()

PROMPT = \
"""
Please evaluate the provided "Thought Process" based on the "Problem" and "Standard Answer" using the following five criteria.

Example:

- Case Summary: A 55-year-old male with a long history of smoking and hypertension presents to the emergency department with a two-hour history of ``sudden-onset, retrosternal crushing pain''. The pain radiates to his left arm and is accompanied by diaphoresis. Physical examination is unremarkable. An electrocardiogram (ECG) shows ST-segment elevation in leads V2-V4. The patient denies that the pain is related to breathing (non-pleuritic) and has no fever or unilateral leg swelling.

- Most Likely Diagnosis: Acute Anterior Myocardial Infarction (AMI).

Five Criteria:

1. Logical Coherence
Core Definition: Assesses whether the chain of reasoning from the evidence (case information) to the conclusion (diagnosis) is complete, sound, and free of contradictions. It evaluates if the final conclusion follows logically and necessarily from the analytical process.

Score 2 (Excellent): The reasoning is logically seamless. The conclusion is a direct and necessary result of the evidence presented. The entire argument is solid and convincing from premise to conclusion, with no logical leaps or internal contradictions.

Example: "The patient presents with typical ischemic chest pain (crushing, radiating to the left arm, with diaphoresis). The key evidence is the ST-segment elevation in leads V2-V4, which directly localizes and confirms an acute anterior myocardial infarction. Therefore, the final diagnosis is AMI."

Score 1 (Adequate): The reasoning process generally supports the conclusion but contains minor logical flaws. This may include insufficient justification for secondary points, minor inferential gaps, or a conclusion that is too broad or narrow to be precisely supported by the evidence.

Example: "The patient has chest pain and ECG abnormalities, indicating a cardiac issue. The ECG changes are consistent with cardiac ischemia. Therefore, the diagnosis is Acute Coronary Syndrome (ACS)."

Score 0 (Inadequate): The reasoning process contains fundamental logical fallacies or severely contradicts the conclusion. The final answer appears disconnected from or contradictory to the analysis.

Example: "The ST-segment elevation on the ECG strongly suggests AMI. The absence of pleuritic pain also lowers the likelihood of a pulmonary embolism. Therefore, the final diagnosis is pulmonary embolism."

2. Factual Accuracy
Core Definition: Assesses the accuracy of all medical knowledge cited in the reasoning process, ensuring it aligns with current, evidence-based clinical guidelines, textbooks, and consensus.

Score 2 (Excellent): All cited medical facts are accurate and current. This includes disease definitions, pathophysiology, diagnostic criteria, interpretation of findings, and treatment principles, all of which are consistent with authoritative sources.

Example: "The ECG shows ST-segment elevation in leads V2-V4, a hallmark of an acute anterior wall myocardial infarction, which is typically associated with the occlusion of the Left Anterior Descending (LAD) coronary artery."

Score 1 (Adequate): The reasoning contains non-critical factual errors that do not alter the main diagnostic pathway. For instance, citing a slightly inaccurate statistic or a minor error in a non-essential value range.

Example: "The ECG shows ST-segment elevation in leads V2-V4, which is indicative of an acute inferior wall myocardial infarction." (Note: This is factually incorrect localization).

Score 0 (Inadequate): The reasoning contains one or more critical factual errors that fundamentally mislead the analysis or could lead to patient harm.

Example: "ST-segment elevation is a benign early repolarization pattern, common in healthy individuals, and therefore has no clinical significance in this case."

3. Evidence Faithfulness
Core Definition: Assesses whether the reasoning is strictly and exclusively based on the information provided in the case, avoiding any fabrication of data (i.e., "hallucination").

Score 2 (Excellent): Every step of the reasoning is explicitly traceable to specific information within the provided case. All arguments are directly cited from or based on the source text, with no extrapolation beyond the given evidence.

Example: "Based on the case description of 'retrosternal crushing pain,' 'radiating to the left arm,' and 'accompanied by diaphoresis,' an acute cardiac event is highly suspected."

Score 1 (Adequate): The reasoning is primarily based on case information but includes minor, reasonable clinical assumptions or slight misinterpretations of the evidence. It introduces small, clinically plausible details not explicitly stated.

Example: "The patient's chest pain, likely accompanied by shortness of breath, points towards an acute cardiac event." (Note: Shortness of breath was not mentioned in the case).

Score 0 (Inadequate): The reasoning contains clear "hallucinations" by fabricating key information not present in the case and using it as a central pillar for the argument.

Example: "Laboratory results show the patient's troponin level is elevated at 15 ng/mL, confirming myocardial necrosis." (Note: No lab results were provided).

4. Interpretability & Clarity
Core Definition: Assesses whether the reasoning is presented in a structured, professional, and concise manner that is easily understood by a clinical peer.

Score 2 (Excellent): The presentation is clear, well-structured, and uses professional, precise language. It follows a standard clinical logic flow (e.g., presentation → differentials → analysis → conclusion), allowing a peer to effortlessly follow the complete thought process.

Example: "1. Clinical Presentation: The patient's symptoms (crushing chest pain, radiation, diaphoresis) are highly suggestive of cardiac-origin pain. 2. Key Investigations: The ECG finding of ST-segment elevation is definitive evidence for AMI. 3. Conclusion: Integrating the clinical picture and ECG, the diagnosis is clearly AMI."

Score 1 (Adequate): The core idea is understandable, but the presentation is flawed by redundancy, disorganized structure, or ambiguous language. It requires extra effort from the reader to parse the logic.

Example: "The patient has chest pain, very painful, and the EKG is also not good, it has changes. So we think it's a heart problem, because the pain and the EKG both point to the heart. So it should be a heart attack."

Score 0 (Inadequate): The presentation is convoluted, lacks a logical structure, and is filled with meaningless jargon or inappropriate terminology, making the core reasoning difficult or impossible to understand.

Example: "Vectorial changes of myocardial repolarization confirm the electrophysiological basis of transmural ischemia. Therefore, despite the negative evidence of pleuritic pain, the differential of dissection persists. The etiology is thus attributed to a cardiac source; an MI is considered."

5. Comprehensiveness of Information Utilization
Core Definition: Assesses how thoroughly all key clinical information was utilized, especially how both positive findings and important pertinent negatives were integrated to form the final judgment.

Score 2 (Excellent): The reasoning comprehensively considers all diagnostically significant positive and negative findings. It not only identifies evidence supporting the conclusion but also explicitly explains how key negative findings help to rule out other relevant diagnoses.

Example: "The diagnosis of AMI is based not only on positive findings like crushing chest pain and ST elevation, but is also supported by pertinent negatives: the patient's denial of pleuritic pain and absence of leg swelling significantly lower the probability of other fatal causes like pulmonary embolism."

Score 1 (Adequate): The reasoning focuses on the most critical clinical information to support the conclusion but overlooks some secondary or diagnostically valuable clues (either positive or negative). The analysis is not fully comprehensive.

Example: "The patient's crushing chest pain and ST-segment elevation on the ECG are classic signs of an acute myocardial infarction." (Note: This ignores pertinent negatives).

Score 0 (Inadequate): The reasoning exhibits clear "cherry-picking" behavior. It selectively focuses on evidence that supports a preconceived conclusion while systematically ignoring critical information or pertinent negatives that contradict it.

Example (Assuming the ECG in this case was normal): "The patient's chest pain is classic crushing, radiating pain; therefore, the diagnosis is acute myocardial infarction."

Problem: %s


Standard Answer: %s


Thought Process: %s


Output Format:
```json
{
    "Logical Coherence": 0/1/2,
    "Factual Accuracy": 0/1/2,
    "Evidence Faithfulness": 0/1/2,
    "Interpretability & Clarity": 0/1/2,
    "Comprehensiveness of Information Utilization": 0/1/2
}
```
"""


def process_item(item, output_path, lock):
    """
    Processes a single data item using the OpenAI API, parses the result,
    and writes it to a file in a thread-safe manner upon success.
    """
    if not all(k in item for k in ["question", "gt_answer", "model_response"]):
        # print(f"[yellow]Skipping item with missing keys: {item.get('id', 'N/A')}[/yellow]")
        return

    question = item["question"]
    gt_answer = item["gt_answer"]
    model_response = item["model_response"]

    for attempt in range(MAX_TRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": PROMPT % (question, gt_answer, model_response)},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content

            match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
            if match:
                content_str = match.group(1).strip()
            else:
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end > start:
                    content_str = content[start:end + 1]
                else:
                    raise ValueError("No valid JSON object found in the response.")

            content_json = json.loads(content_str)
            
            required_keys = [
                "Logical Coherence", "Factual Accuracy", "Evidence Faithfulness",
                "Interpretability & Clarity", "Comprehensiveness of Information Utilization"
            ]
            if all(key in content_json for key in required_keys):
                item.update(content_json)
                # Use a lock to ensure thread-safe file writing
                with lock:
                    with open(output_path, "a", encoding="utf-8") as f_out:
                        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                return # Processed successfully, exit the function
            else:
                raise ValueError("Incomplete JSON response, missing necessary evaluation keys.")

        except Exception:
            if attempt == MAX_TRIES - 1:
                # print(f"[red]Item {item.get('id', 'N/A')} failed to process after {MAX_TRIES} attempts.[/red]")
                pass # Can log failures here
    return # All attempts failed

# --- Main Processing Loop ---
def main():
    """
    Main function to set up directories, load data, and process it using a thread pool.
    """
    output_dir = f"{TARGET_MODEL}-results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to the '{output_dir}' directory.")

    for file in FILE_LIST:
        if not os.path.exists(file):
            print(f"[bold yellow]Warning: File '{file}' does not exist. Skipped.[/bold yellow]")
            continue

        output_path = os.path.join(output_dir, os.path.basename(file))
        processed_ids = set()
        if os.path.exists(output_path):
            print(f"[cyan]Found existing output file '{output_path}'. Loading processed IDs...[/cyan]")
            with open(output_path, "r", encoding="utf-8") as f_out:
                for line in f_out:
                    try:
                        processed_item = json.loads(line)
                        if 'id' in processed_item:
                            processed_ids.add(processed_item['id'])
                    except json.JSONDecodeError:
                        print(f"[yellow]Skipping malformed line in output file: {line.strip()}[/yellow]")
        print(f"[cyan]Loaded {len(processed_ids)} processed IDs. These items will be skipped.[/cyan]")
        
        raw_data = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    raw_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[yellow]Skipping malformed line in {file}: {line.strip()}[/yellow]")
        
        # Filter out already processed items
        data_to_process = []
        for item in raw_data:
            item_id = item.get('id')
            if item_id is None:
                print(f"[yellow]Warning: Found an item missing the 'id' field. This item will always be attempted: {str(item)[:100]}...[/yellow]")
                data_to_process.append(item)
            elif item_id not in processed_ids:
                data_to_process.append(item)

        if not data_to_process:
            print(f"[bold green]All items in file '{file}' have been processed. Skipping to the next file.[/bold green]")
            continue
            
        print(f"File '{file}': {len(raw_data)} total items, {len(data_to_process)} items to process.")
        
        file_lock = threading.Lock()
        
        # Use ThreadPoolExecutor for multi-threaded processing
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a list of future tasks
            futures = [executor.submit(process_item, item, output_path, file_lock) for item in data_to_process]
            
            # Use tqdm to display a progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {file}"):
                # Loop to wait for tasks to complete; errors are handled within the thread
                try:
                    future.result() 
                except Exception as e:
                    # This exception is usually caught within the thread, but print it here just in case
                    print(f"\n[bold red]A thread unexpectedly threw an exception: {e}[/bold red]")
        
        print(f"[bold green]File '{file}' processing complete.[/bold green]")

if __name__ == "__main__":
    main()
