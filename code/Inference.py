# -*- coding: utf-8 -*-
"""
This script demonstrates how to use the transformers library to load the 
XXX/MedCEG model and generate text using the apply_chat_template 
and generate methods.
"""

import transformers
import torch

# Check if a CUDA-enabled GPU is available, otherwise use the CPU.
# The MedCEG model is relatively small and can run on a CPU, but it will be slower.
if not torch.cuda.is_available():
    print("Warning: No CUDA GPU detected. Running on CPU, which may be slower.")

# 1. Define the model ID
model_id = "ICLRAnonymous/MedCEG"
print(f"Loading model: {model_id}...")

# 2. Load the tokenizer and model
try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
                device_map="auto",
    )
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your internet connection and ensure the `accelerate` library is installed (`pip install accelerate`).")
    exit()

# 3. Create the input messages
question = "A 78-year-old Caucasian woman presented with an 8-day history of nausea, multiple bilious vomiting episodes, anorexia, right hypochondrial and epigastric discomfort, and fevers up to 38.5°C. On examination, she appeared ill, with pale skin and mucous membranes, signs of dehydration, tachypnea, tachyarrhythmia, and a temperature of 39°C. The abdomen was soft, mildly distended, and tender in the epigastrium and right upper quadrant.\n\nHer medical history included chronic hepatitis C infection, hypertension, atrial fibrillation, type 2 diabetes mellitus, prior stroke with left hemiparesis, hypothyroidism, hyperlipidemia, and recent NSAID use. Laboratory studies showed WBC 10.72×10^9/L (neutrophils 88.5% with toxic granulation), CRP 72.7 IU/L, glucose 139 IU/L, urea 73.4 IU/L, creatinine 1.38 IU/L, γ-GT 87 IU/L, total bilirubin 1.37 IU/L, direct bilirubin 1.10 IU/L, and LDH 517 IU/L.\n\nA chest radiograph suggested possible aspiration pneumonia. Abdominal radiography raised suspicion for air within the gallbladder. Ultrasonography showed a poorly visualized gallbladder with suspected intrahepatic biliary pneumobilia, no biliary dilation, and a dilated, fluid-filled stomach. Contrast-enhanced CT of the abdomen demonstrated a calculus within the second to third portion of the duodenum, dilatation of the first part of the duodenum, mild gastric dilatation, and air and contrast within the gallbladder.\nWhat is your diagnosis?"
suffix = "\nPut your final answer in \\boxed{}."
messages = [
    {"role": "user", "content": question + suffix},
]

# 4. Apply the chat template and tokenize
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# 5. Generate a response using the generate method
outputs = model.generate(
    input_ids,
    max_new_tokens=8196, 
    do_sample=False,
)

# 6. Decode the generated text
response_ids = outputs[0][input_ids.shape[-1]:]
decoded_response = tokenizer.decode(response_ids, skip_special_tokens=True)

# 7. Print the result
print("\n" + "="*20)
print("Model Response:")
print("="*20)
print(decoded_response.strip())

