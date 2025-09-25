import itertools
import sys
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rich import print
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Update this path to where your BGE model is stored
BGE_PATH = "..."

# --- Model Pool for GPU Load Balancing ---
class ModelPool:
    """A model pool for loading and cycling models on multiple GPUs for load balancing."""
    
    def __init__(self, model_path: str):
        self.models: List[SentenceTransformer] = []
        self.model_cycler = None
        self._load_models(model_path)

    def _load_models(self, model_path: str):
        """Loads models based on the number of available GPUs."""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"[bold green]Found {gpu_count} available GPUs. Loading a model for each...[/bold green]")
                for i in range(gpu_count):
                    device = f'cuda:{i}'
                    print(f"  -> Loading model onto device: {device}")
                    model = SentenceTransformer(model_path, device=device)
                    self.models.append(model)
            else:
                print("[yellow]No CUDA GPU detected, loading model onto CPU...[/yellow]")
                model = SentenceTransformer(model_path, device='cpu')
                self.models.append(model)

            if self.models:
                self.model_cycler = itertools.cycle(self.models)
                print("[bold green]Model pool loaded successfully and is ready.[/bold green]")

        except Exception as e:
            print(f"[bold red]CRITICAL: BGE model loading failed. Path: '{model_path}'. Error: {e}[/bold red]", file=sys.stderr)
            print("Please ensure the model path is correct and dependencies are installed.", file=sys.stderr)
            self.models = []
            self.model_cycler = None

    def get_model(self) -> SentenceTransformer | None:
        """Gets the next available model from the pool."""
        if not self.model_cycler:
            return None
        return next(self.model_cycler)

    def is_ready(self) -> bool:
        """Checks if the model pool has successfully loaded models."""
        return bool(self.models)

# --- FastAPI App Setup ---
# Use a dictionary to hold the state, as it's mutable and works well with FastAPI's lifecycle events
app_state = {}

app = FastAPI(
    title="Sentence Embedding API",
    description="An API to get sentence embeddings using a pool of BGE models.",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    """Load the model pool when the server starts."""
    print("--- [Server Startup] ---")
    model_pool = ModelPool(BGE_PATH)
    app_state["model_pool"] = model_pool
    if not model_pool.is_ready():
        print("[bold red]WARNING: Model pool is not ready. The /encode endpoint will fail.[/bold red]")
    print("--- [Startup Complete] ---")

# --- API Data Models ---
class EncodeRequest(BaseModel):
    texts: List[str] = Field(
        ..., 
        example=["This is a sample sentence.", "Here is another one."],
        description="A list of strings to be encoded."
    )

class EncodeResponse(BaseModel):
    embeddings: List[List[float]]
    model_device: str

# --- API Endpoint ---
@app.post("/encode", response_model=EncodeResponse)
async def encode_texts(request: EncodeRequest):
    """
    Accepts a list of texts and returns their normalized embeddings.
    """
    model_pool: ModelPool = app_state.get("model_pool")
    
    if not model_pool or not model_pool.is_ready():
        raise HTTPException(
            status_code=503, 
            detail="Model pool is not available or failed to load. Please check server logs."
        )

    model = model_pool.get_model()
    
    try:
        # Using convert_to_numpy is efficient for later conversion to a list
        embeddings = model.encode(
            request.texts, 
            normalize_embeddings=True,
            convert_to_numpy=True 
        )
        return EncodeResponse(
            embeddings=embeddings.tolist(),
            model_device=str(model.device)
        )
    except Exception as e:
        print(f"[bold red]Error during encoding: {e}[/bold red]", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"An error occurred during the encoding process: {e}")

if __name__ == "__main__":
    import uvicorn
    # To run, use the command: uvicorn model_server:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)