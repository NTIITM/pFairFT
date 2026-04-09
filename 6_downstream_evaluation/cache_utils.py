import os
import torch
import pickle
import numpy as np
from typing import Optional, Dict

class PreLogitsCache:
    """
    Cache for baseline (pre) logits to avoid redundant inference across different intervention tasks.
    Stored at: {cache_dir}/{model_name}/{sample_idx}_maxlen{max_length}.pkl
    """
    def __init__(self, cache_dir: str, model_name: str):
        self.cache_root = os.path.join(cache_dir, model_name)
        os.makedirs(self.cache_root, exist_ok=True)
        self.model_name = model_name

    def _get_path(self, sample_idx: int, max_length: int) -> str:
        return os.path.join(self.cache_root, f"sample_{sample_idx}_len{max_length}.pkl")

    def get(self, sample_idx: int, max_length: int, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        path = self._get_path(sample_idx, max_length)
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            # Integrity check: Ensure stored input_ids match current input_ids
            stored_ids = data.get("input_ids")
            if stored_ids is not None and not torch.equal(torch.as_tensor(stored_ids), input_ids.cpu()):
                return None
                
            return torch.as_tensor(data["logits"])
        except Exception:
            return None

    def save(self, sample_idx: int, max_length: int, input_ids: torch.Tensor, logits: torch.Tensor):
        path = self._get_path(sample_idx, max_length)
        data = {
            "input_ids": input_ids.cpu().numpy(),
            "logits": logits.cpu().numpy(),
            "model_name": self.model_name
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
