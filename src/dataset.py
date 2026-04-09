import torch
import numpy as np

class CounterfactualDataset:
    \"\"\"
    Maintains the fact and counterfactual data pairing mechanism.
    Used for causal interventions and fairness evaluations.
    \"\"\"
    def __init__(self, data_path, sensitive_groups=(\"group_s\", \"group_s_prime\")):
        self.data_path = data_path
        self.sensitive_groups = sensitive_groups
        self.data = self._load_data()

    def _load_data(self):
        \"\"\"
        Load the original prompts and their counterfactual variations.
        \"\"\"
        # Placeholder for data loading
        # Should return a list of dicts: [{\"fact\": \"...\", \"counterfactual\": \"...\", \"label\": \"...\"}, ...]
        return []

    def get_batches(self, batch_size=8):
        \"\"\"
        Yield pairs of factual and counterfactual batches.
        \"\"\"
        for i in range(0, len(self.data), batch_size):
            yield self.data[i:i+batch_size]

    def get_group_data(self, group_name):
        \"\"\"
        Retrieve all data corresponding to a specific demographic group (e.g., 'group_s' or 'group_s_prime').
        \"\"\"
        return [d for d in self.data if d.get(\"group\") == group_name]
