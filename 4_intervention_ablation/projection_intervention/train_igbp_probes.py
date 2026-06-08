import os
import argparse
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class Probe(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_iterations", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Llama-3 needs float16 to fit loosely or bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    
    print("Loading data...")
    with open(args.dataset_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    labels = []
    for item in data[:5000]:
        race = item.get("race", "")
        # Very permissive match. For dataset_paired or discrim, parse properly.
        if "Black" in race or "White" in race:
            label = 1.0 if "Black" in race else 0.0
            text = item.get("summary", item.get("context", item.get("text", "")))
            if text:
                samples.append(text)
                labels.append(label)

    print(f"Extracted {len(samples)} valid samples.")
    if len(samples) == 0:
        print("Error: No samples extracted! Check dataset format.")
        return

    X_list = []
    batch_size = 1
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size)):
            batch_texts = samples[i:i+batch_size]
            encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            out = model(**encoded, output_hidden_states=True)
            hidden = out.hidden_states[-1] 
            for j in range(hidden.shape[0]):
                length = encoded.attention_mask[j].sum().item()
                last_token_hidden = hidden[j, length-1, :]
                X_list.append(last_token_hidden.cpu())

    X = torch.stack(X_list).float().to(device)
    Y = torch.tensor(labels).float().to(device).unsqueeze(-1)

    probes = []
    for k in range(args.num_iterations):
        print(f"Iteration {k+1}/{args.num_iterations}")
        probe = Probe(X.shape[-1]).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        X_train = X.detach()
        for epoch in range(150):
            optimizer.zero_grad()
            logits = probe(X_train)
            loss = criterion(logits, Y)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            preds = (probe(X_train) > 0).float()
            acc = (preds == Y).float().mean().item()
        print(f"  Probe Acc: {acc:.4f} (Loss: {loss.item():.4f})")
        probes.append(probe.cpu())
        
        X_opt = X.detach().clone().requires_grad_(True)
        probe_gpu = probe.to(device)
        logits = probe_gpu(X_opt)
        grad_x = torch.autograd.grad(logits.sum(), X_opt)[0]
        with torch.no_grad():
             X = X_opt - (logits / (grad_x.norm(dim=-1, keepdim=True)**2 + 1e-8)) * grad_x
        print(f"  X shifted by norm: {(X - X_opt).norm().item():.4f}")

    torch.save(probes, os.path.join(args.output_dir, "igbp_probes.pt"))
    print("Saved to", os.path.join(args.output_dir, "igbp_probes.pt"))

if __name__ == "__main__":
    main()
