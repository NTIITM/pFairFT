import re

with open('analyze_race_sensitive_heads_moefreeze.py', 'r') as f:
    code = f.read()

helpers = """
moe_routing_cache = {}
current_batch_indices = []

def get_moe_save_hook(layer_idx):
    def hook(module, args, output):
        bsz, seq_len, _ = args[0].shape
        topk_idx, topk_weight, aux_loss = output
        idx = topk_idx.view(bsz, seq_len, -1).cpu()
        wt = topk_weight.view(bsz, seq_len, -1).cpu()
        global current_batch_indices
        for i, b_idx in enumerate(current_batch_indices):
            if b_idx not in moe_routing_cache:
                moe_routing_cache[b_idx] = {}
            moe_routing_cache[b_idx][layer_idx] = (idx[i].clone(), wt[i].clone())
        return output
    return hook

def get_moe_force_hook(layer_idx):
    def hook(module, args, output):
        bsz, seq_len, _ = args[0].shape
        
        forced_topk_idx = torch.zeros((bsz, seq_len, output[0].shape[-1]), dtype=output[0].dtype, device=args[0].device)
        forced_topk_wt = torch.zeros((bsz, seq_len, output[1].shape[-1]), dtype=output[1].dtype, device=args[0].device)
        
        global current_batch_indices
        for i, b_idx in enumerate(current_batch_indices):
            if b_idx in moe_routing_cache and layer_idx in moe_routing_cache[b_idx]:
                forced_topk_idx[i] = moe_routing_cache[b_idx][layer_idx][0].to(args[0].device)
                forced_topk_wt[i] = moe_routing_cache[b_idx][layer_idx][1].to(args[0].device)
            else:
                forced_topk_idx[i] = output[0].view(bsz, seq_len, -1)[i]
                forced_topk_wt[i] = output[1].view(bsz, seq_len, -1)[i]
                
        forced_topk_idx = forced_topk_idx.view(bsz * seq_len, -1)
        forced_topk_wt = forced_topk_wt.view(bsz * seq_len, -1)
        return (forced_topk_idx, forced_topk_wt, output[2])
    return hook
"""

# Inject helpers after imports
code = code.replace("from prompt import add_yes_no_instruction", "from prompt import add_yes_no_instruction\n\n" + helpers)

# 1. Inject moe_save_hook into Fact Inference
save_hook_target = """        for l in range(num_layers):
            if hasattr(model.model.layers[l].self_attn, "o_proj"):"""
save_hook_injection = """        moe_hooks_fact = []
        for l in range(num_layers):
            if hasattr(model.model.layers[l], "mlp") and hasattr(model.model.layers[l].mlp, "gate"):
                moe_hooks_fact.append(model.model.layers[l].mlp.gate.register_forward_hook(get_moe_save_hook(l)))
                
        for l in range(num_layers):
            if hasattr(model.model.layers[l].self_attn, "o_proj"):"""
code = code.replace(save_hook_target, save_hook_injection)

# Global current batch indices tracker in Fact Inference
fact_batch_target = """            batch_range = torch.arange(fact_inputs["input_ids"].shape[0], device=device)

            batch_activations_buffer.clear()"""
fact_batch_injection = """            batch_range = torch.arange(fact_inputs["input_ids"].shape[0], device=device)

            global current_batch_indices
            current_batch_indices = indices.numpy().tolist()

            batch_activations_buffer.clear()"""
code = code.replace(fact_batch_target, fact_batch_injection)

# Remove fact hooks
rm_hooks_target = """        # 移除 Fact hook
        for h in hooks_fact:
            h.remove()"""
rm_hooks_injection = """        # 移除 Fact hook
        for h in hooks_fact:
            h.remove()
        for h in moe_hooks_fact:
            h.remove()"""
code = code.replace(rm_hooks_target, rm_hooks_injection)


# 2. Inject moe_force_hook in Intervention Loop
intervention_start_target = """        heatmap_kl = np.zeros((num_layers, num_heads), dtype=np.float64)

        # DEBUG: 标记是否已输出首次干预的调试信息
        first_intervention_printed = False"""
intervention_start_injection = """        heatmap_kl = np.zeros((num_layers, num_heads), dtype=np.float64)

        print("Registering MoE Routing Freeze Hooks...")
        moe_force_hooks = []
        for l in range(num_layers):
            if hasattr(model.model.layers[l], "mlp") and hasattr(model.model.layers[l].mlp, "gate"):
                moe_force_hooks.append(model.model.layers[l].mlp.gate.register_forward_hook(get_moe_force_hook(l)))

        # DEBUG: 标记是否已输出首次干预的调试信息
        first_intervention_printed = False"""
code = code.replace(intervention_start_target, intervention_start_injection)

# Global current batch indices tracker in Intervention Loop
interv_batch_target = """                last_token_indices = get_last_token_indices_safe(
                    fact_inputs["input_ids"], attention_mask, tokenizer
                )"""
interv_batch_injection = """                last_token_indices = get_last_token_indices_safe(
                    fact_inputs["input_ids"], attention_mask, tokenizer
                )
                
                global current_batch_indices
                current_batch_indices = indices.numpy().tolist()"""
code = code.replace(interv_batch_target, interv_batch_injection)

# Cleanup moe hooks
cleanup_target = """            avg_kl = layer_head_kl_sum / total_samples
            heatmap_kl[l, :] = avg_kl.numpy()

        # ==========================================
        # Step 4: Visualization & Saving"""
cleanup_injection = """            avg_kl = layer_head_kl_sum / total_samples
            heatmap_kl[l, :] = avg_kl.numpy()
            
        for h in moe_force_hooks:
            h.remove()

        # ==========================================
        # Step 4: Visualization & Saving"""
code = code.replace(cleanup_target, cleanup_injection)

with open('analyze_race_sensitive_heads_moefreeze.py', 'w') as f:
    f.write(code)

print("Patching successful.")
