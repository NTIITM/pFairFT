import torch
import torch.nn as nn

def make_intervention_hook_igbp(probes_list):
    """
    Creates a PyTorch forward hook that intervenes on the hidden states
    by projecting them iteratively away from the probes' decision boundaries
    using the IGBP method.
    """
    def hook(module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        device = x.device
        dtype = x.dtype
        
        with torch.enable_grad():
            x_clean = x.detach().clone().float().requires_grad_(True)
            for probe in probes_list:
                probe = probe.to(device)
                probe.eval()
                # predict logits
                logits = probe(x_clean)
                # Compute gradient w.r.t x_clean
                grad_x = torch.autograd.grad(logits.sum(), x_clean)[0]
                # Orthogonal Projection eq: x - f(x)/||grad||^2 * grad
                # Add small epsilon to avoid div by zero
                x_clean = x_clean - (logits / (grad_x.norm(dim=-1, keepdim=True)**2 + 1e-8)) * grad_x
        
        x_clean = x_clean.detach().to(dtype)
        if isinstance(output, tuple):
            return (x_clean,) + output[1:]
        return x_clean
    return hook
