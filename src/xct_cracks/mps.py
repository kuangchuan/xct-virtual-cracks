"""Explicit Apple MPS checks for the portable neural examples."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch


def require_mps():
    """Verify allocation, forward, backward and an optimiser step on Metal."""
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS is not accessible in this process. Run the command in a "
            "native macOS Terminal with an MPS-enabled PyTorch environment. "
            "CPU fallback is disabled."
        )
    torch.set_num_threads(1)
    model = torch.nn.Conv2d(1, 2, 3, padding=1).to("mps")
    sample = torch.randn(2, 1, 32, 32, device="mps")
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimiser.zero_grad(set_to_none=True)
    loss = model(sample).square().mean()
    loss.backward()
    optimiser.step()
    torch.mps.synchronize()
    if sample.device.type != "mps" or not bool(torch.isfinite(loss)):
        raise RuntimeError("MPS allocation/forward/backward verification failed")
    result = {
        "backend": "mps",
        "torch": torch.__version__,
        "cpu_fallback": False,
        "allocation_forward_backward_optimiser_passed": True,
    }
    del optimiser, model, sample, loss
    torch.mps.empty_cache()
    return result
