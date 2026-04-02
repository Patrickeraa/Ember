import torch
import torch.nn as nn


def compute_grad_stats(model):
    total_norm = 0.0
    min_grad, max_grad = float('inf'), float('-inf')
    for p in model.parameters():
        if p.grad is None:
            continue
        grad = p.grad.data
        param_norm = grad.norm(2).item()
        total_norm += param_norm ** 2
        min_grad = min(min_grad, grad.min().item())
        max_grad = max(max_grad, grad.max().item())
    total_norm = total_norm ** 0.5
    return total_norm, min_grad, max_grad

def compute_weight_stats(model):
    total_norm = 0.0
    min_w, max_w = float('inf'), float('-inf')
    for p in model.parameters():
        w = p.data
        param_norm = w.norm(2).item()
        total_norm += param_norm ** 2
        min_w = min(min_w, w.min().item())
        max_w = max(max_w, w.max().item())
    total_norm = total_norm ** 0.5
    return total_norm, min_w, max_w


def bytes_to_gib(x_bytes):
    return float(x_bytes) / (1024 ** 3)  # GiB

def log_gpu_metrics(writer, epoch, device=None, reset_peak=False):
    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.synchronize(device)

    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)

    mem_alloc = torch.cuda.memory_allocated(device)
    mem_res = torch.cuda.memory_reserved(device)
    peak_alloc = torch.cuda.max_memory_allocated(device) 
    peak_res = torch.cuda.max_memory_reserved(device) 
    mem_free, mem_total = torch.cuda.mem_get_info(device)

    # gb
    alloc_gib = bytes_to_gib(mem_alloc)
    res_gib = bytes_to_gib(mem_res)
    peak_alloc_gib = bytes_to_gib(peak_alloc)
    peak_res_gib = bytes_to_gib(peak_res)
    free_gib = bytes_to_gib(mem_free)
    total_gib = bytes_to_gib(mem_total)

    pct_used = (alloc_gib / total_gib * 100.0) if total_gib > 0 else 0.0

    writer.add_scalar("GPU/MemoryAllocated_GiB", alloc_gib, epoch)
    writer.add_scalar("GPU/MemoryReserved_GiB", res_gib, epoch)
    writer.add_scalar("GPU/PeakMemoryAllocated_GiB", peak_alloc_gib, epoch)
    writer.add_scalar("GPU/PeakMemoryReserved_GiB", peak_res_gib, epoch)
    writer.add_scalar("GPU/MemoryFree_GiB", free_gib, epoch)
    writer.add_scalar("GPU/MemoryTotal_GiB", total_gib, epoch)
    writer.add_scalar("GPU/AllocatedPct", pct_used, epoch)

    try:
        summary = torch.cuda.memory_summary(device=device, abbreviated=True)
        writer.add_text(f"GPU{device}/MemorySummary", summary, epoch)
    except Exception as e:
        writer.add_text(f"GPU{device}/MemorySummary_Error", str(e), epoch)

    try:
        max_alloc = torch.cuda.max_memory_allocated(device)
        max_res = torch.cuda.max_memory_reserved(device)
        writer.add_scalar(f"GPU{device}/MaxMemoryAllocated_GiB", bytes_to_gib(max_alloc), epoch)
        writer.add_scalar(f"GPU{device}/MaxMemoryReserved_GiB", bytes_to_gib(max_res), epoch)
    except Exception as e:
        writer.add_text(f"GPU{device}/MaxMemory_Error", str(e), epoch)