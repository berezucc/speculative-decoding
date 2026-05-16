import time
import torch
from utils import get_device, set_seed
from draft import DraftModel
from verifier import VerifierModel
from acceptance import acceptance_prob, sample_residual
from speculative_cached import _get_probs, _batch_get_probs, _sample, _truncate_kv


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def profile_speculative(draft_obj, verifier_obj, prompt, n_tokens, k, device):
    """Run cached speculative with per-phase timing."""
    tokenizer = verifier_obj.tokenizer
    initial_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    L0 = initial_ids.shape[1]
    current_ids = initial_ids.clone()

    phases = ["prefill", "draft_proposal", "verifier_score", "accept_reject", "cache_mgmt", "setup_next"]
    phase_times = {p: 0.0 for p in phases}
    n_iters = 0

    # Prefill
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_v = verifier_obj.model(initial_ids, use_cache=True)
        verifier_kv = out_v.past_key_values
        last_v_logit = out_v.logits[0, -1, :].clone()
        out_d = draft_obj.model(initial_ids, use_cache=True)
        draft_kv = out_d.past_key_values
        last_d_logit = out_d.logits[0, -1, :].clone()
    sync(device)
    phase_times["prefill"] = time.perf_counter() - t0

    while current_ids.shape[1] - L0 < n_tokens:
        n_iters += 1
        T_old = current_ids.shape[1]

        # Phase 1: Draft proposal
        sync(device); t0 = time.perf_counter()
        draft_tokens = []
        draft_probs_list = []
        q1 = _get_probs(last_d_logit, 0.0)
        t1 = _sample(q1)
        draft_tokens.append(t1)
        draft_probs_list.append(q1)
        with torch.no_grad():
            for i in range(1, k):
                input_tok = draft_tokens[-1].view(1, 1)
                out = draft_obj.model(input_tok, past_key_values=draft_kv, use_cache=True)
                draft_kv = out.past_key_values
                logit = out.logits[0, -1, :]
                probs = _get_probs(logit, 0.0)
                draft_probs_list.append(probs)
                draft_tokens.append(_sample(probs))
        draft_tokens_tensor = torch.stack(draft_tokens)
        draft_probs_tensor = torch.stack(draft_probs_list)
        sync(device)
        phase_times["draft_proposal"] += time.perf_counter() - t0

        # Phase 2: Verifier score
        sync(device); t0 = time.perf_counter()
        with torch.no_grad():
            verifier_input = draft_tokens_tensor.view(1, -1)
            out_v = verifier_obj.model(verifier_input, past_key_values=verifier_kv, use_cache=True)
            verifier_kv = out_v.past_key_values
            new_v_logits = out_v.logits[0]
        verifier_logits_all = torch.cat([last_v_logit.unsqueeze(0), new_v_logits], dim=0)
        verifier_probs = _batch_get_probs(verifier_logits_all, 0.0)
        sync(device)
        phase_times["verifier_score"] += time.perf_counter() - t0

        # Phase 3: Accept/reject
        sync(device); t0 = time.perf_counter()
        new_tokens = []
        rejected = False
        for i in range(k):
            token_id = draft_tokens[i].item()
            p = verifier_probs[i]
            q = draft_probs_tensor[i]
            alpha = acceptance_prob(p, q, token_id)
            r = torch.rand(1).item()
            if r < alpha:
                new_tokens.append(token_id)
            else:
                correction = sample_residual(p, q)
                new_tokens.append(correction)
                rejected = True
                break
        n_accepted = len(new_tokens) - (1 if rejected else 0)
        last_token = new_tokens[-1]
        sync(device)
        phase_times["accept_reject"] += time.perf_counter() - t0

        # Phase 4: Cache management
        sync(device); t0 = time.perf_counter()
        if rejected:
            target_len = T_old + n_accepted
            verifier_kv = _truncate_kv(verifier_kv, target_len)
            draft_kv = _truncate_kv(draft_kv, target_len)
        else:
            bonus = _sample(verifier_probs[k])
            new_tokens.append(bonus.item())
            last_token = bonus.item()
            tK = draft_tokens_tensor[k - 1].view(1, 1)
            with torch.no_grad():
                out_d = draft_obj.model(tK, past_key_values=draft_kv, use_cache=True)
                draft_kv = out_d.past_key_values
        sync(device)
        phase_times["cache_mgmt"] += time.perf_counter() - t0

        # Phase 5: Setup next iter
        sync(device); t0 = time.perf_counter()
        last_tok_tensor = torch.tensor([[last_token]], device=device)
        with torch.no_grad():
            out_v = verifier_obj.model(last_tok_tensor, past_key_values=verifier_kv, use_cache=True)
            verifier_kv = out_v.past_key_values
            last_v_logit = out_v.logits[0, -1, :].clone()
            out_d = draft_obj.model(last_tok_tensor, past_key_values=draft_kv, use_cache=True)
            draft_kv = out_d.past_key_values
            last_d_logit = out_d.logits[0, -1, :].clone()
        sync(device)
        phase_times["setup_next"] += time.perf_counter() - t0

        new_tokens_tensor = torch.tensor([new_tokens], dtype=current_ids.dtype, device=device)
        current_ids = torch.cat([current_ids, new_tokens_tensor], dim=1)

    return phase_times, n_iters, current_ids.shape[1] - L0


def kernel_overhead_test(model, device, label):
    """Time forward pass with different input lengths to measure fixed vs marginal cost."""
    set_seed()
    results = {}
    for length in [1, 4, 16, 64]:
        input_ids = torch.randint(0, 50000, (1, 100), device=device)
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        kv = out.past_key_values

        n_trials = 20
        times = []
        for _ in range(n_trials):
            new_input = torch.randint(0, 50000, (1, length), device=device)
            sync(device); t0 = time.perf_counter()
            with torch.no_grad():
                model(new_input, past_key_values=kv, use_cache=True)
            sync(device)
            times.append(time.perf_counter() - t0)
        results[length] = sum(times) / len(times)

    print(f"\n{label} forward pass time vs input length")
    print(f"{'input len':>10} {'time (ms)':>12} {'ms per token':>14}")
    print("-" * 40)
    for length, t in results.items():
        per_tok = t * 1000 / length
        print(f"{length:>10} {t*1000:>11.3f} {per_tok:>13.3f}")
    return results


if __name__ == "__main__":
    device = get_device()
    print(f"device: {device}\n")

    print("loading models...")
    draft = DraftModel("distilgpt2", device)
    verifier = VerifierModel("gpt2-medium", device)
    print()

    prompts = [
        "The transformer architecture",
        "Photosynthesis is the process by which",
        "def fibonacci(n):",
        "Once upon a time in a small village",
        "Climate change is one of the most",
    ]
    k = 4
    n_tokens = 100

    print(f"profiling speculative K={k}, {n_tokens} tokens, {len(prompts)} prompts...\n")

    phases = ["prefill", "draft_proposal", "verifier_score", "accept_reject", "cache_mgmt", "setup_next"]
    agg = {p: 0.0 for p in phases}
    total_iters = 0
    total_generated = 0

    for prompt in prompts:
        phase_times, n_iters, n_gen = profile_speculative(draft, verifier, prompt, n_tokens, k, device)
        for p in phases:
            agg[p] += phase_times[p]
        total_iters += n_iters
        total_generated += n_gen

    total = sum(agg.values())

    print(f"{'Phase':<18} {'Total (s)':>10} {'Per iter (ms)':>15} {'% of total':>12}")
    print("-" * 60)
    for phase in phases:
        t = agg[phase]
        per_iter = (t / len(prompts) * 1000) if phase == "prefill" else (t / total_iters * 1000)
        pct = t / total * 100
        print(f"{phase:<18} {t:>10.3f} {per_iter:>15.3f} {pct:>11.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<18} {total:>10.3f}")
    print(f"\ntotal iters: {total_iters}, total tokens: {total_generated}, tok/s: {total_generated/total:.1f}")

    # Kernel launch overhead test
    print("\n" + "=" * 60)
    print("Kernel launch overhead test (fixed vs marginal cost per forward pass)")
    print("=" * 60)
    kernel_overhead_test(verifier.model, device, "Verifier (gpt2-medium)")
    kernel_overhead_test(draft.model, device, "Draft (distilgpt2)")
