# From Jacob Dunefsky, 2025
# https://github.com/jacobdunefsky/one-shot-steering-repro/blob/master/steering_opt.py
# Modified by Dani Roytburg, 2025 to move tokens on/off GPUs.

import torch
from typing import List, Tuple, Callable, Optional, Union
import dataclasses
from contextlib import contextmanager
import mdmm
import gc
import numpy as np

# ---------------------------
# Utilities
# ---------------------------

def _nested_list_max(l):
    if isinstance(l, list):
        return max((_nested_list_max(l_) for l_ in l)) if len(l) > 0 else float('-inf')
    return l

def make_abl_mat(x: torch.Tensor):
    return (-torch.outer(x, x)/(x.norm().item()**2))

def _log1mexp(log_p: torch.Tensor):
    # stable log(1 - exp(log_p)); log_p <= 0
    cutoff = -0.6931471805599453  # -log(2)
    return torch.where(
        log_p <= cutoff,
        torch.log1p(-torch.exp(log_p)),
        torch.log(-torch.expm1(log_p)),
    )

# ---------------------------
# Hook contexts / makers
# ---------------------------

def get_base_model(model):
    return model.module if hasattr(model, 'module') else model


def get_text_config(model):
    base_model = get_base_model(model)
    config = base_model.config
    return getattr(config, "text_config", config)


def get_hidden_size(model):
    config = get_text_config(model)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError("Could not infer hidden_size from model config/text_config")
    return hidden_size


def get_layer_stack(model):
    base_model = get_base_model(model)
    candidate_paths = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
    ]
    for path in candidate_paths:
        obj = base_model
        try:
            for attr in path[:-1]:
                obj = getattr(obj, attr)
            return getattr(obj, path[-1])
        except AttributeError:
            continue
    raise AttributeError(
        "Could not locate transformer layers. Tried: "
        + ", ".join(".".join(path) for path in candidate_paths)
    )


# context manager for running a HuggingFace Llama model with hooks
@contextmanager
def hf_hooks_contextmanager(model, hook_infos : List[Tuple[int, Callable]]):
    layers = get_layer_stack(model)
    hooks = [layers[cur_layer].register_forward_pre_hook(hook_fn) for cur_layer, hook_fn in hook_infos]
    try:
        yield
    finally:
        for hook in hooks: hook.remove()

# classic steering hook (HF)
def make_steering_hook_hf(vector_, matrix=None, token=None):
    if token is None:
        token = slice(None)
    def hook_fn(module, args):
        x = args[0]
        vector = vector_.to(x) if isinstance(vector_, torch.Tensor) else vector_
        x_sliced = x[:, token].detach().clone()
        x[:, token] = x_sliced + vector
        if matrix is not None:
            affine_term = torch.zeros_like(x)
            affine_term[:, token] = torch.einsum('...n, mn -> ...m', x_sliced, matrix.to(x))
            x = x + affine_term
        return x
    return hook_fn

# classic steering hook (TransformerLens)
def make_steering_hook_tflens(vector, matrix=None, token=None):
    if token is None:
        token = slice(None)
    def hook_fn(x, hook):
        x_sliced = x[:, token]
        x[:, token] = x_sliced + vector
        if matrix is not None:
            affine_term = torch.zeros_like(x)
            affine_term[:, token] = torch.einsum('...n, mn -> ...m', x_sliced, matrix.to(x))
            x = x + affine_term
        return x
    return hook_fn

# hooks for getting activations
def make_activs_hook_hf(outlist):
    def hook_fn(module, args):
        x = args[0]
        outlist.append(x)
        return x
    return hook_fn

# ---------------------------
# NEW: Batched steering hook + helpers (HF path)
# ---------------------------

# Global storage for DataParallel mask handling
_STEERING_CONTEXT = {}

def make_steering_hook_hf_mask(vector_getter, matrix_getter=None, token_mask_getter=None, use_pre_add=True):
    """
    vector_getter(): returns (d_model,)
    matrix_getter(): returns (d_model, d_model) or None
    token_mask_getter(): returns (B, T, 1) mask; can contain -1 to flip direction per-example
    """
    def hook_fn(module, args):
        x = args[0] # (B, T, d_model)
        base = x if use_pre_add else x.detach()
        v = vector_getter().to(x) # (d_model,)
        
        # Handle mask for DataParallel
        if token_mask_getter is not None:
            full_mask = token_mask_getter()
            
            # Check if this is a DataParallel forward pass (smaller batch than full mask)
            if x.shape[0] < full_mask.shape[0]:
                # Get the batch indices being processed on this GPU
                batch_size = x.shape[0]
                device_idx = x.device.index if x.is_cuda else 0
                
                # Calculate the starting index for this GPU's batch
                start_idx = 0
                if device_idx > 0 and 'batch_offsets' in _STEERING_CONTEXT:
                    start_idx = _STEERING_CONTEXT['batch_offsets'][device_idx]
                else:
                    # Estimate based on device index and batch size
                    start_idx = device_idx * batch_size
                
                end_idx = min(start_idx + batch_size, full_mask.shape[0])
                mask = full_mask[start_idx:end_idx].to(x)
            else:
                mask = full_mask.to(x)
        else:
            mask = 1.0
            
        x = x + mask * v                               # broadcast add

        if matrix_getter is not None:
            M = matrix_getter()
            if M is not None:
                M = M.to(x)                            # (d_model, d_model)
                affine = torch.einsum('btd,dm->btm', base, M)
                x = x + affine * mask
        return x
    return hook_fn

def _tokenize_pairs(tokenizer, pairs, device, padding_side="left"):
    # pairs: List[(prompt, completion)]
    enc = tokenizer([p + c for (p,c) in pairs], return_tensors='pt',
                    padding=True, padding_side=padding_side)
    input_ids = enc.input_ids.to(device, non_blocking=True)       # (B,T)
    attn_mask = enc.attention_mask.to(device, non_blocking=True)  # (B,T)
    prompt_lens = [len(tokenizer(p).input_ids) for (p, _) in pairs]
    return input_ids, attn_mask, prompt_lens

def _build_completion_mask(attn_mask, prompt_lens):
    # returns (B,T-1) mask for next-token loss positions that belong to the completion
    B, T = attn_mask.shape
    comp_mask = torch.zeros_like(attn_mask, dtype=torch.float32)
    for i, pl in enumerate(prompt_lens):
        seq_len = int(attn_mask[i].sum().item())
        offset = T - seq_len  # left-padding offset
        start = max(pl - 1, 0)            # predict token at pl..end-1 using positions (pl-1..end-2)
        s = offset + start
        e = offset + max(seq_len - 1, 0)  # last predict position is seq_len-2
        if s < e:
            comp_mask[i, s:e] = 1.0
    return comp_mask[:, :-1]  # align with logits[:, :-1]

def _build_steer_mask(attn_mask, prompt_lens, token_specs, signs):
    """
    token_specs: per-example steering target:
        - 'prompt' -> positions [0:prompt_len)
        - None     -> all tokens [0:seq_len)
        - int      -> single position idx
        - slice    -> slice in [0:seq_len)
    signs: per-example +1.0 or -1.0 (for datapoint.is_negative)
    Returns (B,T,1) mask in padded coordinates (left padding honored).
    """
    B, T = attn_mask.shape
    mask = torch.zeros((B, T, 1), device=attn_mask.device, dtype=torch.float32)
    for i, (spec, sgn) in enumerate(zip(token_specs, signs)):
        seq_len = int(attn_mask[i].sum().item())
        offset = T - seq_len
        if spec == 'prompt':
            start, end = 0, prompt_lens[i]
        elif spec is None:
            start, end = 0, seq_len
        elif isinstance(spec, int):
            idx = spec if spec >= 0 else seq_len + spec
            start, end = max(0, idx), min(seq_len, idx + 1)
        elif isinstance(spec, slice):
            st = 0 if spec.start is None else (spec.start if spec.start >= 0 else seq_len + spec.start)
            ed = seq_len if spec.stop is None else (spec.stop if spec.stop >= 0 else seq_len + spec.stop)
            start, end = max(0, st), max(0, min(seq_len, ed))
        else:
            start, end = 0, seq_len
        s = offset + start
        e = offset + end
        if s < e:
            mask[i, s:e, 0] = sgn
    return mask

def _batched_losses_hf_with_hook(
    model, tokenizer, layer, *, pairs, token_specs, signs,
    temperature, do_one_minus, device, vector_ref, matrix_ref,
    normalize_token_length=False, padding_side="left"
):
    """
    Returns per-example losses (B,). Grad flows to vector/matrix via the hook.
    """
    input_ids, attn_mask, prompt_lens = _tokenize_pairs(tokenizer, pairs, device, padding_side)
    steer_mask = _build_steer_mask(attn_mask, prompt_lens, token_specs, signs)  # (B,T,1)

    # Store batch offsets for DataParallel handling
    global _STEERING_CONTEXT
    if hasattr(model, 'module') and torch.cuda.device_count() > 1:
        # Calculate how DataParallel will split the batch
        batch_size = input_ids.shape[0]
        num_gpus = torch.cuda.device_count()
        # DataParallel splits as evenly as possible
        base_size = batch_size // num_gpus
        extra = batch_size % num_gpus
        
        offsets = {}
        current_offset = 0
        for i in range(num_gpus):
            offsets[i] = current_offset
            current_offset += base_size + (1 if i < extra else 0)
        
        _STEERING_CONTEXT['batch_offsets'] = offsets

    hook = make_steering_hook_hf_mask(
        vector_getter=vector_ref,
        matrix_getter=matrix_ref,
        token_mask_getter=lambda: steer_mask,
    )
    hook_infos = [(layer, hook)] if not isinstance(layer, list) else [(l, hook) for l in layer]

    with hf_hooks_contextmanager(model, hook_infos):
        logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits  # (B,T,V)

    logp = torch.log_softmax(logits * temperature, dim=-1)              # (B,T,V)
    targets = input_ids[:, 1:]                                          # (B,T-1)
    logp_next = logp[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B,T-1)

    comp_mask = _build_completion_mask(attn_mask, prompt_lens).to(logp_next)   # (B,T-1)
    if do_one_minus:
        token_loss = -_log1mexp(logp_next.float()) * comp_mask
    else:
        token_loss = -logp_next.float() * comp_mask

    per_ex_loss = token_loss.sum(dim=1)  # (B,)

    if normalize_token_length:
        denom = comp_mask.sum(dim=1).clamp_min(1.0)
        per_ex_loss = per_ex_loss / denom

    return per_ex_loss  # (B,)

# ---------------------------
# Sampling / helpers (original)
# ---------------------------

def get_completion_logprob(model, prompt, completion, tokenizer=None, temperature=1, return_all_probs=False, do_one_minus=False, do_log=True, eps=0, use_transformer_lens=True, device='cuda:0', **kwargs):
    if use_transformer_lens:
        get_tokens = lambda prompt: torch.tensor(model.to_tokens(prompt).tolist()[0], device=device)
        get_logits = lambda prompt: model(prompt, **kwargs)[0].to(device)
    else:
        if tokenizer is None:
            raise Exception("Not using TransformerLens -- but tokenizer is None!")
        get_tokens = lambda prompt: torch.tensor(tokenizer(prompt).input_ids, device=device)
        def get_logits(prompt):
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device, non_blocking=True)
            logits = model(input_ids, **kwargs).logits[0].to(device)
            del input_ids
            return logits

    prompt_tokens = get_tokens(prompt)
    prompt_len = len(prompt_tokens)
    all_tokens = get_tokens(prompt + completion)
    completion_tokens = all_tokens[prompt_len:]
    completion_len = len(completion_tokens)

    logits = get_logits(prompt + completion).float()
    probs = torch.nn.functional.softmax(logits * temperature, dim=-1)
    if do_one_minus: probs = 1 - probs

    cur_loss = 0 if do_log else 1
    if return_all_probs:
        all_probs = []
    for completion_token_idx in range(0, completion_len):
        completion_token = completion_tokens[completion_token_idx]
        prompt_token_idx = prompt_len + completion_token_idx - 1
        target_prob = probs[prompt_token_idx, completion_token]
        if do_log: target_prob = torch.log(target_prob + 1e-12)
        if do_log:
            cur_loss += target_prob
        else:
            cur_loss *= target_prob
        if return_all_probs: all_probs.append(target_prob.item())
    del logits, probs, all_tokens, completion_tokens
    return cur_loss if not return_all_probs else (cur_loss, all_probs)

def get_completion_logprob_hf(model, prompt, completion, tokenizer, **kwargs):
    return get_completion_logprob(model, prompt, completion, tokenizer=tokenizer, use_transformer_lens=False, **kwargs)

@torch.no_grad()
def sample_most_likely_completions_hf(model, tokenizer, dst_prompt, src_prompt=None, k=5, iters=5, temperature=1, do_one_minus=False, gc_interval=3, use_total_probs=False, reverse=False, return_log_probs=False, return_token_probs=True, device='cuda:0', **kwargs):
    src_logits = model(tokenizer(src_prompt, return_tensors='pt').input_ids.to(device)).logits[:,-1].float() if src_prompt is not None else None
    dst_logits = model(tokenizer(dst_prompt, return_tensors='pt').input_ids.to(device)).logits[:,-1].float()
    src_probs = torch.nn.functional.softmax(src_logits*temperature, dim=-1) if src_prompt is not None else 0
    dst_probs = torch.nn.functional.softmax(dst_logits*temperature, dim=-1)
    prob_diffs = dst_probs - src_probs
    prob_diffs = prob_diffs * (-1 if reverse else 1)
    top_prob_diffs, token_idxs = torch.topk(prob_diffs, k=k)
    cur_completions = tokenizer.batch_decode(token_idxs.T)
    cur_completion_probs = top_prob_diffs.T.tolist()

    i = 0
    for i in range(iters):
        if src_prompt is not None:
            src_logits = model(tokenizer([src_prompt + x for x in cur_completions], return_tensors='pt').input_ids.to(device)).logits[:,-1].float()
            src_probs = torch.nn.functional.softmax(src_logits, dim=-1)
        else:
            src_probs = 0
        dst_logits = model(tokenizer([dst_prompt + x for x in cur_completions], return_tensors='pt').input_ids.to(device)).logits[:,-1].float()
        dst_probs = torch.nn.functional.softmax(dst_logits, dim=-1)
        prob_diffs = dst_probs - src_probs
        prob_diffs = prob_diffs * (-1 if reverse else 1)

        if not use_total_probs:
            v, idxs = torch.topk(prob_diffs.flatten(), k=k)
        else:
            prod_val = torch.tensor(cur_completion_probs).to(device).prod(dim=-1)
            total_prob_diffs = torch.einsum('nd, n -> nd', prob_diffs, prod_val)
            _, idxs = torch.topk(total_prob_diffs.flatten(), k=k)
            v = prob_diffs.flatten()[idxs]
            
        completion_idxs, token_idxs = torch.unravel_index(idxs, prob_diffs.shape)
        
        new_completions = []
        new_probs = []
        for completion_idx, token_idx, token_prob in zip(completion_idxs, token_idxs, v):
            new_completions.append(tokenizer.batch_decode([tokenizer(cur_completions[completion_idx], add_special_tokens=False).input_ids + [token_idx]])[0])
            new_probs.append(cur_completion_probs[completion_idx] + [token_prob.item()])
        cur_completions = new_completions
        cur_completion_probs = new_probs

    if gc_interval is not None and i+1 % gc_interval == 0:
        gc.collect()
    cur_completion_probs = np.array(cur_completion_probs)
    if return_log_probs:
        cur_completion_probs = np.log(cur_completion_probs)
        if not return_token_probs: cur_completion_probs = np.sum(cur_completion_probs, axis=-1)
    else:
        if not return_token_probs: cur_completion_probs = np.prod(cur_completion_probs, axis=-1)
    return cur_completions, cur_completion_probs

# ---------------------------
# Optimization core
# ---------------------------

def mdmm_grad_accumulate_backward(mdmm_module):
    for c in mdmm_module:
        c_return = c()
        c_return.value.backward()

@dataclasses.dataclass
class TrainingDatapoint:
    prompt: str
    src_completions: List[str] = dataclasses.field(default_factory=list)
    dst_completions: List[str] = dataclasses.field(default_factory=list)
    src_completions_target_losses: Optional[List[float]] = None
    dst_completions_target_losses: Optional[List[float]] = None
    token: Optional[Union[slice, int]] = None
    is_negative: bool = False

def optimize_completion(model, datapoints, layer,
    eps=1e-6, lr=0.01, max_iters=None, temperature=0.7,
    normalize_token_length=False, only_hook_prompt=False, use_transformer_lens=True, tokenizer=None,
    target_loss=None, return_loss=False, do_target_loss_avg=True, return_loss_history=False, return_vec_history=False,
    target_loss_target_iters=1, satisfice=False, do_one_minus=True,
    max_norm=None, starting_norm=1, starting_vec=None,
    vector_clamp=None, affine_rank=None, max_affine_norm=2, starting_affine_norm=1, do_output_constr=False,
    custom_output_constr_loss_func=None, custom_output_constr_pre_loss_func=None,
    output_constr_norm_initial_scale=1, output_constr_lr=None, debug=True,
    noise_scale=None, do_tangent_space_noise=True, do_noise_abl_relu=False, noise_iters=1,
    device='cuda:0',
    batch_size=None,
):
    if use_transformer_lens:
        if output_constr_lr is None: output_constr_lr = lr

    # --- Choose path (TL vs HF) ---
    if use_transformer_lens:
        d_model = model.cfg.d_model
        get_tokens = lambda prompt: model.to_tokens(prompt).tolist()[0]
        def get_hooked_logits(prompt, hook_infos):
            fwd_hooks = [(f'blocks.{cur_layer}.hook_resid_pre', hook_fn) for cur_layer, hook_fn in hook_infos]
            with model.hooks(fwd_hooks=fwd_hooks):
                return model(prompt)[0]
        make_steering_hook = make_steering_hook_tflens
    else:
        if tokenizer is None:
            raise Exception("Not using TransformerLens -- but tokenizer is None!")
        d_model = get_hidden_size(model)
        get_tokens = lambda prompt: tokenizer(prompt).input_ids
        def get_hooked_logits(prompt, hook_infos):
            cur_tokens = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            with hf_hooks_contextmanager(model, hook_infos):
                logits = model(cur_tokens, use_cache=False).logits[0].to(device)
            return logits
        make_steering_hook = make_steering_hook_hf

    # --- Initialize parameters ---
    if starting_vec is None:
        with torch.no_grad():
            vector = torch.randn(d_model, device=device)
            vector = starting_norm * vector / vector.norm()
    else:
        vector = starting_vec.detach().clone().to(device)
    vector.requires_grad_(True)

    if affine_rank is not None:
        with torch.no_grad():
            matrix_left = torch.randn(affine_rank, d_model, device=device)
            matrix_right = torch.randn(affine_rank, d_model, device=device)
            matrix_left = torch.einsum('rm, r -> rm', matrix_left, starting_affine_norm/matrix_left.norm(dim=1))
            matrix_right = torch.einsum('rm, r -> rm', matrix_right, starting_affine_norm/matrix_right.norm(dim=1))
        matrix_left.requires_grad_(True)
        matrix_right.requires_grad_(True)
    else:
        matrix_left = None
        matrix_right = None

    # --- Pre-tokenize completions (for legacy path + constraints bookkeeping) ---
    all_src_completions_tokens = []
    all_dst_completions_tokens = []
    all_prompt_lens = []

    all_completion_losses = []
    loss_history = []
    vec_history = []

    def check_if_target_loss_hit(all_completion_losses, target_loss):
        target_loss_hit = True
        for datapoint, datapoint_losses in zip(datapoints, all_completion_losses):
            for i, src_completion_loss in enumerate(datapoint_losses[0]):
                cur_target_loss = target_loss if datapoint.src_completions_target_losses is None else datapoint.src_completions_target_losses[i]
                if src_completion_loss > cur_target_loss:
                    target_loss_hit = False
                    break
            if not target_loss_hit: break
            for i, dst_completion_loss in enumerate(datapoint_losses[1]):
                cur_target_loss = target_loss if datapoint.dst_completions_target_losses is None else datapoint.dst_completions_target_losses[i]
                if dst_completion_loss > cur_target_loss:
                    target_loss_hit = False
                    break
            if not target_loss_hit: break
        return target_loss_hit

    for datapoint in datapoints:
        prompt = datapoint.prompt
        prompt_tokens = get_tokens(prompt)
        prompt_len = len(prompt_tokens)

        src_completions = datapoint.src_completions
        dst_completions = datapoint.dst_completions

        src_completions_tokens = []
        for src_completion in src_completions:
            src_completions_tokens.append(get_tokens(prompt + src_completion)[prompt_len:])
        dst_completions_tokens = []
        for dst_completion in dst_completions:
            dst_completions_tokens.append(get_tokens(prompt + dst_completion)[prompt_len:])

        all_completion_losses.append([
            [None for _ in range(len(src_completions))],
            [None for _ in range(len(dst_completions))],
        ])

        all_src_completions_tokens.append(src_completions_tokens)
        all_dst_completions_tokens.append(dst_completions_tokens)
        all_prompt_lens.append(prompt_len)

    params = [vector]
    if affine_rank is not None:
        params = params + [matrix_left, matrix_right]

    # --- Legacy single-sample loss (kept for constraints + fallback) ---
    def get_completion_loss(datapoint_idx, completion_idx, vector, matrix, is_src_completion=True, do_one_minus=True, vector_clamp=vector_clamp):
        datapoint = datapoints[datapoint_idx]
        prompt = datapoint.prompt
        prompt_len = all_prompt_lens[datapoint_idx]

        completion = datapoint.src_completions[completion_idx] if is_src_completion else datapoint.dst_completions[completion_idx]
        completion_tokens = all_src_completions_tokens[datapoint_idx][completion_idx] if is_src_completion else all_dst_completions_tokens[datapoint_idx][completion_idx]
        completion_len = len(completion_tokens)
        if datapoint.is_negative: vector = -vector

        if only_hook_prompt:
            hook_fn = make_steering_hook(vector_clamp*vector if vector_clamp is not None else vector,
                                         matrix=make_abl_mat(vector) if vector_clamp is not None else matrix,
                                         token=slice(0,prompt_len))
        else:
            hook_fn = make_steering_hook(vector_clamp*vector if vector_clamp is not None else vector,
                                         matrix=make_abl_mat(vector) if vector_clamp is not None else matrix,
                                         token=datapoint.token)
        hook_infos = [ (cur_layer, hook_fn) for cur_layer in (layer if isinstance(layer, list) else [layer]) ]

        cur_loss = 0.0
        logits = get_hooked_logits(prompt + completion, hook_infos).to(device)
        probs = torch.nn.functional.softmax(logits*temperature, dim=-1)

        for completion_token_idx in range(0, completion_len):
            completion_token = completion_tokens[completion_token_idx]
            prompt_token_idx = prompt_len+completion_token_idx-1
            target_prob = torch.log(1-probs[prompt_token_idx, completion_token] + eps) if is_src_completion and do_one_minus else torch.log(probs[prompt_token_idx, completion_token] + eps)
            if is_src_completion and not do_one_minus: target_prob = -target_prob
            cur_loss -= target_prob
        if normalize_token_length and completion_len > 0:
            cur_loss = cur_loss / completion_len
        del logits, probs
        return cur_loss

    def get_completion_loss_with_noise(datapoint_idx, completion_idx, vector, matrix, is_src_completion=True, do_one_minus=True, vector_clamp=vector_clamp):
        if noise_scale is None:
            return get_completion_loss(datapoint_idx, completion_idx, vector, matrix, is_src_completion=is_src_completion, do_one_minus=do_one_minus, vector_clamp=vector_clamp)

        noise = torch.randn(vector.shape, device=device) * noise_scale
        noise = noise.detach()

        if not do_tangent_space_noise:
            return get_completion_loss(datapoint_idx, completion_idx, vector + noise, matrix, is_src_completion=is_src_completion, do_one_minus=do_one_minus, vector_clamp=vector_clamp)

        # tangent-space noise
        zero_vec = torch.zeros_like(vector, device=device).requires_grad_(True)
        unsteered_loss = get_completion_loss(datapoint_idx, completion_idx, zero_vec, None, is_src_completion=is_src_completion, do_one_minus=do_one_minus, vector_clamp=vector_clamp)
        grad = torch.autograd.grad(outputs=unsteered_loss, inputs=zero_vec)[0]
        with torch.no_grad():
            abl_component = torch.dot(noise.to(grad), grad)/(grad.norm()**2)
            if do_noise_abl_relu:
                abl_component = -torch.nn.functional.relu(-abl_component)
            ablated_noise = noise.to(grad) + abl_component
        return get_completion_loss(datapoint_idx, completion_idx, vector + ablated_noise, matrix, is_src_completion=is_src_completion, do_one_minus=do_one_minus, vector_clamp=vector_clamp)

    optimizer = torch.optim.Adam(params, lr=lr)

    loss = None
    prev_loss = None
    iters = 0
    target_loss_cur_iters = 0
    prev_loss_cur_iters = 0

    # ---------------------------
    # NEW: Batched HF fast path setup
    # ---------------------------
    hf_batched_ok = (not use_transformer_lens) and (noise_scale is None) and (noise_iters == 1)

    if hf_batched_ok:
        # Precompute pair arrays + per-example steering specs
        src_pairs, dst_pairs = [], []
        src_specs, dst_specs = [], []   # token specs per pair
        src_signs, dst_signs = [], []   # +1/-1 per pair
        src_index_map, dst_index_map = [], []  # to write back per-example loss

        for dp_idx, dp in enumerate(datapoints):
            dp_spec = 'prompt' if only_hook_prompt else dp.token  # None / int / slice / 'prompt'
            sgn = -1.0 if dp.is_negative else 1.0

            for c_idx, c in enumerate(dp.src_completions):
                src_pairs.append((dp.prompt, c))
                src_specs.append(dp_spec)
                src_signs.append(sgn)
                src_index_map.append((dp_idx, c_idx))

            for c_idx, c in enumerate(dp.dst_completions):
                dst_pairs.append((dp.prompt, c))
                dst_specs.append(dp_spec)
                dst_signs.append(sgn)
                dst_index_map.append((dp_idx, c_idx))

    # ---------------------------
    # Training loop
    # ---------------------------
    while True:
        if max_iters is not None and iters > max_iters:
            break
        else:
            print(f"Iteration {iters}/{max_iters}")

        if target_loss is not None and loss is not None:
            if do_target_loss_avg:
                if loss <= (target_loss if not satisfice else target_loss + eps):
                    target_loss_cur_iters += 1
                else:
                    target_loss_cur_iters = 0
            else:
                target_loss_hit = check_if_target_loss_hit(all_completion_losses, target_loss if not satisfice else target_loss + eps)
                if target_loss_hit:
                    target_loss_cur_iters += 1
                else:
                    target_loss_cur_iters = 0
            if target_loss_cur_iters >= target_loss_target_iters:
                break

        optimizer.zero_grad()
        prev_loss = loss

        # ---------------------------
        # Fast batched HF path
        # ---------------------------
        if hf_batched_ok:
            def _vector_ref():
                return (vector_clamp * vector) if (vector_clamp is not None) else vector

            def _matrix_ref():
                if vector_clamp is not None:
                    return make_abl_mat(vector)
                if affine_rank is not None:
                    return matrix_left.T @ matrix_right
                return None

            total_loss_tensor = None
            
            # Determine chunk size (use batch_size if specified, otherwise process all at once)
            chunk_size = batch_size if batch_size is not None else max(len(src_pairs), len(dst_pairs))

            # SRC batch (with chunking and immediate gradient accumulation)
            if len(src_pairs) > 0:
                for chunk_start in range(0, len(src_pairs), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(src_pairs))
                    src_chunk_pairs = src_pairs[chunk_start:chunk_end]
                    src_chunk_specs = src_specs[chunk_start:chunk_end]
                    src_chunk_signs = src_signs[chunk_start:chunk_end]
                    src_chunk_index_map = src_index_map[chunk_start:chunk_end]
                    
                    src_losses = _batched_losses_hf_with_hook(
                        model, tokenizer, layer,
                        pairs=src_chunk_pairs, token_specs=src_chunk_specs, signs=src_chunk_signs,
                        temperature=temperature, do_one_minus=True, device=device,
                        vector_ref=_vector_ref, matrix_ref=_matrix_ref,
                        normalize_token_length=normalize_token_length
                    )
                    for (dp_i, c_i), val in zip(src_chunk_index_map, src_losses):
                        all_completion_losses[dp_i][0][c_i] = float(val.item())
                    src_term = ((src_losses - target_loss).clamp_min(0) ** 2).sum() if (satisfice and (target_loss is not None)) else src_losses.sum()
                    
                    # Skip this chunk if loss is NaN
                    if torch.isnan(src_term).any():
                        continue
                    
                    # Save loss value before backward
                    if total_loss_tensor is None:
                        total_loss_tensor = 0.0
                    total_loss_tensor += src_term.item()
                    
                    # Backward immediately to free memory
                    src_term.backward()
                    del src_losses, src_term

            # DST batch (with chunking and immediate gradient accumulation)
            if len(dst_pairs) > 0:
                for chunk_start in range(0, len(dst_pairs), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(dst_pairs))
                    dst_chunk_pairs = dst_pairs[chunk_start:chunk_end]
                    dst_chunk_specs = dst_specs[chunk_start:chunk_end]
                    dst_chunk_signs = dst_signs[chunk_start:chunk_end]
                    dst_chunk_index_map = dst_index_map[chunk_start:chunk_end]
                    
                    dst_losses = _batched_losses_hf_with_hook(
                        model, tokenizer, layer,
                        pairs=dst_chunk_pairs, token_specs=dst_chunk_specs, signs=dst_chunk_signs,
                        temperature=temperature, do_one_minus=False, device=device,
                        vector_ref=_vector_ref, matrix_ref=_matrix_ref,
                        normalize_token_length=normalize_token_length
                    )
                    for (dp_i, c_i), val in zip(dst_chunk_index_map, dst_losses):
                        all_completion_losses[dp_i][1][c_i] = float(val.item())
                    dst_term = ((dst_losses - target_loss).clamp_min(0) ** 2).sum() if (satisfice and (target_loss is not None)) else dst_losses.sum()
                    
                    # Skip this chunk if loss is NaN
                    if torch.isnan(dst_term).any():
                        continue
                    
                    # Save loss value before backward
                    if total_loss_tensor is None:
                        total_loss_tensor = 0.0
                    total_loss_tensor += dst_term.item()
                    
                    # Backward immediately to free memory
                    dst_term.backward()
                    del dst_losses, dst_term

            loss = float(total_loss_tensor) if total_loss_tensor is not None else 0.0

            # Debug: Print losses in datapoint order (matching unbatched format)
            if debug:
                for dp_idx in range(len(datapoints)):
                    dp = datapoints[dp_idx]
                    # Print SRC completions
                    for c_idx in range(len(dp.src_completions)):
                        if all_completion_losses[dp_idx][0][c_idx] is not None:
                            completion = dp.src_completions[c_idx]
                            token_id = tokenizer(completion, add_special_tokens=False).input_ids[0] if completion else 0
                            print(dp_idx, c_idx, 0, True, -all_completion_losses[dp_idx][0][c_idx], token_id)
                    # Print DST completions
                    for c_idx in range(len(dp.dst_completions)):
                        if all_completion_losses[dp_idx][1][c_idx] is not None:
                            completion = dp.dst_completions[c_idx]
                            token_id = tokenizer(completion, add_special_tokens=False).input_ids[0] if completion else 0
                            print(dp_idx, c_idx, 0, False, -all_completion_losses[dp_idx][1][c_idx], token_id)

            # early stop check against prev_loss
            if prev_loss is not None and abs(prev_loss - loss) < eps:
                prev_loss_cur_iters += 1
            if prev_loss_cur_iters >= target_loss_target_iters:
                if debug:
                    print("prev_loss reached")
                    print("prev_loss, loss:", prev_loss, loss)
                break

            # Apply accumulated gradients
            # Clip gradients to prevent NaN
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            # Check for NaN in gradients
            if any(torch.isnan(p.grad).any() if p.grad is not None else False for p in params):
                continue
            
            optimizer.step()

        # ---------------------------
        # Legacy (unbatched) path: TL or when noise is requested
        # ---------------------------
        else:
            loss_val = 0.0
            for datapoint_idx, datapoint in enumerate(datapoints):
                for src_completion_idx in range(len(datapoint.src_completions)):
                    for _ in range(noise_iters):
                        matrix = matrix_left.T @ matrix_right if affine_rank is not None else None
                        cur_loss = get_completion_loss_with_noise(datapoint_idx, src_completion_idx, vector, matrix, is_src_completion=True, do_one_minus=do_one_minus)
                        loss_val += cur_loss.item()
                        all_completion_losses[datapoint_idx][0][src_completion_idx] = cur_loss.item()
                        if satisfice:
                            cur_loss = (cur_loss - target_loss)**2
                        cur_loss.backward()

                for dst_completion_idx in range(len(datapoint.dst_completions)):
                    for _ in range(noise_iters):
                        matrix = matrix_left.T @ matrix_right if affine_rank is not None else None
                        cur_loss = get_completion_loss_with_noise(datapoint_idx, dst_completion_idx, vector, matrix, is_src_completion=False, do_one_minus=False)
                        loss_val += cur_loss.item()
                        all_completion_losses[datapoint_idx][1][dst_completion_idx] = cur_loss.item()
                        if satisfice:
                            cur_loss = (cur_loss - target_loss)**2
                        cur_loss.backward()

            loss = loss_val

            if prev_loss is not None and abs(prev_loss - loss) < eps:
                prev_loss_cur_iters += 1
            if prev_loss_cur_iters >= target_loss_target_iters:
                if debug:
                    print("prev_loss reached")
                    print("prev_loss, loss:", prev_loss, loss)
                break

            optimizer.step()

        # post-step norms (both paths)
        with torch.no_grad():
            if max_norm is not None and (cur_norm := torch.linalg.norm(vector)) > max_norm:
                vector[:] = max_norm * vector / torch.linalg.norm(vector)

            if affine_rank is not None and max_affine_norm is not None:
                cur_affine_norms_left = matrix_left.norm(dim=1)
                affine_coeffs_left = torch.where(cur_affine_norms_left > max_affine_norm, max_affine_norm/cur_affine_norms_left, 1) 
                cur_affine_norms_right = matrix_right.norm(dim=1)
                affine_coeffs_right = torch.where(cur_affine_norms_right > max_affine_norm, max_affine_norm/cur_affine_norms_right, 1) 
                matrix_left[:] = torch.einsum('rm, r -> rm', matrix_left, affine_coeffs_left)
                matrix_right[:] = torch.einsum('rm, r -> rm', matrix_right, affine_coeffs_right)

        if return_loss_history: loss_history.append(loss)
        if return_vec_history: vec_history.append([x.detach().cpu().float().numpy() for x in params])
        iters += 1

    if debug:
        print("Final loss:", loss)
        print("Number of iters:", iters)
        if prev_loss is not None: print("Difference between current loss and previous iter's loss:", abs(prev_loss - loss))

    retdict = {}
    retdict['iters'] = iters
    retdict['loss'] = loss if do_target_loss_avg else (all_completion_losses if not return_loss_history else loss_history)
    if return_vec_history: retdict['vec_history'] = vec_history
    retdict['norm'] = vector.norm().item()

    if not do_output_constr:
        retvals = (vector,)
        if affine_rank is not None:
            retvals = retvals + (matrix_left.T @ matrix_right,)
        if return_loss:
            retvals = retvals + (retdict,)
        return retvals

    # ---------------------------
    # Output-Constrained Optimization (legacy per-item constraints)
    # ---------------------------
    old_loss = loss
    if target_loss is None: target_loss = _nested_list_max(all_completion_losses)

    with torch.no_grad():
        starting_norm = vector.norm().item()
        if matrix_left is not None and matrix_right is not None:
            starting_norm += ((matrix_left.T @ matrix_right)**2).sum().sqrt().item()
        scale_factor = starting_norm/(eps+target_loss)

    output_constraints = []
    def make_output_constraint_func(datapoint_idx, completion_idx, vector, matrix_left=matrix_left, matrix_right=matrix_right, is_src_completion=True, do_one_minus=True, vector_clamp=vector_clamp):
        def constraint():
            matrix = None
            if matrix_left is not None and matrix_right is not None:
                matrix = matrix_left.T @ matrix_right
            return get_completion_loss_with_noise(datapoint_idx, completion_idx, vector, matrix, is_src_completion=is_src_completion, do_one_minus=do_one_minus, vector_clamp=vector_clamp)
        return constraint 

    for datapoint_idx, datapoint in enumerate(datapoints):
        for src_completion_idx, src_completion in enumerate(datapoint.src_completions):
            output_constraint_func = make_output_constraint_func(datapoint_idx, src_completion_idx, vector, matrix_left, matrix_right, is_src_completion=True, do_one_minus=do_one_minus)
            output_constraints.append(
                mdmm.MaxConstraint(output_constraint_func, scale=scale_factor, max=min(target_loss, all_completion_losses[datapoint_idx][0][src_completion_idx]+eps))
            )
        for dst_completion_idx, dst_completion in enumerate(datapoint.dst_completions):
            output_constraint_func = make_output_constraint_func(datapoint_idx, dst_completion_idx, vector, matrix_left, matrix_right, is_src_completion=False)
            output_constraints.append(
                mdmm.MaxConstraint(output_constraint_func, scale=scale_factor, max=min(target_loss, all_completion_losses[datapoint_idx][1][dst_completion_idx]+eps))
            )

    if custom_output_constr_loss_func is not None:
        def norm_constraint_func():
            loss = torch.linalg.norm(vector)
            if matrix_left is not None and matrix_right is not None:
                loss += ((matrix_left.T @ matrix_right)**2).sum().sqrt()
            return loss
        output_constraints.append(mdmm.MaxConstraint(norm_constraint_func, scale=1, max=output_constr_norm_initial_scale*norm_constraint_func().item()))

    custom_output_constr_dict = None
    if custom_output_constr_pre_loss_func is not None:
        custom_output_constr_dict = custom_output_constr_pre_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=only_hook_prompt)

    mdmm_module = mdmm.MDMM(output_constraints)
    optimizer = mdmm_module.make_optimizer(params, lr=output_constr_lr if use_transformer_lens else (output_constr_lr or lr))

    loss = None
    prev_loss = None
    iters = 0
    while prev_loss is None or loss <= prev_loss:
        prev_loss = loss
        prev_vec = vector.detach().clone()
        
        optimizer.zero_grad()

        if custom_output_constr_loss_func is not None and use_transformer_lens:
            if custom_output_constr_dict is not None:
                loss = custom_output_constr_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=only_hook_prompt, **custom_output_constr_dict)
            else:
                loss = custom_output_constr_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=only_hook_prompt)
        else:
            my_loss = torch.linalg.norm(vector)
            if matrix_left is not None and matrix_right is not None:
                my_loss += ((matrix_left.T @ matrix_right)**2).sum().sqrt()
            my_loss.backward()
            loss = my_loss.item()

        mdmm_grad_accumulate_backward(mdmm_module)

        optimizer.step()
        
        iters += 1

    retvals = (prev_vec,)
    retdict['norm'] = prev_vec.norm().item()
    retdict['output_constr_iters'] = iters
    if affine_rank is not None:
        retvals = retvals + (matrix_left.T @ matrix_right,)
    if return_loss:
        retvals = retvals + (retdict,)
    return retvals

# ---------------------------
# MELBO helpers (unchanged)
# ---------------------------

def make_melbo_loss_funcs(target_layer):
    make_steering_hook = make_steering_hook_tflens
    def melbo_pre_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt=None):
        hook_point = f'blocks.{target_layer}.hook_resid_pre'
        retdict = {'target_layer_activs': []}
        for datapoint in datapoints:
            prompt = datapoint.prompt
            prompt_len = len(model.to_tokens(prompt).tolist()[0])

            src_completion_activs = []
            for src_completion in datapoint.src_completions:
                with torch.no_grad():
                    _, cache = model.run_with_cache(prompt + src_completion, stop_at_layer=target_layer+1, names_filter=[hook_point])
                    activs = cache[hook_point][0, prompt_len-1:]
                src_completion_activs.append(activs)

            dst_completion_activs = []
            for dst_completion in datapoint.dst_completions:
                with torch.no_grad():
                    _, cache = model.run_with_cache(prompt + dst_completion, stop_at_layer=target_layer+1, names_filter=[hook_point])
                    activs = cache[hook_point][0, prompt_len-1:]
                dst_completion_activs.append(activs)

            datapoint_activs = [src_completion_activs, dst_completion_activs]
            retdict['target_layer_activs'].append(datapoint_activs)
        return retdict

    hook_dict = {}
    def capture_hook(x, hook):
        hook_dict['activs'] = x
        return x

    def melbo_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, target_layer_activs=None, only_hook_prompt=None, only_calculate_loss=False):
        loss = 0
        hook_point = f'blocks.{target_layer}.hook_resid_pre'
        for datapoint_idx, datapoint in enumerate(datapoints):
            prompt = datapoint.prompt
            prompt_len = len(model.to_tokens(prompt).tolist()[0])

            matrix = matrix_left.T @ matrix_right if matrix_left is not None and matrix_right is not None else None 
            if only_hook_prompt:
                if vector_clamp is None: hook_fn = make_steering_hook(vector, matrix=matrix, token=slice(0,prompt_len))
                else: hook_fn = make_steering_hook(vector_clamp*vector, matrix=make_abl_mat(vector), token=slice(0,prompt_len))
            else:
                if vector_clamp is None: hook_fn = make_steering_hook(vector, matrix=matrix, token=datapoint.token)
                else: hook_fn = make_steering_hook(vector_clamp*vector, matrix=make_abl_mat(vector), token=datapoint.token)
            if isinstance(layer, list):
                hook_infos = [ (f'blocks.{cur_layer}.hook_resid_pre', hook_fn) for cur_layer in layer]
            else:
                hook_infos = [ (f'blocks.{layer}.hook_resid_pre', hook_fn) ]

            for completion_idx, src_completion in enumerate(datapoint.src_completions):
                with model.hooks(fwd_hooks=hook_infos + [(hook_point, capture_hook)]):
                    model(prompt + src_completion, stop_at_layer=target_layer+1)
                activs = hook_dict['activs'][0, prompt_len-1:]
                original_activs = target_layer_activs[datapoint_idx][0][completion_idx]
                mean_distance = -((activs-original_activs).norm(dim=-1).mean())
                loss += mean_distance.item()
                if not only_calculate_loss:
                    mean_distance.backward()
                
            for completion_idx, dst_completion in enumerate(datapoint.dst_completions):
                with model.hooks(fwd_hooks=hook_infos + [(hook_point, capture_hook)]):
                    model(prompt + dst_completion, stop_at_layer=target_layer+1)
                activs = hook_dict['activs'][0, prompt_len-1:]
                original_activs = target_layer_activs[datapoint_idx][1][completion_idx]
                mean_distance = -((activs-original_activs).norm(dim=-1).mean())
                loss += mean_distance.item()
                if not only_calculate_loss:
                    mean_distance.backward()

        return loss
    return melbo_pre_loss_func, melbo_loss_func

# ---------------------------
# Minibatch helper (unchanged logic)
# ---------------------------

def optimize_minibatch_completion_hf(model, tokenizer, prompts, layer,
    src_completions=None, dst_completions=None,
    minibatch_size=5,
    eps=1e-6, lr=0.01, max_iters=None, temperature=0.7,
    target_loss=None, target_loss_target_iters=1, satisfice=False, target_loss_max_loss=True,
    starting_norm=1, max_norm=None,
    affine_rank=None, max_affine_norm=None,
    debug=True, return_loss=True,
    do_abl_hook=False, abl_hook_coeff=2,
    device='cuda:0',
):
    if src_completions is None: src_completions = []
    if dst_completions is None: dst_completions = []
    d_model = get_hidden_size(model)
    get_tokens = lambda prompt: tokenizer(prompt).input_ids
    def get_hooked_logits(prompt, hook_infos):
        cur_tokens = tokenizer(prompt, return_tensors='pt', padding=True, padding_side='left').input_ids.to(device)
        with hf_hooks_contextmanager(model, hook_infos):
            logits = model(cur_tokens, use_cache=False).logits.to(device)
        return logits 
    make_steering_hook = make_steering_hook_hf

    with torch.no_grad():
        vector = torch.randn(d_model, device=device)
        vector = starting_norm * vector / vector.norm()
    vector.requires_grad_(True)

    def get_completion_minibatch_loss(prompts, completion, vector, matrix=None, is_src_completion=True, vector_clamp=None):
        prompt_lens = [len(get_tokens(prompt)) for prompt in prompts]

        hook_fn = (make_steering_hook(abl_hook_coeff*vector, make_abl_mat(vector)) if do_abl_hook
                   else make_steering_hook(vector, matrix=matrix))
        hook_infos = [ (cur_layer, hook_fn) for cur_layer in (layer if isinstance(layer, list) else [layer]) ]

        cur_loss = 0
        all_tokens = tokenizer([prompt + completion for prompt in prompts], padding=True, padding_side='left', return_tensors='pt')
        all_tokens.input_ids = all_tokens.input_ids.to(device, non_blocking=True)
        with hf_hooks_contextmanager(model, hook_infos):
            logits = model(**all_tokens, use_cache=False).logits.to(device)
        probs = torch.nn.functional.softmax(logits*temperature, dim=-1)

        for prompt_idx in range(len(prompts)):
            prompt_len = prompt_lens[prompt_idx]
            cur_tokens = all_tokens.input_ids[prompt_idx]
            cur_prompt_probs = probs[prompt_idx]
            token_idx = prompt_len-1
            while token_idx < len(cur_tokens)-1 and (next_token := cur_tokens[token_idx+1]) != tokenizer.pad_token:
                target_prob = (1-cur_prompt_probs[token_idx, next_token]) if is_src_completion else cur_prompt_probs[token_idx, next_token]
                target_logprob = torch.log(target_prob + eps)
                cur_loss -= target_logprob
                token_idx += 1
        del logits, probs, all_tokens
        return cur_loss

    optimizer = torch.optim.Adam([vector], lr=lr)

    loss = None
    prev_loss = None
    iters = 0
    target_loss_cur_iters = 0
    prev_loss_cur_iters = 0

    minibatch_start_idx = 0
    minibatch_end_idx = None
    minibatch_rollover_end_idx = None

    while True:
        if max_iters is not None and iters > max_iters:
            break
        if target_loss is not None and loss is not None:
            if loss < target_loss:
                target_loss_cur_iters += 1
            else:
                target_loss_cur_iters = 0

            if target_loss_cur_iters >= target_loss_target_iters:
                break

        optimizer.zero_grad()
        prev_loss = loss
        loss = 0

        minibatch_start_idx = minibatch_rollover_end_idx if minibatch_rollover_end_idx is not None else minibatch_end_idx if minibatch_end_idx is not None else 0
        minibatch_end_idx = minibatch_start_idx + minibatch_size
        if minibatch_end_idx > len(prompts):
            minibatch_rollover_end_idx = minibatch_end_idx % len(prompts)
            minibatch_end_idx = len(prompts)
        else:
            minibatch_rollover_end_idx = None
        minibatch = prompts[minibatch_start_idx:minibatch_end_idx]
        if minibatch_rollover_end_idx is not None:
            minibatch += prompts[:minibatch_rollover_end_idx]

        for src_completion in src_completions:
            matrix = matrix_left.T @ matrix_right if affine_rank is not None else None
            cur_loss = get_completion_minibatch_loss(minibatch, src_completion, vector, matrix, is_src_completion=True)
            loss += cur_loss.item()
            if satisfice: cur_loss = (cur_loss - target_loss)**2
            cur_loss.backward()

        for dst_completion in dst_completions:
            matrix = matrix_left.T @ matrix_right if affine_rank is not None else None
            cur_loss = get_completion_minibatch_loss(minibatch, dst_completion, vector, matrix, is_src_completion=False)
            loss += cur_loss.item()
            if satisfice: cur_loss = (cur_loss - target_loss)**2
            cur_loss.backward()

        loss /= minibatch_size*(len(src_completions)+len(dst_completions))
        if prev_loss is not None and abs(prev_loss - loss) < eps:
            prev_loss_cur_iters += 1
        if prev_loss_cur_iters >= target_loss_target_iters:
            if debug:
                print("prev_loss reached")
                print("prev_loss, loss:", prev_loss, loss)
            break

        optimizer.step()

        with torch.no_grad():
            if max_norm is not None and (cur_norm := torch.linalg.norm(vector)) > max_norm:
                vector[:] = max_norm * vector / torch.linalg.norm(vector)

            if affine_rank is not None and max_affine_norm is not None:
                cur_affine_norms_left = matrix_left.norm(dim=1)
                affine_coeffs_left = torch.where(cur_affine_norms_left > max_affine_norm, max_affine_norm/cur_affine_norms_left, 1) 
                cur_affine_norms_right = matrix_right.norm(dim=1)
                affine_coeffs_right = torch.where(cur_affine_norms_right > max_affine_norm, max_affine_norm/cur_affine_norms_right, 1) 
                matrix_left[:] = torch.einsum('rm, r -> rm', matrix_left, affine_coeffs_left)
                matrix_right[:] = torch.einsum('rm, r -> rm', matrix_right, affine_coeffs_right)
        
        iters += 1

    if debug:
        print("Final loss:", loss)
        print("Number of iters:", iters)
        if prev_loss is not None: print("Difference between current loss and previous iter's loss:", abs(prev_loss - loss))

    retdict = {}
    retdict['iters'] = iters
    retdict['loss'] = loss
    retdict['norm'] = vector.norm().item()

    retvals = (vector,)
    if affine_rank is not None:
        retvals = retvals + (matrix_left.T @ matrix_right,)
    if return_loss:
        retvals = retvals + (retdict,)
    return retvals
