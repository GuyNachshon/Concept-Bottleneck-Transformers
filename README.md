# Concept-Bottleneck Transformers (CBT)

## What it is (one line)

Add a tiny, **sparse “concept layer”** inside each block so the model must compress its hidden state into **m human-auditable channels** (e.g., 64) before continuing. Those channels become named, steerable *concepts*.

---

## How we’d build it

### 1) Where it lives in the block

Residual stream $h_\ell$ → **Concept Encoder** $E_\ell$ → sparse concept vector $c_\ell\in\mathbb{R}^m$ → **Concept Decoder** $D_\ell$ → reconstructed state $\tilde h_\ell$ → rest of block (attn/MLP).

$$
c_\ell=\mathrm{top}\text{-}k\big(\mathrm{entmax}(W^E_\ell h_\ell + b^E_\ell)\big),
\quad 
\tilde h_\ell = W^D_\ell c_\ell + b^D_\ell
$$

* **m (concepts per block):** 32–128 (start with 64).
* **top-k:** keep, say, 8 active concepts per token → readability.
* **entmax/sparsemax:** encourages exact zeros (clean sparsity).

**Bypass mix:** to avoid early damage,

$$
h'_\ell = (1-\alpha)\,h_\ell + \alpha\,\tilde h_\ell, \quad \alpha: 0\to 1 \text{ on a schedule.}
$$

### 2) Losses (trained jointly with LM loss)

* **Task loss**: standard next-token cross-entropy.
* **Reconstruction**: $\|\tilde h_\ell - h_\ell\|_2^2$ (keeps capacity).
* **Sparsity**: $\lambda_1\|c_\ell\|_1$ and top-k.
* **Diversity/orthogonality** on decoder columns:
  $\lambda_2\|{W^D_\ell}^\top W^D_\ell - I\|_F^2$ (avoid duplicated concepts).
* **Stability** (prevents concept ID shuffling across runs): Procrustes alignment to a moving anchor of $W^D_\ell$ + penalty on drift.
* **Distillation to base**: $\mathrm{KL}(\text{logits}_\text{CBT}\,\|\,\text{logits}_\text{base})$ (protects quality while $\alpha\uparrow$).

### 3) Practical training schedule

1. **Warm-start**: insert CBT with $\alpha=0$ (no effect).
2. **Reconstruct only** (few epochs): train $E,D$ to copy $h_\ell$.
3. **Blend-in**: ramp $\alpha$ → 0.5, then → 1.0 while keeping KL distillation on.
4. **Sparsify**: increase $\lambda_1$, enable top-k once quality plateaus.
5. **Freeze**: optionally freeze $D_\ell$ (solidifies concept meanings), train light adapters on $E_\ell$ to fine-tune routing.

### 4) Guardrails against “cheating”

* **Capacity cap**: limit m and k; no dense shortcut.
* **No side channels**: regularize LN/scale params so the model can’t hide info outside $c_\ell$.
* **Concept dropout**: randomly mask active concepts during training so each concept learns a distinct, necessary role.
* **Rate–distortion view**: treat $\|c_\ell\|_0$ as a “bit-budget”; tune it until perplexity cost is acceptably small.

---

## Why this shouldn’t hurt the model (and how we prove it)

* **Soft landing (α-schedule):** we only replace $h_\ell$ as the concept layer can reconstruct it; KL to the base keeps outputs on-manifold.
* **Redundant computation:** mid-block representations are overcomplete; compressing to 64–128 sparse channels per token is usually plenty.
* **Bypass insurance:** if a token needs nuance, $1-\alpha$ retains raw $h_\ell$ during the ramp.
* **Empirical gates:** we set hard success criteria (below) and stop if they’re not met.

**Success criteria (per model & eval set):**

* **Quality:** ≤2% perplexity/accuracy hit vs. baseline.
* **Sparsity:** median ≤8 active concepts/token/block.
* **Stability:** ≥0.8 alignment of concept decoders across seeds (after Procrustes).
* **Causality:** editing one concept produces **large, targeted** behavior deltas with minimal spillover.
* **Nameability:** ≥70% concepts receive consistent human/LLM labels.

---

## How concepts get names (and stay stable)

1. **Top-activations mining:** for each concept, collect the top N token contexts that light it up.
2. **Auto-label:** prompt a strong LLM with those snippets: “What unifies these?”
3. **Human pass (lightweight):** validate/merge synonymous labels.
4. **Stability check:** across seeds/checkpoints, align decoders (orthogonal Procrustes) and measure label consistency + Jaccard overlap of top contexts.

---

## Experiments you run (fast to heavy)

* **Ablation matrix:** off/on each concept → track which tasks move (precision of effects).
* **Granularity sweep:** m∈{32,64,128}; k∈{4,8,12} → plot sparsity/quality frontier.
* **Placement study:** concepts in early vs. mid vs. late blocks; likely sweet-spot = middle third.
* **Drift/transfer:** fine-tune on a new domain; see which concepts survive or split; test cross-model transfer of a concept by copying $W^D$ direction into a sibling model.

---

## Minimal code sketch (PyTorch-ish pseudocode)

```python
class ConceptLayer(nn.Module):
    def __init__(self, d_model=768, m=64, k=8):
        super().__init__()
        self.enc = nn.Linear(d_model, m, bias=True)
        self.dec = nn.Linear(m, d_model, bias=True)
        self.k = k

    def forward(self, h, alpha):
        # logits -> sparse probs
        p = entmax15(self.enc(h))           # (B, L, m)
        topk_vals, topk_idx = p.topk(self.k, dim=-1)
        c = torch.zeros_like(p).scatter(-1, topk_idx, topk_vals)  # hard top-k
        h_tilde = self.dec(c)
        return (1-alpha)*h + alpha*h_tilde, c
```

Train with task loss + recon + sparsity + orthogonality + KL to base; ramp `alpha`.

---

## What makes CBT *better than* post-hoc SAEs

* **Inside the loop:** SAEs analyze logged activations after the fact. CBT **changes the computation** so the model *must* use a small, stable set of channels.
* **Handles, not hypotheses:** CBT gives you levers you can flip during inference; SAEs give you guesses you must re-project back in.
* **Stability by design:** Procrustes alignment + decoder freezing makes concept IDs sticky across training.

---

## Risks & mitigations

* **Perplexity cliff:** slow α-ramp; keep a low-rank residual adapter alongside CBT for headroom.
* **Concept collapse/duplication:** decoder orthogonality + diversity losses + concept dropout.
* **Meaning drift across seeds:** stability loss + alignment to anchor decoders + label consensus filtering.
* **“Gaming” the bottleneck:** monitor MI($c_\ell$;$h_\ell$) and variance across positions; penalize suspiciously uniform or spiky usage.

---

## 90-day plan (lean team)

**Month 1 (GPT-2-small):**

* Insert CBT in middle 4 blocks, m=64, k=8, α:0→0.5, with KL distill.
* Hit ≤2% ppl hit; enable sparsity + orthogonality; basic ablations.

**Month 2:**

* Concept mining + auto-labels; full ablation grid; stability across 3 seeds.
* Expand to all blocks; test m/k sweeps.

**Month 3 (scale & utility):**

* Port to a 7B model via LoRA-style insertion.
* Run domain fine-tune; measure concept survival & transfer.
* Prepare a demo UI: per-token top-k concepts + toggle/edit.

---

## Why this matters (impact)

* **Governable internals:** You can *edit causes*, not just filter outputs.
* **Reproducible science:** Same concept IDs across runs → comparable results, sharable patches.
* **Safety foundation:** Once concepts exist, you can add **concept-gated policy**, **causal safety locks**, and **invariance training**—all *inside* the model, not bolted on.

If we pull this off, we move from “peering at neurons” to **operating the model via a small, named interface**. That’s a step-change for interpretability and alignment.
