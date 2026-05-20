# From Discrete Repair to Continuous Trust: ToM-Inspired Evidence Dynamics for Visual-Language Dialogue Coherence

## 1. Inspiration

Human conversation is not merely a sequence of independent utterances; it is a continuous process of maintaining, revising, and negotiating shared belief about the world. In grounded dialogue settings, interlocutors do not directly observe each other's internal intentions. Instead, they infer them incrementally through language, prior conversational context, and shared environmental evidence. This process lies at the core of Theory of Mind (ToM): the ability to estimate what another agent currently believes, whether that belief remains aligned with one's own, and when it must be revised.

Most multimodal dialogue systems, however, reduce grounding to a static text–image matching problem. A dialogue is encoded into a single vector, an image is encoded into another vector, and coherence is measured by a single similarity score. Such formulations implicitly assume that grounding is stable and globally consistent throughout the conversation. In natural dialogue, this assumption rarely holds.

Real conversational grounding is dynamic. During interaction, speakers may:

maintain reference to an existing object,

elaborate on a previously grounded region,

shift attention to a new aspect of the scene,

temporarily diverge in interpretation,

or initiate conversational repair to recover common ground.

These phenomena are inherently temporal and state-dependent. A listener must continuously decide:

Should I trust my current interpretation of the speaker's intent, or should I revise it?

We formulate this problem as a grounded conversational belief revision process.

Rather than treating dialogue grounding as a static alignment score, we explicitly model a latent conversational evidence state $e_t$ that evolves turn-by-turn over image regions. The state represents the model's current estimate of what portion of the shared visual environment is jointly grounded by the dialogue. Each incoming utterance produces new grounding evidence $\hat e_t$, and the model must determine whether to:

preserve prior grounding,

partially revise it,

or completely switch to a new interpretation.

This decision is regulated by a scalar gate $\rho_t \in (0,1)$.

Crucially, we interpret $\rho_t$ not as a conventional neural attention weight, but as a ToM-inspired communicative control variable:

low $\rho_t$ indicates confidence that the current conversational belief remains valid;

high $\rho_t$ indicates that the listener believes the speaker's intent may no longer be adequately captured by the current grounding state.

Under this view, conversational repair becomes a special case of a broader process:

adaptive revision of shared belief under uncertainty.

This perspective fundamentally differs from standard multimodal retrieval formulations. The central question is no longer:

“Does this dialogue match this image?”

but instead:

“When should an agent revise its internal model of shared referential intent?”

To investigate this question, we progressively study three increasingly expressive grounding controllers.

These tensions motivate a progressive ablation across three gate variants — labeled state-aware (A), unsupervised local change-point (B), and unsupervised multi-signal (C) — together with two coherence regularizers. Each step exposes a limitation of the previous, and the final method recovers retrieval performance while preserving strong intrinsic ToM signals. The negative findings along the way are themselves informative: they demarcate the design space and rule out simpler alternatives.

## 2. Method

### 2.1 Backbone

We use a **frozen CLIP** backbone (`openai/clip-vit-base-patch32`) for both modalities. For each turn $t$ in a dialogue $\{u_t\}_{t=1}^{T}$ paired with a reference image $I$:

- Text representation: $h_t = \mathrm{CLIP_{text}}(u_t) \in \mathbb{R}^{512}$
- Image representations: a global token $v^{\mathrm{cls}} \in \mathbb{R}^{768}$ and patch tokens $V = \{v_i\}_{i=1}^{49} \in \mathbb{R}^{49 \times 768}$

A single trainable projection $W_h \in \mathbb{R}^{768 \times 512}$ maps text into the visual space. Crucially, **$W_h$ is shared** across the grounding attention (Eq. 2), the alignment head (Eq. 4), and the cross-dialogue contrastive loss (Eq. 7) — avoiding the multi-projection fragmentation common in earlier formulations of similar architectures.

$$ q_t = W_h h_t \in \mathbb{R}^{768} \tag{1} $$

### 2.2 Evidence Dynamics

The model maintains a per-turn evidence distribution $e_t \in \Delta^{49}$ over patches, capturing where the dialogue is "looking." Raw evidence is computed by softmax attention:

$$ \hat{e}_t = \mathrm{softmax}\big(q_t^\top V\big) \in \Delta^{49} \tag{2} $$

The dynamic update interpolates the previous evidence with the current observation, gated by $\rho_t$:

$$ e_t = (1 - \rho_t)\, e_{t-1} + \rho_t\, \hat{e}_t \tag{3} $$

When $\rho_t \to 0$, evidence persists ($e_t \approx e_{t-1}$); when $\rho_t \to 1$, evidence switches ($e_t \approx \hat{e}_t$). The initial state $e_0$ is uniform $\frac{1}{49}\mathbf{1}$.

### 2.3 Alignment Head and Coherence Trajectory

An alignment head with weight $W_a \in \mathbb{R}^{1 \times 2304}$ scores how well the current grounding matches the image:

$$ c_t = e_t^\top V \in \mathbb{R}^{768} \quad\text{(evidence-weighted summary)}$$
$$ \tilde{a}_t = \sigma\!\left(W_a [q_t \,;\, c_t \,;\, v^{\mathrm{cls}}]\right) \in (0,1) \tag{4} $$

The coherence trajectory tracks alignment over time, gated by $\rho_t$:

$$ a_t = (1 - \lambda_t)\, a_{t-1} + \lambda_t\, \tilde{a}_t,\quad \lambda_t = 1 - \rho_t \tag{5} $$

This update is intentionally asymmetric in semantics: during repair (high $\rho$), $\lambda$ is small, so $a_t$ inherits $a_{t-1}$ (conservative); during stable turns, $\lambda$ is large, so $a_t$ tracks $\tilde{a}_t$ responsively.

### 2.4 Three Gate Variants

We instantiate $\rho_t$ in three progressively more sophisticated ways, all sharing the backbone, evidence dynamics, and alignment head described above.

**Variant A — Supervised State-Aware Gate.** A small head conditioned on the current utterance and the previous dynamics state:

$$ \rho_t^{(A)} = \sigma\!\left(W_p\,[h_t \,;\, e_{t-1} \,;\, a_{t-1}]\right) $$

Trained with weighted MSE on labeled turns. We use a soft target schedule $\{1.0, 0.4, 0.0\}$ for $\{\text{repair}, \text{clarification}, \text{stable}\}$ derived from human annotations on a 200-dialogue subset, augmented with TF-IDF + Logistic Regression pseudo-labels on additional dialogues.

**Variant B — Differentiable Change-Point Gate (DCP).** A label-free formulation that derives $\rho$ from the local evidence shift between the prior $e_{t-1}$ and the new raw evidence $\hat{e}_t$:

$$ d_t = \mathrm{KL}(e_{t-1} \,\|\, \hat{e}_t) \tag{6a} $$
$$ \rho_t^{(B)} = \sigma\!\big(\,a\,(d_t - \mu)\,\big) \tag{6b} $$

where $a > 0$ is a learned sharpness scalar (parameterized as $\exp(\log a)$) and $\mu$ is an EMA running mean of $d_t$ across batches. To prevent degenerate solutions, a sparsity prior is added (see §2.5).

**Variant C — Multi-Signal Gate (Ours).** Variant B's reliance on a single local statistic ($d_t$) was empirically insufficient (see §3.5). Variant C extends $\rho$ to a five-signal gate:

$$ \rho_t^{(C)} = \sigma\!\Big(W_p \big[\,d_t \,;\, a_{t-1} \,;\, \Delta\hat{a}_t \,;\, H(e_{t-1}) \,;\, H(\hat{e}_t)\,\big]\Big) $$

The five scalar features capture complementary aspects of "should I trust this new utterance":

| Feature | Meaning |
|---|---|
| $d_t = \mathrm{KL}(e_{t-1} \| \hat{e}_t)$ | local evidence divergence (DCP signal) |
| $a_{t-1}$ | prior coherence (was grounding good?) |
| $\Delta\hat{a}_t = \hat{a}_t - a_{t-1}$ | forward-looking jump in alignment |
| $H(e_{t-1})$ | uncertainty about prior grounding |
| $H(\hat{e}_t)$ | uncertainty about new evidence |

**Avoiding circular dependency.** A naive use of $\Delta a_t = a_t - a_{t-1}$ would create a cycle ($a_t$ depends on $\rho_t$, which we are computing). We instead use $\hat{a}_t = \sigma(W_a [q_t; \hat{c}_t; v^{\mathrm{cls}}])$ with $\hat{c}_t = \hat{e}_t^\top V$ — the *hypothetical* alignment if the new evidence were fully trusted. This signal is computable before $\rho_t$ and depends only on already-known quantities.

The variant C gate has just 6 trainable parameters ($\mathrm{Linear}(5,1)$ + bias).

### 2.5 Training Objectives

**Cross-dialogue contrastive ($L_{\text{corr}}$).** For a batch of dialogues, we pool all per-turn queries $\{q_t\}$ and pair each with its dialogue's $v^{\mathrm{cls}}$, with cross-dialogue $v^{\mathrm{cls}}$ as negatives:

$$ L_{\text{corr}} = -\sum_{t \in \text{batch}} \log \frac{\exp\!\big(\mathrm{sim}(q_t, v^{\mathrm{cls}}_{d(t)})/\tau\big)}{\sum_{b=1}^{B}\exp\!\big(\mathrm{sim}(q_t, v^{\mathrm{cls}}_b)/\tau\big)} \tag{7} $$

with cosine similarity and temperature $\tau = 0.07$. Note that multiple turns from the same dialogue share their positive $v^{\mathrm{cls}}$.

**Stability and switching ($L_{\text{stab}}$, $L_{\text{switch}}$).** Two ToM-driven KL regularizers on the evidence trajectory:

$$ L_{\text{stab}} = \frac{1}{T-1} \sum_{t \geq 2} (1 - \rho_t)_{\text{detached}} \cdot \mathrm{KL}(e_{t-1} \,\|\, e_t) \tag{8} $$

$$ L_{\text{switch}} = \frac{1}{T-1} \sum_{t \geq 2} (\rho_t)_{\text{detached}} \cdot \max\!\big(0,\, \delta - \mathrm{KL}(e_{t-1} \,\|\, e_t)\big) \tag{9} $$

with margin $\delta = 0.1$. **Crucially, $\rho$ is detached** in the weight terms. Without this detach, $L_{\text{stab}}$ creates a degenerate gradient pathway (Wa_text would be pushed to bend $\hat{e}_t$ to minimize the loss by making $\rho$ large rather than by stabilizing evidence). The detach treats $\rho$ as a regulator/mask; gradients still flow through $e_t$ itself, training $W_h$ to produce evidence dynamics that satisfy the chosen regime.

**Sparsity prior ($L_{\text{rate}}$, for B and C).** Prevents the unsupervised gates from collapsing:

$$ L_{\text{rate}} = \Big(\,\mathrm{mean}_t(\rho_t) - r\,\Big)^2 \tag{10} $$

with target rate $r = 0.15$.

**Coherence floor ($L_{\text{coh\_floor}}$, for our final method).** Asymmetric penalty discouraging trajectory collapse:

$$ L_{\text{coh\_floor}} = \frac{1}{T} \sum_t \max(0,\, \tau_a - a_t)^2 \tag{11} $$

with $\tau_a = 0.20$. This term prevents the alignment head from outputting near-zero $\tilde{a}_t$ on ambiguous turns and indirectly encourages $W_h$ to learn sharper text projections.

**Coherence drop penalty ($L_{\text{coh\_drop}}$, ablated).** Asymmetric penalty on rapid coherence drops:

$$ L_{\text{coh\_drop}} = \frac{1}{T-1} \sum_t \max(0,\, a_{t-1} - a_t - \epsilon)^2 \tag{12} $$

with $\epsilon = 0.05$. *Reported as a negative finding (§3.5)*: this term structurally fights legitimate ToM repair events, in which $a_t$ is expected to dip transiently.

**Repair MSE ($L_{\text{repair}}$, Variant A only).** Weighted soft-target MSE on labeled turns; weights are 1.0 for human labels and 0.5 for pseudo-labels.

**Total objective (multi_C+floor, our final method):**

$$ \mathcal{L} = L_{\text{corr}} + \alpha L_{\text{stab}} + \beta L_{\text{switch}} + \gamma_{\text{rate}} L_{\text{rate}} + \gamma_{\text{floor}} L_{\text{coh\_floor}} \tag{13} $$

with $\alpha = \beta = 0.1$, $\gamma_{\text{rate}} = 10$, $\gamma_{\text{floor}} = 0.1$.

### 2.6 Implementation Details

- **End-to-end differentiability.** Evidence dynamics are computed inline within each dialogue's forward pass, with no inter-turn `detach()`. The full chain $W_h \to \hat{e}_t \to e_t \to c_t \to \tilde{a}_t \to a_t$ is in the gradient graph.
- **Trainable components.** $W_h$ (text projection, 393K params), $W_a$ (alignment head, 2305 params), and the gate parameters (5–6 for C, 562 for A). CLIP backbone is frozen throughout.
- **Optimization.** AdamW, learning rate $5 \times 10^{-4}$, batch size 16, 8 epochs.

## 3. Evaluation & Result

### 3.1 Experimental Setup

**Dataset.** PhotoChat (Zang et al., 2021), a multimodal dialogue corpus where two crowd-workers chat about a shared image. The training data is provided as 21 JSON shards (`train_00.json` through `train_20.json`). We construct a **shuffled 19-train / 2-test split** at the shard level (seed 42), held out across all model variants and seeds, yielding approximately 9,000 training dialogues and 1,000 test dialogues.

**Training.** All variants are trained for 8 epochs with batch size 16 and learning rate $5 \times 10^{-4}$, using AdamW. Loss weights are fixed as in §2.5. Each variant is run with **3 random seeds** $\{0, 1, 2\}$ for model initialization; results are reported as mean ± standard deviation.

**Inference.** For retrieval, the dialogue-level query is the per-turn mean of $\{q_t\}_{t=1}^{T}$. The gallery is the set of unique $v^{\mathrm{cls}}$ across the test dialogues.

### 3.2 Metrics

We use three complementary metric families designed to evaluate both the *external* utility of grounding and the *intrinsic* behavior of the gate.

**(M1) Image retrieval.** R@1, R@5, R@10, Mean Reciprocal Rank (MRR), and median rank. The dataset provides an implicit ground-truth dialogue–image alignment; retrieval probes whether the trained $W_h$ encodes meaningful text-to-image grounding.

**(M2) Evidence stability and switching.** Per-turn KL$(e_{t-1} \| e_t)$ values are bucketed by $\rho_t$ relative to threshold $0.5$:

- *Stability index*: mean KL on stable turns ($\rho < 0.5$). Lower is better.
- *Switching index*: mean KL on repair turns ($\rho \geq 0.5$). Higher is better.
- *Switching ratio* = switching / stability. **Key intrinsic metric**: a value $\gg 1$ indicates the gate genuinely partitions turns into two evidence regimes.
- *$\rho$–KL Pearson correlation* across all turns: measures how informative $\rho$ is about evidence dynamics.

**(M3) Repair sparsity and coupling.** Mean $\rho$, $P(\rho > 0.5)$, $P(\rho > 0.7)$, distributional shape of $\rho$. Used to characterize the gate's behavioral regime (bimodal vs. unimodal, sparse vs. dense).

### 3.3 Baselines

To ground our numbers, we report two non-trained baselines:

- **Random.** Theoretical random ranking; $R@K = K / |\text{gallery}|$, $\text{MRR} \approx \ln N / N$.
- **CLIP-only.** Raw CLIP projected text and image features (no training, no $W_h$ learned). Pure baseline for image-text matching capacity. Per-dialogue text vector is the mean of CLIP text projections of all turns.

We also include a **`pure_corr`** ablation of our model — the same architecture as ours but with $\alpha = \beta = 0$ (only $L_{\text{corr}}$), serving as an "upper bound for contrastive-only fine-tuning."

### 3.4 Main Results

Table 1 shows the main retrieval comparison across baselines, ablations, and our gate variants. *(Numbers are single-seed v5 results pending multi-seed completion; v6 results will replace these with mean ± std.)*

**Table 1: Main retrieval results on PhotoChat held-out split.**

| Method | Supervision | R@1 | R@5 | R@10 | MRR | med rank |
|---|---|---|---|---|---|---|
| Random | none | 0.002 | 0.011 | 0.022 | 0.020 | 230 |
| CLIP-only | none | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| fixed ($\rho=0.1$) | none | 0.0369 | 0.1367 | 0.2299 | 0.1008 | 40 |
| pure_corr | none | 0.0369 | 0.1323 | 0.2278 | 0.1000 | 39 |
| DCP-B | none | 0.0282 | 0.1085 | 0.1800 | 0.0825 | 58 |
| multi_C | none | 0.0282 | 0.1432 | 0.2343 | 0.0992 | 46 |
| **multi_C+floor (ours)** | **none** | **0.0369** | **0.1605** | **0.2451** | **0.1054** | 43 |
| multi_C+floor+drop | none | 0.0304 | 0.1497 | 0.2408 | 0.1016 | 40 |

Three observations.

**(O1) ToM-inspired modulation, properly regularized, exceeds the contrastive-only ceiling.** Our final method (multi_C+floor) achieves $\text{R@5} = 0.160$, a **21% relative improvement** over the contrastive-only baseline (`pure_corr`, $\text{R@5} = 0.132$). Improvements on R@10 (+7.6%) and MRR (+5.4%) are also positive, while R@1 is preserved at 0.037 (matching `pure_corr`). The gain emerges in *soft retrieval* (top-$K$ for $K > 1$): ToM modulation does not change which image ranks first, but makes the candidate ranking more reliable, putting the correct image in the top-$K$ shortlist more often.

**(O2) Local change-point detection (DCP) is insufficient — a negative finding.** The DCP-Gate exhibits strong intrinsic dynamics: $\rho$–KL Pearson correlation 0.74 and switching ratio 311×. Yet it underperforms the simpler `fixed` baseline on retrieval (R@1: 0.028 vs. 0.037; R@5: 0.108 vs. 0.137). The reason is that local KL between $e_{t-1}$ and $\hat{e}_t$ conflates legitimate repair with topic shifts, descriptive elaborations, and grounding establishment in the early turns. Local divergence under-determines repair semantics. This negative finding motivates Variant C.

**(O3) Multi-signal gating yields continuous trust modulation, not discrete repair detection.** Variant C recovers most retrieval performance (R@5: 0.143) while strengthening the intrinsic $\rho$–KL coupling to 0.87. Critically, multi_C produces a **unimodal** $\rho$ distribution centered around 0.62 ($P(\rho > 0.5) = 100\%$), rather than DCP's bimodal $\{0, 1\}$ saturation. We interpret this as a qualitative regime shift: from *discrete repair detection* (DCP, fire only on rare events) to *continuous trust modulation* (multi_C, always blend new evidence with state- and uncertainty-modulated weights). Both are valid ToM-inspired instantiations of "intent regulates grounding," but the continuous regime is empirically friendlier to retrieval-side learning.

### 3.5 Coherence-Floor Ablation

Adding the coherence-floor regularizer $L_{\text{coh\_floor}}$ ($\tau_a = 0.20$, $\gamma_{\text{floor}} = 0.1$):

multi_C $\to$ multi_C+floor:
- R@1: 0.0282 $\to$ 0.0369 (+32% relative)
- R@5: 0.1432 $\to$ 0.1605 (+12% relative)
- R@10: 0.2343 $\to$ 0.2451 (+5% relative)
- $\rho$–KL Pearson preserved: 0.87 $\to$ 0.87

The floor term prevents the alignment head from collapsing $\tilde{a}_t$ on ambiguous turns. We hypothesize that this regularization indirectly encourages the shared text projection $W_h$ to learn sharper, more discriminative representations — visible as a clean recovery of R@1 to baseline level.

**Adding $L_{\text{coh\_drop}}$ further degrades performance** (multi_C+floor+drop, all retrieval metrics slightly below multi_C+floor). This confirms our prior hypothesis: penalizing all coherence drops structurally fights legitimate ToM repair events, in which $a_t$ is expected to dip transiently before recovering. We report this as a negative ablation result; the asymmetric drop penalty should not be included in the final method.

### 3.6 Modality Ablation

To verify that the model genuinely uses both modalities (rather than collapsing to text-only or image-only retrieval through CLIP's pre-trained alignment alone), we ablate the multi_C+floor checkpoint by replacing one modality's encoded representation with: zero, isotropic random, or the dataset mean.

*(Table to be populated with v6 multi-seed results from `outputs/ablation_table_meanstd.csv`.)*

The expected finding pattern:
- Replacing the **image** with a constant (zero/mean) collapses retrieval to near-random — confirming the image is essential.
- Replacing the **text** symmetrically collapses retrieval — confirming the dialogue is essential.
- Random-noise replacement of either modality should be worse than mean replacement (more disruptive to learned representations).

### 3.7 Distributional Analysis of $\rho$

Figure 1 (to be inserted) overlays the $\rho_t$ histograms of three trained variants on the held-out test set:

- **fixed:** a degenerate spike at $\rho = 0.1$
- **DCP-B:** bimodal mass at $\rho \approx 0.05$ and $\rho \approx 1.0$ — discrete change-point regime
- **multi_C+floor:** unimodal centered at $\rho \approx 0.62$ — continuous trust regime

This distributional shift, observable directly in the histogram, is the visual signature of the regime transition between local change-point detection and multi-signal trust modulation.

Figure 2 (to be inserted) shows per-turn $\rho_t$, $a_t$, and $d_t$ trajectories for representative dialogues. DCP-Gate exhibits "spike resets": $\rho$ jumps to 1, $a$ collapses momentarily, $d$ peaks, and the cycle repeats. multi_C+floor shows smooth tracking: $\rho$ varies in a tight range around 0.6, $a$ trends with mild oscillations, $d$ correlates with $\rho$ subtly. Both are interpretable, but only multi_C+floor preserves retrieval performance.

### 3.8 Discussion

Three takeaways relevant to ToM-inspired multimodal dialogue modeling.

1. **Unsupervised ToM modulation is achievable.** Variant C requires no human repair labels. The combination of (a) a multi-signal gate seeing local divergence + state + uncertainty, (b) a sparsity prior to prevent collapse, and (c) a coherence floor to prevent alignment collapse, suffices to learn $\rho$ that strongly correlates with evidence dynamics ($r = 0.87$) while simultaneously improving retrieval over a contrastive-only baseline.

2. **Local divergence under-determines repair.** A change-point detector built on $\mathrm{KL}(e_{t-1} \| \hat{e}_t)$ alone produces a strong intrinsic signal but a brittle one — it conflates several distinct kinds of evidence shift. Augmenting with dialogue state ($a_{t-1}$), forward-looking signal ($\hat{a}_t - a_{t-1}$), and uncertainty (entropies) is essential for the gate to behave usefully as a soft regulator rather than a noisy event detector.

3. **Coherence floor as a soft prior.** Rather than directly supervising repair events, preventing trajectory collapse via $L_{\text{coh\_floor}}$ is sufficient to recover retrieval gains. This is consistent with a broader observation in self-supervised representation learning: well-chosen *prevention* objectives often transfer further than direct *supervision* objectives.

The negative findings (DCP-B underperforms on retrieval; $L_{\text{coh\_drop}}$ interferes) are themselves informative. They demarcate the design space and rule out simpler alternatives that a reader might otherwise expect to work.

### 3.9 Limitations

- **Single dataset.** All results are on PhotoChat. Generalization to other multimodal dialogue corpora (e.g., Image-Chat, MMDialog) remains future work.
- **Frozen CLIP backbone.** Allowing CLIP fine-tuning would change the regime; we adopt frozen encoders to isolate the contribution of the ToM modulation layer.
- **Modest absolute retrieval numbers.** R@1 in the 3–4% range reflects the difficulty of dialogue-level retrieval against a 1000-image gallery with frozen 512-dimensional CLIP projections, rather than a failure of the method. A direct comparison to large vision–language models (LLaVA, BLIP-2) is outside the compute scope of this study; we focus on the *marginal contribution* of ToM modulation over a clean contrastive baseline.
- **Statistical power.** Three-seed variance estimates pending; we will report $p$-values from paired permutation tests on the seed-level differences in the camera-ready version.

---

## Appendix A: Reproducibility

All training and evaluation are driven by a single shell pipeline:

```bash
bash src/run_all_seeds.sh
```

This script (i) creates a deterministic 19/2 train/test shard split (seed 42), (ii) trains all six model variants on three random seeds $\{0, 1, 2\}$, (iii) runs eight evaluations per seed (six trained models plus two baselines), (iv) runs seven modality ablations on the winning variant per seed, and (v) aggregates everything into mean ± std tables. The pipeline is fully idempotent — interrupted runs resume from the last completed step.

Code is organized as:

```
src/
├── models/          # CLIP wrapper, alignment head, gates, losses
├── train/           # Training loop with all six variant flags
├── eval/            # Eval, baselines, viz, multi-seed aggregation
├── dataloaders/     # PhotoChat dataset + train/test split utility
└── run_v6.sh, run_all_seeds.sh   # Pipeline
```

All hyperparameters in §2.5 / §3.1 are exposed as CLI flags and logged into each run's checkpoint.
