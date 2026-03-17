# Binary Reward Summarized from Next State

Online RL for agentic tool-use, using binary process reward signals from environment feedback.

## Method Overview

The policy model is deployed as an OpenAI-compatible chat proxy. External environments (e.g. OpenClaw) send multi-turn conversations through this proxy. For each **main-line turn**, the system:

1. Forwards the request to the policy model (served by SGLang) and collects the response along with per-token log-probabilities.
2. When the **next turn** arrives, its user/environment message serves as the "next state" for the previous turn.
3. A **Process Reward Model (PRM)** judges the previous response quality given the next state (could be user or env feedback). It produces `m` independent evaluations via majority vote, scoring each turn as `+1` (good), `-1` (bad), or `0` (neutral).
4. The majority-voted score becomes the scalar reward for that turn.
5. Turns that never receive a next state (i.e. the last turn in a session) are excluded from training (`loss_mask = 0`), unless they are the only turn in the session (at-least-one guarantee).

### Advantage Estimation (GRPO)

Advantages are computed using **Group Relative Policy Optimization (GRPO)**. For each sample with scalar reward `r`, the advantage is broadcast uniformly to all response tokens:

$$A_t = r, \quad \forall t \in \text{response tokens}$$

No reward normalization is applied (`--disable-rewards-normalization`).

### Policy Gradient Loss

Standard PPO-style clipped surrogate objective with asymmetric clipping:

$$\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\Big[\min\!\big(\rho_t A_t,\ \text{clip}(\rho_t,\, 1-\varepsilon,\, 1+\varepsilon_{\text{high}}) \cdot A_t\big)\Big]$$

where $\varepsilon = 0.2$, $\varepsilon_{\text{high}} = 0.28$.

### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{pg}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

where $\beta_{\text{KL}} = 0.02$. Entropy bonus is disabled ($\beta_{\text{ent}} = 0$).



## How to Run

```bash
cd slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

### Low-Precision Modes

The launch script now supports independent rollout and training precision knobs:

- `INFER_PRECISION=bf16|fp8|int4` selects the policy serving checkpoint used by SGLang.
- `TRAIN_PRECISION=bf16|fp8` selects the Megatron training precision.
- `PRM_PRECISION=bf16|fp8|int4` optionally changes the PRM serving checkpoint; it defaults to `bf16` to avoid reward-quality regressions.

FP8 rollout + FP8 training:

```bash
cd slime
INFER_PRECISION=fp8 \
TRAIN_PRECISION=fp8 \
HF_CKPT_FP8=/path/to/Qwen3-4B-Thinking-2507-FP8 \
REF_LOAD=/path/to/Qwen3-4B-Thinking-2507_torch_dist \
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

INT4 rollout with BF16 training:

```bash
cd slime
INFER_PRECISION=int4 \
HF_CKPT_INT4=/path/to/Qwen3-4B-Thinking-2507-INT4 \
REF_LOAD=/path/to/Qwen3-4B-Thinking-2507_torch_dist \
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

If you also want the PRM to use low-precision serving, set `PRM_PRECISION=fp8` or `PRM_PRECISION=int4` and provide the matching `PRM_MODEL_PATH_FP8` / `PRM_MODEL_PATH_INT4`.



## File Structure

```
openclaw-rl/
├── README.md
├── run_qwen3_4b_openclaw_rl.sh     # Launch script
├── openclaw_api_server.py           # FastAPI proxy + PRM scoring + sample submission
├── openclaw_rollout.py              # Async rollout worker (bridges API server ↔ SLIME trainer)
└── results/                         # Runtime records (auto-created)
```
