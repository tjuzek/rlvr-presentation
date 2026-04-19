## The Training Loop

### Step by step

1. **Start** from the DPO checkpoint (not base, not SFT)
2. **Sample** a batch of prompts from the 30K training set
3. **Generate** completions from the current policy $\pi_\theta$
4. **Verify** each completion with the appropriate verifier
5. **Assign reward**: $r = 10$ if correct, $0$ if wrong, $-10$ if no end-of-sequence token
6. **Update** the policy using PPO with KL penalty against the reference

### Implementation details that matter

- **Value model** initialised from a general reward model (not the policy)
- **Dropout disabled** during RL (deterministic log-probs for PPO ratio)
- **Non-EOS penalty**: $-10$ reward if model doesn't produce a stop token
- **~13 epochs** over 30K prompts, checkpointing every 40&ndash;100 steps

### Infrastructure

Asynchronous RL with **vLLM** for inference, **DeepSpeed ZeRO-3** for training. 8B model: ~65h on 8&times;H100.

<aside class="notes">
Each of these implementation details was validated by ablation. Getting PPO right is notoriously tricky -- Engstrom et al. 2020 showed that implementation details can matter more than the algorithm choice. Additional details: advantage whitening (normalise advantages by subtracting mean, dividing by std) is critical for stable training. Prompts are shuffled between epochs. Best checkpoint selected on development evaluation.
</aside>
