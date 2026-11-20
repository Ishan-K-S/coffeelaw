# Coffee Law

> **Note for reviewers.** This is an anonymized copy for double-blind review. The anonymizer replaces some names and links, so the three `config_frozen.json` files in `experiments/ceilings/prereg/` won't match the SHA-256 hashes printed in the paper until the copy is de-anonymized. The experiment code, frozen configurations and result files are in `experiments/ceilings/`. The raw run records aren't included.

> **A collective of agents can never know more about the world than its
> starting data plus what its tools can carry.**

```
I(Θ ; everything the collective holds)  ≤  min( H(Θ),  I(Θ ; D₀) + Σ_t C_t )
```

`Θ` is the truth, `D₀` is the starting data and `C_t` is the Shannon capacity of
round `t`'s tool calls. The law holds for any kind of agent — LLMs, people, or
programs — any number of them, any communication protocol and any query
strategy, under four assumptions about the environment. Fano's inequality turns
it into a floor on error. Debate and voting add nothing to the budget; only
starting data and informative tool calls raise it.

Underneath sits a second ceiling, **reasoning**:
`usable knowledge ≤ available information ≤ budget`. More computation (thinking
longer, more agents, a code interpreter) raises what the collective can *use*,
up to what it *holds*, never past it
([Theorem 8](formal_proofs/formal_proofs.md#theorem-8)). A third ceiling is
**communication**: when data is split across agents, no single agent can use it
until enough has been shared
([Theorem 14](formal_proofs/extensions.md#theorem-14)). Only total compute
counts, and the compute needed ranges from linear to infeasible depending on the
task ([Theorems 9–10](formal_proofs/formal_proofs.md#theorem-9)).

**Read it:** <https://mbhatt1.github.io/coffeelaw/>

- [Blog post: Agent collectives have bounds](formal_proofs/agent-collectives-have-bounds.md)
- [Formal proofs](formal_proofs/formal_proofs.md) — the Coffee Law is
  [Theorem 7](formal_proofs/formal_proofs.md#theorem-7); the reasoning ceiling is
  [Theorem 8](formal_proofs/formal_proofs.md#theorem-8)
- [Paper (IEEE format, PDF)](paper/coffee_law.pdf) — see [`paper/README.md`](paper/README.md) for the target venue
- [Extensions](formal_proofs/extensions.md) — sandboxes, acting, communication,
  a changing world, adversaries, continuous truths, what a model's weights
  know, and what remains open
- [Information geometry](formal_proofs/proof.md)
- [From any agent to LLMs](formal_proofs/llm_specialization.md)

**Check it:**

```bash
cd formal_proofs
python3 verify.py && python3 verify_llm.py && python3 verify_geometry.py
python3 verify_extensions.py   # needs numpy
python3 verify_capacity.py
cd lean && lake exe cache get && lake build   # Lean 4 proofs
```

The Python checks are exact numerical evaluations of the worked examples plus
randomized tests of the general inequalities. The core of the law (the Markov
condition `W — (D₀, 𝒯) — A` for any collective, data processing, and the
output-entropy bound `I(W; O) ≤ log |𝒪|` per tool call) is proved in Lean 4.
[Theorem 20](formal_proofs/extensions.md#theorem-20) turns tool capacity into a
number read off the interface: byte or token limits and response schemas. See
[`formal_proofs/README.md`](formal_proofs/README.md) for details.

[MIT License](LICENSE)
