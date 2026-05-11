# zk-machine-learning

A small ZKML pipeline in Rust: a [halo2](https://github.com/zcash/halo2) PLONK circuit that proves a linear-regression inference

```
y = w₁·x₁ + w₂·x₂ + ⋯ + b
```

was computed correctly — without revealing the model weights, the bias, or the input features. Only the predicted output `y` is public.

## What's in the circuit

`src/main.rs` defines `LinearRegressionCircuit<F: Field>` over the BN256 scalar field, with:

- Two custom gates:
  - **`mul_add`** — at each row, constrain `acc_next = acc_cur + xᵢ · wᵢ`.
  - **`add_bias`** — at the final row, constrain `acc_final = acc_cur + b`.
- Four advice columns (`x`, `w`, `b`, `acc`) and one instance column for the public output.
- A single region in `synthesize` that walks the dataset row-by-row, accumulating the dot product and exposing `acc_final` as the public instance.

Witnesses (private):
- `x` — input features
- `w` — model weights
- `b` — bias

Instance (public): the predicted `y`.

## Running

```bash
cargo run --release
```

Output:

```
Starting ZKML Pipeline Simulation...
Model: y = 5*x1 + 2*x2 + 10 (Weights and Bias will be Private)
Dataset of 4 items loaded. Input Data will be Private.
Running inference and generating Zero-Knowledge Proofs...

Sample 0:
  ✅ Proof verified! ... Output: 21
Sample 1:
  ✅ Proof verified! ... Output: 32
...

Testing Invalid Scenarios...
Scenario A: Tampering with public prediction output...
  ✅ Success: Prover correctly rejected tampered public output (999).
Scenario B: Prover uses incorrect model weights...
  ✅ Success: Prover correctly rejected proof generated with tampered weights.
```

Proofs are generated and verified in-process via `halo2_proofs::dev::MockProver` — useful for circuit-correctness testing without setting up a trusted setup or doing real proof aggregation.

## Stack

- [`halo2_proofs`](https://crates.io/crates/halo2_proofs) `0.3.2` — PLONKish arithmetization
- [`halo2curves`](https://crates.io/crates/halo2curves) `0.9.0` — BN256 curve / `Fr` scalar field

## Why this exists

A minimal worked example of ZKML: the smallest interesting ML model (a linear layer) running inside a SNARK, with adversarial scenarios that show why each constraint matters.
