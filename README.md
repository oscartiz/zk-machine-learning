# ZKML Rust Pipeline 🚀

A Zero-Knowledge Machine Learning (ZKML) pipeline built in Rust using the [Halo2](https://github.com/zcash/halo2) proving system. 

This repository demonstrates how to execute a machine learning inference task (specifically, a Linear Regression model) and generate a cryptographic proof that the computation was done correctly—**without revealing the underlying model weights, the bias, or the input data**.

## How It Works 🧠

The core of this project is the `LinearRegressionCircuit`. It operates under a **Full Privacy** model:
- **Private Inputs (Data):** The input features to the model `x` are assigned to Halo2 advice columns, keeping them completely hidden.
- **Private Weights (Model):** The pre-trained weights `w` and bias `b` are also assigned to advice columns. The verifier never sees them.
- **Public Output (Prediction):** The final computed prediction `y` is assigned to an instance column, making it public.

The Prover demonstrates: *"I know a set of weights and some input data that result in exactly this prediction output, and I computed it correctly according to the model architecture."*

### Circuit Architecture
1. **`mul_add` gate:** Multiplies each input feature by its corresponding weight and accumulates the sum (`acc_next = acc_cur + x * w`).
2. **`add_bias` gate:** Adds the final bias to the accumulated sum (`acc_final = acc_cur + b`).

## Getting Started 🛠️

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (Edition 2024 or later)
- Cargo

### Running the Pipeline
To run the ZKML pipeline simulation over the mock dataset, simply execute:
```bash
cargo run
```

### Expected Output
The program will load a mock dataset of 4 items, run the inference using a set of private weights (`y = 5*x1 + 2*x2 + 10`), and verify the Zero-Knowledge proofs for each sample.

```text
Starting ZKML Pipeline Simulation...
Model: y = 5*x1 + 2*x2 + 10 (Weights and Bias will be Private)
Dataset of 4 items loaded. Input Data will be Private.
Running inference and generating Zero-Knowledge Proofs...

Sample 0:
  ✅ Proof verified! Inference computed correctly. Output: 0x0000000000000000000000000000000000000000000000000000000000000015
...
```
*(Note: Output values are displayed in hexadecimal. For example, `0x15` is `21`)*

The program also includes **Invalid Scenarios** testing to ensure the circuit is robust:
- It attempts to verify a tampered public output.
- It attempts to verify a proof generated with incorrect/fraudulent weights.
Both scenarios are expected to be correctly rejected by the verifier.
