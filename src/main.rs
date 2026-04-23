use std::marker::PhantomData;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector},
    poly::Rotation,
};
use halo2curves::bn256::Fr;

// A Zero-Knowledge circuit for Linear Regression inference: y = sum(w_i * x_i) + b
// w: private weights
// x: private inputs
// b: private bias
// y: public output
#[derive(Clone)]
struct LinearRegressionConfig {
    x: Column<Advice>,
    w: Column<Advice>,
    b: Column<Advice>,
    acc: Column<Advice>, // For accumulating the sum
    s_mul: Selector, // To select multiplication & accumulation
    s_bias: Selector, // To select adding the bias at the end
    instance: Column<Instance>, // For the public output y
}

struct LinearRegressionCircuit<F: Field> {
    x: Vec<Value<F>>,
    w: Vec<Value<F>>,
    b: Value<F>,
    _marker: PhantomData<F>,
}

impl<F: Field> Circuit<F> for LinearRegressionCircuit<F> {
    type Config = LinearRegressionConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            x: vec![Value::unknown(); self.x.len()],
            w: vec![Value::unknown(); self.w.len()],
            b: Value::unknown(),
            _marker: PhantomData,
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let x = meta.advice_column();
        let w = meta.advice_column();
        let b = meta.advice_column();
        let acc = meta.advice_column();
        let instance = meta.instance_column();

        let s_mul = meta.selector();
        let s_bias = meta.selector();

        meta.enable_equality(x);
        meta.enable_equality(w);
        meta.enable_equality(b);
        meta.enable_equality(acc);
        meta.enable_equality(instance);

        // Gate for: acc_next = acc_cur + x * w
        meta.create_gate("mul_add", |meta| {
            let s_mul = meta.query_selector(s_mul);
            let x = meta.query_advice(x, Rotation::cur());
            let w = meta.query_advice(w, Rotation::cur());
            let acc_cur = meta.query_advice(acc, Rotation::cur());
            let acc_next = meta.query_advice(acc, Rotation::next());

            vec![s_mul * (acc_next - (acc_cur + x * w))]
        });

        // Gate for: acc_final = acc_cur + b
        meta.create_gate("add_bias", |meta| {
            let s_bias = meta.query_selector(s_bias);
            let b = meta.query_advice(b, Rotation::cur());
            let acc_cur = meta.query_advice(acc, Rotation::cur());
            let acc_final = meta.query_advice(acc, Rotation::next());

            vec![s_bias * (acc_final - (acc_cur + b))]
        });

        LinearRegressionConfig {
            x,
            w,
            b,
            acc,
            s_mul,
            s_bias,
            instance,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let out_cell = layouter.assign_region(
            || "linear regression",
            |mut region| {
                let mut acc_val = Value::known(F::ZERO);

                // Initialize accumulator at row 0
                region.assign_advice(
                    || "acc init",
                    config.acc,
                    0,
                    || acc_val,
                )?;

                let mut offset = 0;

                // Loop over inputs and weights
                for i in 0..self.x.len() {
                    config.s_mul.enable(&mut region, offset)?;

                    region.assign_advice(
                        || format!("x_{}", i),
                        config.x,
                        offset,
                        || self.x[i],
                    )?;

                    region.assign_advice(
                        || format!("w_{}", i),
                        config.w,
                        offset,
                        || self.w[i],
                    )?;

                    acc_val = acc_val + self.x[i] * self.w[i];
                    offset += 1;

                    // Write next acc
                    region.assign_advice(
                        || format!("acc_{}", offset),
                        config.acc,
                        offset,
                        || acc_val,
                    )?;
                }

                // Add bias
                config.s_bias.enable(&mut region, offset)?;
                
                region.assign_advice(
                    || "bias",
                    config.b,
                    offset,
                    || self.b,
                )?;
                
                acc_val = acc_val + self.b;
                offset += 1;

                let final_cell = region.assign_advice(
                    || "acc final",
                    config.acc,
                    offset,
                    || acc_val,
                )?;

                Ok(final_cell)
            },
        )?;

        // Expose the final accumulated prediction as a public input
        layouter.constrain_instance(out_cell.cell(), config.instance, 0)?;

        Ok(())
    }
}

fn main() {
    use halo2_proofs::dev::MockProver;

    println!("Starting ZKML Pipeline Simulation...");

    // 1. Mock Pre-trained Model Weights & Bias
    // y = w1*x1 + w2*x2 + b
    // Let w1 = 5, w2 = 2, b = 10
    let w = vec![Fr::from(5), Fr::from(2)];
    let b = Fr::from(10);
    
    println!("Model: y = 5*x1 + 2*x2 + 10 (Weights and Bias will be Private)");

    // 2. Mock Dataset
    // (x1, x2) -> expected_y
    let dataset = vec![
        (vec![Fr::from(1), Fr::from(3)], Fr::from(5*1 + 2*3 + 10)),   // y = 5 + 6 + 10 = 21
        (vec![Fr::from(4), Fr::from(1)], Fr::from(5*4 + 2*1 + 10)),   // y = 20 + 2 + 10 = 32
        (vec![Fr::from(0), Fr::from(5)], Fr::from(5*0 + 2*5 + 10)),   // y = 0 + 10 + 10 = 20
        (vec![Fr::from(10), Fr::from(10)], Fr::from(5*10 + 2*10 + 10)), // y = 50 + 20 + 10 = 80
    ];

    println!("Dataset of {} items loaded. Input Data will be Private.", dataset.len());
    println!("Running inference and generating Zero-Knowledge Proofs...\n");

    let k = 4; // circuit size parameter

    // 3. Pipeline execution
    for (i, (x, expected_y)) in dataset.iter().enumerate() {
        println!("Sample {}:", i);
        // Note: x, w, and b are private inputs (witnesses)
        let circuit = LinearRegressionCircuit::<Fr> {
            x: x.iter().map(|&v| Value::known(v)).collect(),
            w: w.iter().map(|&v| Value::known(v)).collect(),
            b: Value::known(b),
            _marker: PhantomData,
        };

        // expected_y is our public input (instance)
        let public_inputs = vec![*expected_y];

        // Run MockProver to simulate proof generation and verification
        let prover = MockProver::run(k, &circuit, vec![public_inputs]).unwrap();
        
        match prover.verify() {
            Ok(_) => println!("  ✅ Proof verified! Inference computed correctly. Output: {:?}", expected_y),
            Err(e) => println!("  ❌ Verification failed: {:?}", e),
        }
    }

    // 4. Test Invalid Scenarios
    println!("\nTesting Invalid Scenarios...");
    
    // Scenario A: Tampered public output
    println!("Scenario A: Tampering with public prediction output...");
    let x_val = vec![Fr::from(1), Fr::from(3)];
    let tampered_y = Fr::from(999); // Correct is 21
    
    let circuit_a = LinearRegressionCircuit::<Fr> {
        x: x_val.iter().map(|&v| Value::known(v)).collect(),
        w: w.iter().map(|&v| Value::known(v)).collect(),
        b: Value::known(b),
        _marker: PhantomData,
    };
    
    let prover_a = MockProver::run(k, &circuit_a, vec![vec![tampered_y]]).unwrap();
    match prover_a.verify() {
        Ok(_) => println!("  ❌ Error: Invalid proof was verified!"),
        Err(_) => println!("  ✅ Success: Prover correctly rejected tampered public output (999)."),
    }

    // Scenario B: Tampered weights inside the circuit (prover tries to cheat)
    println!("Scenario B: Prover uses incorrect model weights...");
    let wrong_w = vec![Fr::from(100), Fr::from(200)]; // Tampered weights
    let expected_correct_y = Fr::from(21); // Verifier expects output for w=[5,2], b=10, x=[1,3]

    let circuit_b = LinearRegressionCircuit::<Fr> {
        x: x_val.iter().map(|&v| Value::known(v)).collect(),
        w: wrong_w.into_iter().map(Value::known).collect(),
        b: Value::known(b),
        _marker: PhantomData,
    };

    let prover_b = MockProver::run(k, &circuit_b, vec![vec![expected_correct_y]]).unwrap();
    match prover_b.verify() {
        Ok(_) => println!("  ❌ Error: Invalid proof was verified!"),
        Err(_) => println!("  ✅ Success: Prover correctly rejected proof generated with tampered weights."),
    }
}
