# Optimized Logistic Regression with a Hybrid Optimization Strategy

## Project Overview

This project implements **binary logistic regression from scratch with NumPy** using a neural-network-style workflow: initialize parameters, perform forward propagation, compute binary cross-entropy loss, calculate gradients, update parameters, and make predictions.

The original notebook classifies **cat vs. non-cat RGB images**. Each 64 × 64 × 3 image is converted into a 12,288-feature vector and normalized before training.

The project was extended using ideas from the accompanying `Optimization_methods.ipynb` notebook. The goal is not to turn logistic regression into a deep neural network, but to improve how its parameters are optimized.

## Main Goal

The project has two goals:

1. Build a correct logistic-regression image classifier using only NumPy.
2. Improve training efficiency and convergence by combining the useful parts of several optimization techniques:
   - mini-batch gradient descent,
   - Momentum,
   - Adam,
   - learning-rate decay.

The optimized notebook also adds numerical-stability and vectorization improvements.

## Important Preprocessing Correction

The original notebook reshaped the image tensor directly into `(features, examples)`. Although the resulting dimensions looked correct, this can mix the memory layout of different examples and break the intended image-to-label alignment.

The optimized version uses:

```python
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
```

This first flattens **each image independently**, then transposes the result so every column is one training example.

This correction is separate from the optimizer. Any fair baseline-vs-optimized comparison should use the corrected preprocessing for both models.

## Optimization Strategy

The new training method is called a **hybrid optimizer**. It adapts and renames functions from the optimization notebook for the single weight vector `w` and scalar bias `b` used by logistic regression.

### 1. Random Mini-Batches

The original logistic-regression model performs one full-batch update after processing the entire training set.

The optimized model shuffles the training examples and divides them into smaller mini-batches. The first few epochs use ordinary mini-batch gradient descent before the Momentum and Adam stages.

Benefits:

- parameters are updated multiple times during one pass through the dataset;
- each update uses less data at once;
- shuffled batches introduce useful gradient variation;
- the optimizer can make progress before completing a full-dataset calculation.

The function is renamed to:

```python
create_logistic_mini_batches(...)
```

### 2. Plain Gradient-Descent Warm-Up

The optimizer begins with a very short ordinary gradient-descent stage, adapted from `update_parameters_with_gd()` in the optimization notebook.

Benefits:

- keeps the first updates simple and easy to interpret;
- explicitly preserves the basic gradient-descent method from the optimization notebook;
- gives the model a few direct descent steps before adding moving-average state.

The renamed function is:

```python
gradient_descent_step(...)
```

### 3. Momentum Warm-Up

After the plain-gradient-descent warm-up, the optimizer uses a short Momentum stage.

Momentum keeps an exponentially weighted average of recent gradients:

```text
velocity = beta * previous_velocity + (1 - beta) * current_gradient
```

Benefits:

- smooths noisy mini-batch gradients;
- reduces oscillation;
- builds movement in directions where gradients repeatedly agree.

The project uses Momentum only as a short warm-up instead of applying it throughout the entire Adam phase.

### 4. Adam Optimization

After the warm-up, training switches to Adam.

Adam combines two important ideas:

- a **first moment**, which behaves like Momentum by averaging recent gradients;
- a **second moment**, which tracks squared gradients and scales each parameter update adaptively.

Bias correction is also included, matching the optimization notebook.

The renamed function is:

```python
adam_logistic_step(...)
```

Benefits:

- adapts the effective step size for individual parameters;
- handles gradients with different magnitudes more effectively;
- usually reaches a useful solution with fewer passes through the data than plain fixed-learning-rate gradient descent.

### 5. Scheduled Learning-Rate Decay

A fixed learning rate may be useful early in training but too large close to convergence.

The hybrid optimizer therefore reduces the learning rate gradually:

```text
learning_rate =
initial_learning_rate /
(1 + decay_rate * (epoch / time_interval))
```

The function is renamed to:

```python
scheduled_logistic_lr(...)
```

Benefits:

- larger useful updates early in training;
- smaller, more stable updates later;
- less overshooting near a minimum.

### 6. Why the Update Rules Are Staged Instead of Stacked

Adam already contains a Momentum-like first-moment calculation. Applying an independent Momentum update on top of Adam during every step would duplicate part of the same idea and would not automatically improve the model.

For that reason, this project uses a **staged hybrid**:

```text
Mini-batches
    ↓
Very short plain-gradient-descent warm-up
    ↓
Short Momentum warm-up
    ↓
Adam adaptive updates
    ↓
Learning-rate decay throughout training
```

This uses the main methods from the optimization notebook while keeping their roles clear.

## Additional Efficiency and Stability Improvements

### Vectorized Prediction

The original prediction function used a Python loop to convert probabilities into labels.

The optimized version uses:

```python
(probabilities > 0.5).astype(float)
```

This removes the per-example Python loop and lets NumPy perform the operation in vectorized form.

### Stable Sigmoid

The input to `exp()` is clipped before computing the sigmoid. This reduces the chance of numerical overflow for very large positive or negative values.

### Stable Binary Cross-Entropy

Predicted probabilities are clipped away from exactly `0` and `1` before applying `log()`. This prevents `log(0)` and makes the training code more robust.

## How the Hybrid Method Improves Efficiency

The baseline model is configured for **2,000 full-batch iterations**.

The hybrid model is configured for **800 epochs** by default. Each epoch sees the complete training set once, but performs several mini-batch updates during that pass.

By configuration, this reduces complete passes over the training data from 2,000 to 800, which is a **60% reduction in dataset passes**. Adam performs extra moment calculations and mini-batching adds Python overhead, so wall-clock speed should be measured rather than assumed.

For that reason, the notebook includes a fair comparison that reports:

- training accuracy,
- test accuracy,
- training time,
- cost history,
- learning-rate history.

The actual runtime and final accuracy depend on the computer, package versions, and hyperparameters used.

## Main Optimized Functions

| Function | Purpose |
|---|---|
| `create_logistic_mini_batches()` | Shuffle data and build synchronized mini-batches |
| `gradient_descent_step()` | Perform the short plain-gradient-descent warm-up update |
| `initialize_logistic_optimizer_state()` | Initialize Momentum and Adam state |
| `scheduled_logistic_lr()` | Decay the learning rate |
| `momentum_warmup_step()` | Perform the early Momentum update |
| `adam_logistic_step()` | Perform bias-corrected Adam updates |
| `hybrid_optimize_logistic()` | Combine mini-batches, Momentum, Adam, and decay |
| `train_optimized_logistic_regression()` | Train, predict, time, and evaluate the optimized model |

## Default Hybrid Hyperparameters

```python
num_epochs = 800
learning_rate = 0.005
mini_batch_size = 32
gd_warmup_epochs = 5
momentum_warmup_epochs = 20
momentum_beta = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
decay_rate = 0.1
decay_interval = 100
```

These are reasonable starting values rather than guaranteed universal best settings. The comparison cells make it easy to tune them and observe the effect.

## Running the Project

Open:

```text
Optimized_Logistic_Regression_Hybrid.ipynb
```

Then run the notebook from top to bottom.

The original course helper files are still required by the notebook, including the dataset loader and test utilities imported by the original project, for example:

```text
lr_utils.py
public_tests.py
```

The dataset expected by `lr_utils.load_dataset()` must also be available in the environment.

The final optional custom-image cell expects:

```text
images/my_image.jpg
```

You can replace that filename with your own image.

## Evaluating the Result

The notebook contains a comparison section that trains both:

1. the corrected full-batch logistic-regression baseline;
2. the hybrid optimized logistic-regression model.

When discussing results, use the printed measurements from that section. In particular, do not attribute an accuracy change caused by the corrected image flattening to the optimizer itself.

A useful final report should compare:

- baseline training accuracy vs. optimized training accuracy;
- baseline test accuracy vs. optimized test accuracy;
- baseline runtime vs. optimized runtime;
- cost reduction behavior;
- whether the optimized model reaches a strong result using fewer full passes over the dataset.

## Files

```text
Optimized_Logistic_Regression_Hybrid.ipynb
README_Optimized_Logistic_Regression.md
```

## Summary

The final model keeps the original logistic-regression mathematics while improving the optimization process. The most important optimization change is the staged combination of **mini-batch gradient descent, a short plain-gradient-descent warm-up, Momentum, Adam adaptive updates, and learning-rate decay**.

This design uses the optimization methods in complementary roles instead of treating them as interchangeable algorithms. The result is a cleaner training pipeline that makes more frequent updates, adapts step sizes, stabilizes later training, and provides direct measurements for evaluating whether the optimization actually improves efficiency on the cat/non-cat classification task.
