# LSTM Model Architecture Explanation

## Overview

The Dynamic Dilemma Zone Prediction Model uses a **sequence-to-probability** architecture that predicts whether a vehicle will **STOP** or **GO** at a yellow traffic light. The model takes a temporal sequence of vehicle features and outputs a probability P(STOP).

## Architecture Components

### 1. Input Layer
- **Input Shape**: `(batch_size, sequence_length, feature_dim)`
  - `batch_size`: Number of sequences processed together (default: 32)
  - `sequence_length`: Number of time steps (default: 12 frames)
  - `feature_dim`: Number of features per time step (6 features)

**Features (6 dimensions)**:
1. `speed_ms` - Vehicle speed in m/s
2. `distance_to_stop_line` - Distance to stop line in meters
3. `ttc` - Time to collision (distance/speed)
4. `distance_to_front_vehicle` - Distance to vehicle ahead
5. `traffic_density` - Number of vehicles in scene
6. `class_id` - Vehicle type (car=2, motorcycle=3, bus=5, truck=7)

### 2. LSTM Encoder

The LSTM encoder processes the temporal sequence and extracts meaningful patterns.

#### Architecture Details:
```python
LSTMEncoder(
    input_dim=6,           # 6 features per timestep
    hidden_size=64,        # 64 hidden units
    num_layers=2,          # 2 stacked LSTM layers
    dropout=0.2,           # 20% dropout for regularization
    bidirectional=False    # Unidirectional (forward only)
)
```

#### How LSTM Works:

**LSTM Cell Structure**:
- **Forget Gate**: Decides what information to discard from previous state
- **Input Gate**: Decides what new information to store
- **Cell State**: Long-term memory that flows through the sequence
- **Output Gate**: Decides what parts of the cell state to output

**Forward Pass Through LSTM**:

1. **Time Step 1 (t-11)**: 
   - Input: `[speed, distance, ttc, front_dist, density, class]` at frame t-11
   - LSTM processes and updates hidden state `h₁` and cell state `c₁`

2. **Time Step 2 (t-10)**:
   - Input: Features at frame t-10
   - LSTM uses previous hidden state `h₁` and cell state `c₁`
   - Updates to `h₂` and `c₂`

3. **... continues through all 12 time steps ...**

4. **Time Step 12 (t-0, most recent)**:
   - Input: Features at frame t-0 (just before yellow onset)
   - Final hidden state `h₁₂` contains information from entire sequence

**Stacked Layers**:
- **Layer 1**: Processes raw input sequence → outputs intermediate representation
- **Layer 2**: Processes Layer 1's output → outputs final hidden representation

**Output**: The final hidden state from the last layer: `h_final` of shape `(batch_size, 64)`

### 3. Linear Output Layer

The linear layer maps the LSTM's hidden representation to a single probability value.

```python
Linear(
    in_features=64,    # Hidden size from LSTM
    out_features=1     # Single output (probability)
)
```

**Operation**:
```
logits = W · h_final + b
P(stop) = sigmoid(logits)
```

Where:
- `W`: Weight matrix of shape `(64, 1)`
- `b`: Bias scalar
- `sigmoid`: Activation function that maps logits to [0, 1]

### 4. Output

- **Shape**: `(batch_size, 1)`
- **Value Range**: [0, 1]
- **Interpretation**:
  - `P(stop) ≈ 0` → Vehicle will GO
  - `P(stop) ≈ 1` → Vehicle will STOP
  - `P(stop) ≈ 0.5` → Uncertain (dilemma zone)

## Complete Forward Pass Example

### Input Sequence (12 timesteps, 6 features each):
```
Timestep t-11: [10.5, 45.2, 4.3, 12.0, 5, 2]  # 11 frames before yellow
Timestep t-10: [10.8, 42.1, 3.9, 11.5, 5, 2]  # 10 frames before yellow
...
Timestep t-1:  [11.2, 15.3, 1.4, 10.2, 5, 2]  # 1 frame before yellow
Timestep t-0:  [11.0, 12.1, 1.1, 10.0, 5, 2]  # Just before yellow
```

### Processing Flow:

1. **LSTM Layer 1**:
   - Processes each timestep sequentially
   - Maintains hidden state that accumulates information
   - Output: Intermediate representation for each timestep

2. **LSTM Layer 2**:
   - Processes Layer 1's output
   - Further refines temporal patterns
   - Final hidden state: `h_final = [0.23, -0.45, 0.67, ..., 0.12]` (64 values)

3. **Linear Layer**:
   - `logits = W · h_final + b`
   - Example: `logits = 0.85`

4. **Sigmoid Activation**:
   - `P(stop) = sigmoid(0.85) = 0.70`
   - **Prediction**: 70% probability of STOP

## Training Process

### 1. Data Preparation

**Sequence Building**:
- Extract last 12 frames before yellow light onset for each vehicle
- Each frame contains 6 features
- Label: 0 (GO) or 1 (STOP) based on actual behavior

**Normalization**:
- Features are standardized (mean=0, std=1) or min-max scaled
- Prevents features with large values from dominating

### 2. Training Loop

**Forward Pass**:
```python
# Input: batch of sequences
sequences = (batch_size, 12, 6)

# LSTM processes sequence
hidden_states = LSTM(sequences)  # (batch_size, 64)

# Linear layer
logits = Linear(hidden_states)   # (batch_size, 1)

# Sigmoid activation
predictions = sigmoid(logits)    # (batch_size, 1) in [0, 1]
```

**Loss Calculation**:
```python
# Binary Cross-Entropy Loss
loss = -[y_true * log(pred) + (1 - y_true) * log(1 - pred)]
```

**Backward Pass**:
- Compute gradients using backpropagation through time (BPTT)
- Gradients flow back through:
  1. Linear layer weights
  2. LSTM Layer 2 weights
  3. LSTM Layer 1 weights
- Update all parameters using Adam optimizer

### 3. Key Training Features

**Dropout (0.2)**:
- Randomly sets 20% of LSTM units to zero during training
- Prevents overfitting
- Only active during training, disabled during inference

**Learning Rate Scheduling**:
- Starts with learning rate 0.001
- Reduces by 50% if validation loss plateaus
- Helps fine-tune model in later epochs

**Early Stopping**:
- Stops training if validation loss doesn't improve for 10 epochs
- Prevents overfitting
- Saves best model based on validation performance

## Why This Architecture Works

### 1. Temporal Modeling
- **LSTM** captures long-term dependencies in vehicle behavior
- Can remember patterns from early in the sequence (e.g., initial speed)
- Handles variable-length patterns through hidden state

### 2. Feature Learning
- **Stacked layers** learn hierarchical features:
  - Layer 1: Basic temporal patterns (speed changes, distance changes)
  - Layer 2: Complex patterns (acceleration patterns, approach behavior)

### 3. Interpretability
- **Linear output layer** allows feature importance analysis
- Can extract which features contribute most to predictions
- SHAP values show contribution of each timestep and feature

## Model Parameters

**Total Parameters**: ~51,777

**Breakdown**:
- LSTM Layer 1: 
  - Input-to-hidden: 6 × 64 × 4 = 1,536
  - Hidden-to-hidden: 64 × 64 × 4 = 16,384
  - Biases: 64 × 4 = 256
  - **Subtotal**: ~18,176

- LSTM Layer 2:
  - Input-to-hidden: 64 × 64 × 4 = 16,384
  - Hidden-to-hidden: 64 × 64 × 4 = 16,384
  - Biases: 64 × 4 = 256
  - **Subtotal**: ~33,024

- Linear Output Layer:
  - Weights: 64 × 1 = 64
  - Bias: 1
  - **Subtotal**: 65

- **Total**: ~51,265 (plus dropout parameters)

## Example: How Model Makes a Decision

### Scenario: Vehicle approaching yellow light

**Input Sequence** (last 12 frames):
- Frames t-11 to t-6: Vehicle at constant speed (12 m/s), distance decreasing
- Frames t-5 to t-2: Speed increases slightly (13 m/s), distance to stop line: 25m
- Frame t-1: Speed 13.5 m/s, distance: 15m
- Frame t-0: Speed 14 m/s, distance: 10m

**LSTM Processing**:
1. **Early frames**: LSTM learns "vehicle maintaining speed"
2. **Middle frames**: LSTM detects "slight acceleration"
3. **Recent frames**: LSTM detects "strong acceleration near stop line"

**Hidden State**: Encodes pattern: "Accelerating vehicle approaching stop line"

**Linear Layer**: 
- Weights emphasize acceleration patterns
- Output: High logit value

**Sigmoid**: 
- P(stop) = 0.15 (low probability)
- **Prediction**: GO (vehicle is accelerating, likely to continue)

## Advantages of This Architecture

1. **Temporal Awareness**: Captures how vehicle behavior evolves over time
2. **Feature Interaction**: Learns complex relationships between features
3. **Robustness**: Dropout and regularization prevent overfitting
4. **Interpretability**: Linear output allows feature importance analysis
5. **Efficiency**: Relatively small model (~51K parameters) trains quickly

## Limitations

1. **Fixed Sequence Length**: Requires exactly 12 frames (padded if shorter)
2. **Unidirectional**: Only processes forward in time (no future context)
3. **Feature Engineering**: Relies on pre-computed features (speed, distance, etc.)
4. **Binary Output**: Only predicts STOP/GO probability, not confidence intervals

## Summary

The LSTM model architecture is a **temporal sequence classifier** that:
1. Takes 12 frames of vehicle features as input
2. Processes them through 2 stacked LSTM layers to extract temporal patterns
3. Maps the final hidden state to a probability using a linear layer
4. Outputs P(STOP) which determines the vehicle's predicted behavior

The model learns to recognize patterns like:
- "Decelerating vehicle → likely to STOP"
- "Accelerating vehicle → likely to GO"
- "Constant speed near stop line → uncertain (dilemma zone)"

This architecture is well-suited for time-series prediction tasks where temporal patterns are crucial for making accurate predictions.

