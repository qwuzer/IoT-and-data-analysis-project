# Task List: Dynamic Dilemma Zone Modeling System

## Relevant Files

- `LSTM/sequence_builder.py` - Module for building vehicle sequences from CSV data (extracts last N frames before yellow onset).
- `LSTM/model_architecture.py` - Defines LSTM/CNN sequence encoder and linear output layer for STOP/GO prediction.
- `LSTM/train_model.py` - Main training script that loads data, builds sequences, trains model, and saves artifacts.
- `LSTM/evaluate_model.py` - Evaluation script that computes AUC, F1, calibration curves, ROC, and Precision-Recall curves.
- `LSTM/explainability.py` - SHAP analysis, feature importance, temporal attribution, and partial dependence plots.
- `LSTM/dilemma_zone_generator.py` - Generates dynamic dilemma zone heatmaps and contour plots for different vehicle types.
- `LSTM/visualization.py` - Creates scatter plots, trajectory clusters, time-series analysis, and TTC distribution visualizations.
- `LSTM/utils.py` - Utility functions for data loading, normalization, sequence padding, and date-based splitting.
- `LSTM/config.py` - Configuration file for hyperparameters, paths, and model settings.

### Notes

- All model-related code should be placed in the new `LSTM/` directory (create this folder if it doesn't exist).
- The existing `inference_example.py` already generates the required CSV data, so this task list focuses on the ML pipeline.
- Use PyTorch for model implementation, SHAP for explainability, and matplotlib/seaborn for visualizations.

## Instructions for Completing Tasks

**IMPORTANT:** As you complete each task, you must check it off in this markdown file by changing `- [ ]` to `- [x]`. This helps track progress and ensures you don't skip any steps.

Example:
- `- [ ] 1.1 Read file` → `- [x] 1.1 Read file` (after completing)

Update the file after completing each sub-task, not just after completing an entire parent task.

## Tasks

- [x] 0.0 Create feature branch
  - [x] 0.1 Create and checkout a new branch for this feature (e.g., `git checkout -b feature/dilemma-zone-modeling`)
  - [x] 0.2 Create the `LSTM/` directory in the speed_estimation folder
  - [x] 0.3 Create `__init__.py` file in `LSTM/` directory to make it a Python package
- [x] 1.0 Data Processing and Sequence Building
  - [x] 1.1 Create `LSTM/config.py` with configuration parameters (sequence length, feature dimensions, paths, hyperparameters)
  - [x] 1.2 Create `LSTM/utils.py` with utility functions for CSV loading and data validation
  - [x] 1.3 Implement function to load CSV data and filter rows where `yellow_light_decision` is valid (0 or 1)
  - [x] 1.4 Implement function to group data by `tracker_id` and identify yellow onset frames
  - [x] 1.5 Implement sequence extraction function that extracts last N frames (8-15 frames, ~0.8-1.5 seconds) before yellow onset
  - [x] 1.6 Implement feature extraction for sequences: [speed_ms, distance_to_stop_line, ttc, distance_to_front_vehicle, traffic_density, class_id]
  - [x] 1.7 Implement sequence padding/truncation to ensure consistent sequence length
  - [x] 1.8 Create `LSTM/sequence_builder.py` that combines all sequence building logic
  - [x] 1.9 Implement date-based train/validation split (not random split)
  - [x] 1.10 Add data normalization/scaling functions for features
- [x] 2.0 Model Architecture Implementation
  - [x] 2.1 Create `LSTM/model_architecture.py` with base model class
  - [x] 2.2 Implement LSTM sequence encoder with configurable hidden size (32-64)
  - [x] 2.3 Implement alternative Temporal CNN encoder (1D Conv with kernel=3, stride=1)
  - [x] 2.4 Implement linear output layer with sigmoid activation for binary classification
  - [x] 2.5 Create model factory function to choose between LSTM and CNN encoder
  - [x] 2.6 Implement forward pass that returns STOP probability P(stop) = sigmoid(wᵀz + b)
  - [x] 2.7 Add model initialization with proper weight initialization
  - [x] 2.8 Add model summary/architecture visualization method
- [x] 3.0 Training Pipeline Development
  - [x] 3.1 Create `LSTM/train_model.py` as main training script
  - [x] 3.2 Implement data loading with PyTorch DataLoader
  - [x] 3.3 Implement training loop with binary cross-entropy loss
  - [x] 3.4 Add optimizer setup (Adam) with configurable learning rate
  - [x] 3.5 Implement validation loop to monitor training progress
  - [x] 3.6 Add early stopping mechanism based on validation loss
  - [x] 3.7 Implement model checkpointing to save best model during training
  - [x] 3.8 Add learning rate scheduling
  - [x] 3.9 Implement logging of training metrics (loss, accuracy per epoch)
  - [x] 3.10 Add model saving functionality that exports: model.pt, feature metadata, sequence normalization parameters
  - [x] 3.11 Add command-line arguments for training configuration (epochs, batch size, learning rate, etc.)
- [x] 4.0 Model Evaluation and Metrics
  - [x] 4.1 Create `LSTM/evaluate_model.py` evaluation script
  - [x] 4.2 Implement function to load trained model and normalization parameters
  - [x] 4.3 Implement prediction function that returns probabilities for test sequences
  - [x] 4.4 Calculate and display AUC (Area Under ROC Curve) score
  - [x] 4.5 Calculate and display F1 score for STOP/GO classification
  - [x] 4.6 Generate ROC curve plot and save as image
  - [x] 4.7 Generate Precision-Recall curve plot and save as image
  - [x] 4.8 Generate calibration curve to assess probability calibration
  - [x] 4.9 Calculate confusion matrix and classification report
  - [x] 4.10 Add function to evaluate model on validation set and display all metrics
- [x] 5.0 Explainability Features
  - [x] 5.1 Create `LSTM/explainability.py` module
  - [x] 5.2 Install and configure SHAP library
  - [x] 5.3 Implement SHAP explainer for the trained model (using DeepExplainer or KernelExplainer)
  - [x] 5.4 Generate feature importance rankings using SHAP values
  - [x] 5.5 Implement temporal feature attribution (SHAP values per timestep in sequence)
  - [x] 5.6 Generate SHAP summary plots for feature importance
  - [x] 5.7 Generate SHAP force plots for individual predictions
  - [x] 5.8 Implement partial dependence plots for speed_ms feature
  - [x] 5.9 Implement partial dependence plots for distance_to_stop_line feature
  - [x] 5.10 Implement partial dependence plots for ttc feature
  - [x] 5.11 Save all explainability outputs (plots, values) to organized directory structure
- [x] 6.0 Dynamic Dilemma Zone Generation
  - [x] 6.1 Create `LSTM/dilemma_zone_generator.py` module
  - [x] 6.2 Implement grid generation for speed_ms ∈ [0, 25] and distance_to_stop_line ∈ [0, 60]
  - [x] 6.3 Implement function to construct synthetic input sequences for each grid point
  - [x] 6.4 Implement prediction function that predicts P(stop) for each grid point using trained model
  - [x] 6.5 Generate 2D heatmap showing speed × distance → P(stop) using matplotlib/seaborn
  - [x] 6.6 Draw contour lines at P=0.3, P=0.5, and P=0.7 on heatmap
  - [x] 6.7 Highlight dilemma zone region where P ∈ [0.45, 0.55]
  - [x] 6.8 Implement per-vehicle-type DZ generation (separate maps for cars, trucks, motorcycles)
  - [x] 6.9 Add colorbar and labels to heatmaps for clarity
  - [x] 6.10 Save DZ heatmaps as PNG files with descriptive filenames
- [x] 7.0 Visualization and Analytics Tools
  - [x] 7.1 Create `LSTM/visualization.py` module
  - [x] 7.2 Implement speed-distance scatter plot colored by STOP/GO labels
  - [x] 7.3 Implement trajectory clustering visualization using K-means or DBSCAN
  - [x] 7.4 Generate average deceleration patterns plot (time-series analysis)
  - [x] 7.5 Generate braking behavior clusters visualization
  - [x] 7.6 Implement TTC distribution analysis (histogram/boxplot) comparing STOP vs GO groups
  - [x] 7.7 Generate traffic density influence comparison (DZ curves for low vs high density)
  - [x] 7.8 Add function to create comprehensive analytics report with all visualizations
  - [x] 7.9 Ensure all plots have proper labels, legends, and titles
  - [x] 7.10 Save all visualizations to organized output directory

