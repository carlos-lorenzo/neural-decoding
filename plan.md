## Plan: Professional EEG ML Research Methodology

This plan outlines a robust, iterative methodology for developing and testing novel Machine Learning approaches on EEG data, reflecting the practices of professional researchers.

**Steps**
1. **Literature Review & Ideation (The "Why" and "What")**
    *   **Continuous reading:** Stay updated on bio-signal processing, Graph Neural Networks (GNNs), Transformers, and CNNs applied to EEG.
    *   **Identify gaps:** Look for limitations in existing models (e.g., poor generalization across subjects, high computational cost, ignoring spatial electrode relationships).
    *   **Formulate hypotheses:** E.g., "Integrating spatial adjacency (distance between electrodes) into a GCN will improve cross-subject motor imagery classification compared to standard CNNs."
2. **Data Pipeline & Baseline Establishment (The "Foundation")**
    *   **Robust Preprocessing:** Standardize filtering (e.g., bandpass 4-38Hz), artifact removal, and epoching. Create reusable scripts for this (like your `filtering.py`).
    *   **Solid Baselines:** Implement well-known models (EEGNet, DeepConvNet, CSP+SVM) first. Evaluate them rigorously. You cannot prove a novel approach is better without a strong baseline.
    *   **Validation Validation:** Decide on train/val/test splits. For EEG, prioritize Leave-One-Subject-Out (LOSO) cross-validation to test true generalization, rather than pooling all subjects (which leaks data).
3. **Iterative Prototyping (The "How")**
    *   **Modular Code:** Build models in interchangeable blocks (e.g., temporal extractor separate from spatial extractor, as seen in your STGCN).
    *   **Start Simple:** If proposing a new GNN, start with a tiny dataset or a few subjects. Overfit a single batch to ensure the code and gradients flow correctly.
    *   **Ablation Studies:** If your novel model has 3 new components (e.g., custom adjacency matrix, new convolution type, attention layer), test each one separately against the baseline to see what actually contributes to performance.
4. **Experiment Tracking & Analysis (The "Evaluation")**
    *   **Logging:** Use tools like Weights & Biases (WandB), MLflow, or TensorBoard. Track hyperparameters, train/val loss, and accuracy per subject.
    *   **Beyond Accuracy:** Analyze confusion matrices, ROC curves, and class-wise performance.
    *   **Error Analysis:** Look at the specific trials the model got wrong. Is there noise? Are certain subjects consistently failing?
5. **Refinement & Publication/Scaling (The "Next Steps")**
    *   **Interpretability:** Can you visualize what the model learned? For EEG, visualizing spatial filters or attention maps over electrodes is crucial for clinical/neuroscientific validity.
    *   **Hyperparameter Tuning:** Once the architecture shows promise, use automated searches (Optuna, Ray Tune) to squeeze out performance.
    *   **Document & Share:** Keep detailed notes on failed experiments (they are just as important).

**Verification**
1. Ensure all novel models are compared against EEGNet using a strict Leave-One-Subject-Out cross-validation.
2. Track experiments with a logging tool to compare runs objectively rather than relying on notebook cell outputs.

**Decisions**
- Methodology emphasizes rigorous baselines and cross-subject generalization (LOSO) over simple pooled accuracy.