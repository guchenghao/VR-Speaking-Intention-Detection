# VR-Speaking-Intention-Detection


## Overview

This repository provides the Keras implementation of the multimodal model proposed in the paper.

The model integrates:

- Motion features from a VR headset and controllers
- ECG-derived physiological features
- Modality-specific temporal encoders
- Cross-attention-based multimodal fusion

This repository provides the model implementation.
## Model Input

The model expects two inputs in the paper:

```python
ecg_input_shape = (3, 500)
motion_input_shape = (20, 200)
```

### Personalized Time-Series Split

For personalized evaluation, each participant was processed
independently using a chronological time-series split.

Because speaking-intention events are temporally sparse and their
distributions vary substantially across participants, applying an
identical fixed split to every participant can occasionally produce
validation folds containing only negative samples. Such folds are not
suitable for evaluating binary classification performance, especially
AUROC, which requires both classes to be present.

We therefore used participant-specific temporal fold boundaries. The
boundaries were adjusted, when necessary, to ensure that each validation
fold contained positive speaking-intention samples while strictly
preserving temporal order. No temporal shuffling was performed.

The fold boundaries were not selected based on model performance.
Label information was used only to avoid degenerate validation folds
without positive samples.

In addition, the continuous timeline was split before sliding-window
generation, reducing overlap between training and validation windows
around fold boundaries.



### Metric Clarification

In the published paper, Precision and Recallwere reported without explicitly specifying the averaging strategy. 
These metrics were computed using macro averaging across the speaking-intention and non-intention classes.
The reported F1-score was calculated as the harmonic mean of these macro-averaged Precision and Recall values.
Macro averaging was adopted to give equal weight to both classes under class-imbalanced conditions. 
across participants, the exact fold boundaries may therefore vary
between participants.



# Data Augmentation

As an extension to the data augmentation procedure described in our paper, we provide the implementation used for participant-wise class balancing. The minority class of each participant was augmented using jittering, scaling, time warping, or their combinations.

Jittering introduces point-wise Gaussian perturbations, while scaling modifies signal magnitude over time. Time warping applies a smooth nonlinear temporal deformation while preserving the synchronization among channels within the same modality. ECG and ACC are augmented independently, allowing moderate modality-specific temporal and signal variations.

All augmentations are treated as label-preserving perturbations for improving model robustness.
