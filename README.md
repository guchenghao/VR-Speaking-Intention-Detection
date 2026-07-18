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
