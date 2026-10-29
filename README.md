# HEFDD: Low-Resolution Face Recognition via Hierarchical Efficient Feature Decoupled Distillation

This repository provides the core method reference code for “Low-Resolution Face Recognition via Hierarchical Efficient Feature Decoupled Distillation.”

## Model Overview

HEFDD transfers knowledge from a high-resolution teacher to a low-resolution student through hierarchical auxiliary supervision. The Efficient Attention Feature Expansion (EAFE) module combines depthwise convolution, feature concatenation, SE channel attention, and channel compression. Decoupled knowledge distillation between auxiliary classifiers transfers semantic information across multiple stages.

This repository includes teacher networks, complete student networks, and the core distillation loss. Pretrained weights and complete training and evaluation pipelines are not included.

## Environment

Python 3.8 + PyTorch 2.0.1

```bash
conda create -n HEFDD python=3.8
conda activate HEFDD
pip install -r requirements.txt
```

## Hardware

Experiments use NVIDIA GeForce RTX 4090 D GPUs. Model structures can also be inspected on a CPU.

## Datasets

The manuscript includes experiments on the following datasets:

| Dataset | Purpose |
| --- | --- |
| CASIA-WebFace | Face teacher training and student distillation |
| LFW | Face verification |
| UCCS | Low-resolution face identification |
| TinyFace | Real-world low-resolution face identification |
| CIFAR-100 | Image classification and ablation studies |

This repository does not redistribute these datasets. Please obtain data from their official sources and comply with the respective licenses and terms of use.

## Repository Structure

```text
HEFDD/
├── models/
│   ├── __init__.py
│   ├── cifar_teacher.py
│   ├── cifar_student.py
│   ├── face_teacher.py
│   └── face_student.py
├── losses/
│   ├── __init__.py
│   └── hefdd_loss.py
├── examples/
├── requirements.txt
└── README.md
```

The face models use IResNet implementations, while the CIFAR models use CIFAR ResNet implementations. EAFE remains in the student inference path; auxiliary classifiers provide training-time supervision.

## Model Imports

```python
from models import face_teacher, face_student
from losses import hefdd_loss

teacher = face_teacher(num_classes=10575, is_aux=True)
student = face_student(num_classes=10575, is_aux=True)
```

