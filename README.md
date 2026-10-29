# HEFDD: Core Method Reference

This is a model-only reference snapshot extracted from the research codebase.
It contains teacher and complete student network structures, including feature
generation, channel attention, compression and auxiliary classification heads.
It includes a standalone CE + main KL + hierarchical AUX DKD objective.
It does **not** include training scripts, checkpoints,
pretrained weights, benchmark reproduction pipelines or reported-accuracy guarantees.
All models are randomly initialized. This is not a full experimental release.

## Included variants

| Variant | Teacher | Student | Auxiliary outputs |
| --- | --- | --- | --- |
| CIFAR | CIFAR ResNet110 with auxiliary heads | CIFAR ResNet20 with EAFE | 3 |
| Face | IResNet50 with optional auxiliary heads | IResNet18 with EAFE and optional auxiliary heads | 4 |

The face implementation is **IResNet**, not torchvision ResNet. The two families
are distinct implementations and should not be interchanged in paper tables.
Legacy HDFG identifiers denote the EAFE implementation in this snapshot.
The expansion ratio r is concatenated channels / original channels; newly
generated channels equal (r-1) times the original channels. Default r=2.

## Structure and interfaces

- `models/cifar_teacher.py`: backbone, classification head and auxiliary heads.
- `models/cifar_student.py`: backbone, three EAFE blocks, main and auxiliary heads.
- `models/face_teacher.py`: face backbone, ArcFace head and optional four AUX heads.
- `models/face_student.py`: face backbone, four EAFE blocks, ArcFace and AUX heads.

Import constructors from `models`. Set `num_classes` explicitly to your label
space. CIFAR inputs normally have shape N x 3 x 32 x 32. Face experiments must
state whether the actual network input is16 or112; structural shape support
does not establish accuracy at that resolution. BatchNorm training requires
appropriate batch sizes; model inspection can use eval mode.

For CIFAR, teacher forward returns `(logits, aux_logits)`. Student forward with
`kd=True` returns the same, while `kd=False` returns main logits only. Set
`grad=True` when auxiliary supervision should reach the backbone; the inherited
default `grad=False` detaches intermediate features. This default is preserved,
not silently changed by packaging.

For face models, construct with `is_aux=True` to include auxiliary heads. Forward
with `is_feat=True, is_aux=True, grad=True` returns main and auxiliary logits.
Provide integer identity labels when using the inherited ArcFace margin path.
Without labels the head behavior remains as implemented in the source; do not
interpret its classification output as a deployed face embedding.

Auxiliary heads are training-only structures. Disabling their forward computation
does not delete their allocated parameters. An inference-only artifact must omit
unused AUX and, for face verification, replace the classification endpoint with
the intended embedding endpoint. EAFE stays in the student backbone.
The `_se` names are legacy: teacher SE modules reside in AUX branches, not the
teacher backbone; student EAFE includes SE channel attention.

## Core loss interface

`losses.hefdd_loss` consumes main and AUX logits plus integer labels, and returns
the total loss and individual tensor components. Teacher outputs are detached.
The caller must freeze the teacher, use eval mode, and ensure student AUX
features are not detached (`grad=True`). EAFE remains part of the student.
DKD follows Zhao et al., Decoupled Knowledge Distillation, CVPR 2022; the
implementation retains this project's probability clamp and -1000 target mask.
It is a reference implementation, not a claim of exact historical reproduction.

`aux_reduction='sum'` is the manuscript stage-sum convention. Historical E3
experiments used `mean`; select that explicitly to reproduce that convention.
For N heads, sum with weight w/N equals mean with weight w. Do not compare the
two with the same weight as if they were identical. Defaults T=3, alpha=1,
beta=8 are interface defaults, not a complete validated experimental recipe.
Both families use the same class space within each teacher/student pair.

## Dependencies

Python 3.8+ and PyTorch; see `requirements.txt`. The release does not install
packages or download weights automatically. CPU structural checks were performed
with PyTorch 2.7.1+cpu in the development environment; the server reference
dependency is PyTorch 2.0.1. This is not a hardware performance report.

## Examples and release status

`examples/images/` is reserved for 20 public-dataset sample images. **Images are
not included yet** because the official download has not completed. The selected
source is Face Synthetics, subject to its non-commercial research data agreement.
No CASIA, LFW or TinyFace images are bundled. See `examples/README.md`.

This is a **pre-publication draft**. Confirm the final paper/author metadata,
code ownership and upstream license permissions before publishing. No license
grant is implied by this draft. See `THIRD_PARTY_NOTICES.md`.
Source hashes and mechanical export provenance are in `MODEL_SOURCES.json`.
