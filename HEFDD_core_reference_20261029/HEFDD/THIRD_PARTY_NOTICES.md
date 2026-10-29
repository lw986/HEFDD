# Attribution and licensing review (pending)

This draft is derived from the project's existing model implementations; it is
not a claim of entirely original authorship. Original source comments have been
retained. CIFAR source comments reference facebook/fb.resnet.torch and
pytorch/vision and a Yang Wei copyright notice. Review those upstream licenses,
the actual code lineage, and the project's inherited HDFGD/HEFDD code ownership
before granting a redistribution license. Also verify face backbone, ArcFace,
feature generation and attention implementation attribution.

Do not replace inherited notices with an unconditional new MIT license. The
paper authors and code contributors must confirm the appropriate license.
No dataset or third-party model weights are included.

The decoupled loss implements target/non-target knowledge distillation described
by Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu and Jiajun Liang, Decoupled Knowledge
Distillation, CVPR 2022. The standalone reference follows the project's existing
DKD numerical convention; no upstream training package is bundled. Retain this
method attribution. The new loss does not establish the provenance of historical
paper results or grant a project-wide redistribution license.
