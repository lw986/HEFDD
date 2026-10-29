"""CE + main KL + hierarchical AUX DKD; no training loop or model loading."""
import math

import torch
import torch.nn.functional as F


def _validate(student, teacher, labels, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if student.ndim != 2 or teacher.shape != student.shape or min(student.shape) < 1 or student.shape[1] < 2:
        raise ValueError('Expected matching nonempty BxC logits, C >= 2')
    if student.device != teacher.device or labels.device != student.device:
        raise ValueError('Inputs must share a device')
    if labels.dtype != torch.long or labels.shape != (student.shape[0],):
        raise ValueError('Expected long labels of shape B')
    if not student.is_floating_point() or not teacher.is_floating_point():
        raise ValueError('Logits must be floating point')
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError('Nonfinite logits')
    if labels.min() < 0 or labels.max() >= student.shape[1]:
        raise ValueError('Label outside class range')


def dkd_components(student, teacher, labels, temperature=3.0):
    """Project-compatible FP32 DKD, including its clamp and -1000 mask."""
    _validate(student, teacher, labels, temperature)
    student, teacher = student.float(), teacher.detach().float()
    gt = torch.zeros_like(student).scatter_(1, labels[:, None], 1).bool()
    ps, pt = F.softmax(student / temperature, dim=1), F.softmax(teacher / temperature, dim=1)

    def binary(p):
        return torch.cat(((p * gt).sum(1, keepdim=True),
                          (p * ~gt).sum(1, keepdim=True)), dim=1)

    tckd = F.kl_div(binary(ps).clamp_min(1e-12).log(), binary(pt),
                    reduction='batchmean') * temperature ** 2
    nckd = F.kl_div(F.log_softmax(student / temperature - 1000. * gt, dim=1),
                    F.softmax(teacher / temperature - 1000. * gt, dim=1),
                    reduction='batchmean') * temperature ** 2
    return tckd, nckd


def hefdd_loss(student, teacher, student_aux, teacher_aux, labels, *,
               temperature=3.0, alpha=1.0, beta=8.0,
               ce_weight=1.0, main_weight=1.0, aux_weight=1.0,
               aux_reduction='sum'):
    """Return total and tensor components; teacher logits are always detached.

    'sum' follows the manuscript stage-sum convention. 'mean' matches the
    historical E3 convention. Weights are illustrative, not a benchmark recipe.
    """
    _validate(student, teacher, labels, temperature)
    if aux_reduction not in ('sum', 'mean'):
        raise ValueError('aux_reduction must be sum or mean')
    if len(student_aux) == 0 or len(student_aux) != len(teacher_aux):
        raise ValueError('Expected equal nonempty AUX lists')
    if not all(math.isfinite(w) and w >= 0 for w in
               (alpha, beta, ce_weight, main_weight, aux_weight)):
        raise ValueError('Weights must be finite and nonnegative')
    pieces = []
    for s, t in zip(student_aux, teacher_aux):
        if s.shape != student.shape or t.shape != teacher.shape:
            raise ValueError('AUX logits must share the main class space')
        target, other = dkd_components(s, t, labels, temperature)
        pieces.append(alpha * target + beta * other)
    layers = torch.stack(pieces)
    aux = layers.sum() if aux_reduction == 'sum' else layers.mean()
    ce = F.cross_entropy(student.float(), labels)
    main = F.kl_div(F.log_softmax(student.float() / temperature, dim=1),
                    F.softmax(teacher.detach().float() / temperature, dim=1),
                    reduction='batchmean') * temperature ** 2
    total = ce_weight * ce + main_weight * main + aux_weight * aux
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite HEFDD loss')
    return total, {'ce': ce, 'main_kl': main, 'aux_dkd': aux, 'aux_layers': layers}
