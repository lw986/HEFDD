"""Random-initialized model structures only; no pretrained weights."""
from .cifar_teacher import resnet110_se as cifar_teacher
from .cifar_student import resnet20_HDFG as cifar_student
from .face_teacher import iresnet50_se as face_teacher
from .face_student import iresnet18_hdfg_arcface as face_student

__all__ = ['cifar_teacher', 'cifar_student', 'face_teacher', 'face_student']
