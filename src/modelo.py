"""
Backbone InceptionResnetV1 preentrenado en VGGFace2 (facenet-pytorch) + cabeza
ArcFace (margen angular aditivo), el estandar en reconocimiento facial de
produccion: separa mejor las identidades que un softmax normal.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class ArcMarginProduct(nn.Module):
    def __init__(self, embedding_dim: int, num_clases: int, escala: float, margen: float):
        super().__init__()
        self.escala = escala
        self.margen = margen
        self.peso = nn.Parameter(torch.empty(num_clases, embedding_dim))
        nn.init.xavier_uniform_(self.peso)
        self.cos_m = math.cos(margen)
        self.sin_m = math.sin(margen)
        self.th = math.cos(math.pi - margen)
        self.mm = math.sin(math.pi - margen) * margen

    def forward(self, embeddings, etiquetas=None):
        coseno = F.linear(F.normalize(embeddings), F.normalize(self.peso))

        if etiquetas is None:
            # Inferencia: sin margen, solo similitud coseno escalada (logits interpretables).
            return coseno * self.escala

        seno = torch.sqrt((1.0 - coseno ** 2).clamp(0, 1))
        coseno_con_margen = coseno * self.cos_m - seno * self.sin_m
        coseno_con_margen = torch.where(coseno > self.th, coseno_con_margen, coseno - self.mm)

        one_hot = torch.zeros_like(coseno)
        one_hot.scatter_(1, etiquetas.view(-1, 1), 1.0)
        logits = one_hot * coseno_con_margen + (1.0 - one_hot) * coseno
        return logits * self.escala


class ModeloIdentificador(nn.Module):
    def __init__(self, num_clases: int, embedding_dim: int = 512, escala: float = 64.0, margen: float = 0.5):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained="vggface2", classify=False)
        self.cabeza = ArcMarginProduct(embedding_dim, num_clases, escala, margen)

    def embeber(self, x):
        return self.backbone(x)

    def forward(self, x, etiquetas=None):
        embeddings = self.backbone(x)
        return self.cabeza(embeddings, etiquetas)

    def congelar_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def descongelar_ultimas_capas(self):
        """Fase 2: afina solo el bloque final del backbone, no desde cero."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for modulo in (self.backbone.block8, self.backbone.last_linear, self.backbone.last_bn):
            for p in modulo.parameters():
                p.requires_grad = True
