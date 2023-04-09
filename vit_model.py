import torch
from vit_pytorch.efficient import ViT
from linformer import Linformer
from vit_pytorch.deepvit import DeepViT


def vit_model(channel):
    efficient_transformer = Linformer(dim=128,
                                      seq_len=64 + 1,
                                      depth=12,
                                      heads=8,
                                      k=64)
    model = ViT(
        dim=128,
        image_size= 256,
        patch_size = 32,
        num_classes=10,
        transformer=efficient_transformer,
        channels= channel
    )
    return model


