"""Runtime model builder for the Noise-CNN (inference-only bundle)."""


def make_model(pretrained: bool = True):
    import timm
    return timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=1)
