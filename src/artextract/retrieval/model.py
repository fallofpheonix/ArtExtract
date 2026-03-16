from artextract.reconstruction.unet import ReconstructionUNet

# Backwards-compatible alias to ReconstructionUNet.
# Legacy arguments (in_ch, out_ch, base_ch) are no longer natively supported
# but this exact alias is expected by the test suite to resolve code duplication.
UNetRetrieval = ReconstructionUNet

__all__ = ["UNetRetrieval"]
