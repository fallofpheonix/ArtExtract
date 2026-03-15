from artextract.reconstruction.unet import ReconstructionUNet


class UNetRetrieval(ReconstructionUNet):
    """
    Backwards-compatible wrapper around :class:`ReconstructionUNet`.

    Historically, ``UNetRetrieval`` accepted the keyword arguments
    ``in_ch``, ``out_ch`` and ``base_ch``. ``ReconstructionUNet`` now
    expects ``in_channels``, ``out_channels`` and ``base_channels``.

    This wrapper maps the legacy names to the new ones so that existing
    call sites continue to work.
    """

    def __init__(self, *args, **kwargs):
        # Map legacy kwarg names to the new ones if provided.
        if "in_ch" in kwargs and "in_channels" not in kwargs:
            kwargs["in_channels"] = kwargs.pop("in_ch")
        if "out_ch" in kwargs and "out_channels" not in kwargs:
            kwargs["out_channels"] = kwargs.pop("out_ch")
        if "base_ch" in kwargs and "base_channels" not in kwargs:
            kwargs["base_channels"] = kwargs.pop("base_ch")

        super().__init__(*args, **kwargs)


__all__ = ["UNetRetrieval"]
