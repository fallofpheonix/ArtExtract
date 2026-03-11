try:
    from .crnn import CRNNMultiTask
except Exception:  # pragma: no cover
    CRNNMultiTask = None  # type: ignore

try:
    from .multispectral import MultiSpectralMultiTaskModel, TaskFlags
except Exception:  # pragma: no cover
    MultiSpectralMultiTaskModel = None  # type: ignore
    TaskFlags = None  # type: ignore

__all__ = ["CRNNMultiTask", "MultiSpectralMultiTaskModel", "TaskFlags"]
