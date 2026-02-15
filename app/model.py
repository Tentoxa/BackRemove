import logging
from withoutbg import WithoutBG

logger = logging.getLogger(__name__)

_model = None


def load_model():
    global _model
    if _model is None:
        logger.info("Loading model...")
        _model = WithoutBG.opensource()
        logger.info("Model ready.")
    return _model


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded.")
    return _model
