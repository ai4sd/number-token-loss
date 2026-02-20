from importlib.metadata import version

from .core import NTLoss, NTLossDotProduct, NumberLevelLoss  # noqa
from .deprecated import NumberLevelLossLooped  # noqa

__name__ = "ntloss"
__version__ = version(__name__)
