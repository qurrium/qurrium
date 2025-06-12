"""Post Processing - Magnetization Square (:mod:`qurry.process.magnet_square`)"""

from .magsq_core import BACKEND_AVAILABLE as magnet_square_availability

from .magnet_square import magnet_square, z_dir_magnet_square, MagnetSquare
from .magsq_core import DEFAULT_PROCESS_BACKEND
