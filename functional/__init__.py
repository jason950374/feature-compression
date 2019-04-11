__all__ = [
    'DWTForward',
    'DWTInverse',
    'DWT',
    'IDWT',
    'dct1',
    'idct1',
    'dct',
    'idct',
    'dct_2d',
    'idct_2d',
]

from functional.dwt.transform2d import DWTForward, DWTInverse
from functional.dct import dct1, idct1, dct, idct, dct_2d, idct_2d
DWT = DWTForward
IDWT = DWTInverse
