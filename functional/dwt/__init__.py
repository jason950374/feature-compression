__all__ = [
    'DWTForward',
    'DWTInverse',
    'DWT',
    'IDWT',
]

from functional.dwt.transform2d import DWTForward, DWTInverse
DWT = DWTForward
IDWT = DWTInverse