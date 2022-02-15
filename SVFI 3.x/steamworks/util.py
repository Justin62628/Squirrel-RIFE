import sys

from steamworks.enums import Arch


def get_arch():
    """ Decice between 32 or 64 bit arch """
    if not (sys.maxsize > 2**32):
        return Arch.x86

    return Arch.x64