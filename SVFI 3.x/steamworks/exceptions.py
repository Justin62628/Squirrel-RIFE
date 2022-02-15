

class SteamException(Exception):
    pass


class GenericSteamException(SteamException):
    pass


class UnsupportedPlatformException(SteamException):
    pass


class UnsupportedArchitectureException(SteamException):
    pass


class MissingSteamworksLibraryException(SteamException):
    pass


class SteamNotLoadedException(SteamException):
    pass


class SteamNotRunningException(SteamException):
    pass


class SteamConnectionException(SteamException):
    pass


class UnsupportedSteamStatValue(SteamException):
    pass


class SetupRequired(SteamException):
    pass