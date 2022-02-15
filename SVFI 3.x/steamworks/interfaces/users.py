from steamworks.structs import *
from steamworks.exceptions import *


class SteamUsers(object):
    def __init__(self, steam: object):
        self.steam = steam
        if not self.steam.loaded():
            raise SteamNotLoadedException('STEAMWORKS not yet loaded')

    def GetSteamID(self) -> int:
        """Get user's Steam ID.

        :return: int
        """
        return self.steam.GetSteamID()

    def LoggedOn(self) -> bool:
        """Check, true/false, if user is logged into Steam currently

        :return: bool
        """
        return self.steam.LoggedOn()

    def GetPlayerSteamLevel(self) -> int:
        """Get the user's Steam level.

        :return: int
        """
        return self.steam.GetPlayerSteamLevel()

    def GetGameBadgeLevel(self, series: int, foil: int) -> int:
        """Trading Card badges data access, if you only have one set of cards, the series will be 1.
        # The user has can have two different badges for a series; the regular (max level 5) and the foil (max level 1).

        :param series: int
        :param foil: int
        :return: int
        """
        return self.steam.GetGameBadgeLevel(series, foil)

    def GetAuthSessionTicket(self) -> str:
        """Retrieves an authentication ticket. Immediately usable in AuthenticateUserTicket.
		
		:return: str
		"""
        buffer = create_string_buffer(1024)
        # length = self.steam.GetAuthSessionTicket(buffer)
        # return buffer[0:length].hex().upper()
        result = self.steam.GetAuthSessionTicket(buffer)
        """
        k_EBeginAuthSessionResultOK	0	票证对此游戏和此 Steam ID 有效。
        k_EBeginAuthSessionResultInvalidTicket	1	票证无效。
        k_EBeginAuthSessionResultDuplicateRequest	2	已为此 Steam ID 提交了一个票证。
        k_EBeginAuthSessionResultInvalidVersion	3	票证来自不兼容接口版本。
        k_EBeginAuthSessionResultGameMismatch	4	不是此游戏的票证。
        k_EBeginAuthSessionResultExpiredTicket	5	票证已过期。
        """
        return result
