from steamworks.structs 	import *
from steamworks.exceptions 	import *


class SteamUserStats(object):
    _LeaderboardFindResult_t = CFUNCTYPE(None, FindLeaderboardResult_t)
    _LeaderboardFindResult = None

    def __init__(self, steam: object):
        self.steam = steam
        if not self.steam.loaded():
            raise SteamNotLoadedException('STEAMWORKS not yet loaded')


    def GetAchievement(self, name: str) -> bool:
        """Return true/false if use has given achievement

        :param name: str
        :return: bool
        """
        return self.steam.GetAchievement(name.encode('ascii'))


    def GetNumAchievements(self) -> int:
        """Get the number of achievements defined in the App Admin panel of the Steamworks website.

        :return: int
        """
        return self.steam.GetNumAchievements()


    def GetAchievementName(self, index: int) -> str:
        """Gets the 'API name' for an achievement index between 0 and GetNumAchievements.

        :param index: int
        :return: str
        """
        return self.steam.GetAchievementName(index)


    def GetAchievementDisplayAttribute(self, name: str, key: str) -> str:
        """Get general attributes for an achievement. Currently provides: Name, Description, and Hidden status.

        :param name: str
        :param key: str
        :return: str
        """
        return self.steam.GetAchievementDisplayAttribute(name.encode('ascii'), key)


    def GetStatFloat(self, name: str) -> float:
        """Get the value of a float statistic

        :param name: str
        :return: float
        """
        return self.steam.GetStatFloat(name.encode('ascii'))


    def GetStatInt(self, name: str) -> float:
        """Get the value of an integer statistic

        :param name: str
        :return: float
        """
        return self.steam.GetStatInt(name.encode('ascii'))


    def ResetAllStats(self, achievements: bool) -> bool:
        """Reset all Steam statistics; optional to reset achievements

        :param achievements: bool
        :return: bool
        """
        return self.steam.ResetAllStats(achievements)


    def RequestCurrentStats(self) -> bool:
        """Request all statistics and achievements from Steam servers

        :return: bool
        """
        return self.steam.RequestCurrentStats()


    def SetAchievement(self, name: str) -> bool:
        """Set a given achievement

        :param name: str
        :return: bool
        """
        return self.steam.SetAchievement(name.encode('ascii'))


    def SetStat(self, name: str, value: object) -> bool:
        """Set a statistic

        :param name: str
        :param value: float, int
        :return:
        """
        if isinstance(value, float):
            return self.steam.SetStatFloat(name.encode('ascii'), c_float(value))

        elif isinstance(value, int):
            return self.steam.SetStatInt(name.encode('ascii'), c_int(value))

        else:
            raise UnsupportedSteamStatValue("SetStat value can be only int or float")


    def StoreStats(self) -> bool:
        """Store all statistics, and achievements, on Steam servers; must be called to "pop" achievements

        :return: bool
        """
        return self.steam.StoreStats()


    def ClearAchievement(self, name: str) -> bool:
        """Clears a given achievement

        :param name: str
        :return: bool
        """
        return self.steam.ClearAchievement(name.encode('ascii'))


    def SetFindLeaderboardResultCallback(self, callback: object) -> bool:
        """Set callback for when leaderboard search result becomes available

        :param callback: callable
        :return: bool
        """
        self._LeaderboardFindResult = self._LeaderboardFindResult_t(callback)
        self.steam.Leaderboard_SetFindLeaderboardResultCallback(self._LeaderboardFindResult)
        return True


    def FindLeaderboard(self, name: str, callback: object = None, override_callback: bool = False) -> bool:
        """Find Leaderboard by name

        :param name: str
        :param callback: callable
        :param override_callback: bool
        :return: bool
        """
        if callback:
            if self._LeaderboardFindResult and override_callback:
                self.SetFindLeaderboardResultCallback(callback)

        else:
            self.SetFindLeaderboardResultCallback(callback)

        Steam.cdll.Leaderboard_FindLeaderboard(name.encode())
        return True