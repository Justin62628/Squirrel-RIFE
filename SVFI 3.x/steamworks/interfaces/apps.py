from steamworks.exceptions 	import *


class SteamApps(object):
    def __init__(self, steam: object):
        self.steam = steam
        if not self.steam.loaded():
            raise SteamNotLoadedException('STEAMWORKS not yet loaded')


    def IsSubscribed(self) -> bool:
        """Is user subscribed to current app

        :return: bool
        """
        return self.steam.IsSubscribed()


    def IsLowViolence(self) -> bool:
        """Checks if the license owned by the user provides low violence depots

        :return: bool
        """
        return self.steam.IsLowViolence()


    def IsCybercafe(self) -> bool:
        """Checks whether the current App ID is for Cyber Cafes

        :return: bool
        """
        return self.steam.IsCybercafe()


    def IsVACBanned(self) -> bool:
        """Checks if the user has a VAC ban on their account

        :return: bool
        """
        return self.steam.IsVACBanned() or False


    def GetCurrentGameLanguage(self) -> str:
        """Gets the current language that the user has set

        :return: str language code
        """
        return self.steam.GetCurrentGameLanguage() or 'None'


    def GetAvailableGameLanguages(self) -> str:
        """Gets a comma separated list of the languages the current app supports

        :return: str language codes
        """
        return self.steam.GetAvailableGameLanguages() or 'None'


    def IsSubscribedApp(self, app_id: int) -> bool:
        """Checks if the active user is subscribed to a specified App ID

        :param app_id: int
        :return: bool
        """
        return self.steam.IsSubscribedApp(app_id)


    def IsDLCInstalled(self, dlc_id: int) -> bool:
        """Checks if the user owns a specific DLC and if the DLC is installed

        :param dlc_id: int
        :return: bool
        """
        return self.steam.IsDLCInstalled(dlc_id)


    def GetEarliestPurchaseUnixTime(self, app_id: int) -> int:
        """Gets the time of purchase of the specified app in Unix epoch format (time since Jan 1st, 1970)

        :param app_id: int
        :return: int timestamp
        """
        return self.steam.GetEarliestPurchaseUnixTime(app_id)


    def IsSubscribedFromFreeWeekend(self) -> bool:
        """Checks if the user is subscribed to the current app through a free weekend
        This function will return false for users who have a retail or other type of license.
        Suggested you contact Valve on how to package and secure your free weekend properly.

        :return: bool
        """
        return self.steam.IsSubscribedFromFreeWeekend()


    def GetDLCCount(self) -> int:
        """Get the number of DLC the user owns for a parent application/game

        :return: int
        """
        return self.steam.GetDLCCount()


    def InstallDLC(self, dlc_id: int) -> None:
        """Allows you to install an optional DLC

        :param dlc_id: int
        :return: None
        """
        self.steam.InstallDLC(dlc_id)


    def UninstallDLC(self, dlc_id: int) -> None:
        """Allows you to uninstall an optional DLC

        :param dlc_id: int
        :return: None
        """
        self.steam.UninstallDLC(dlc_id)


    def MarkContentCorrupt(self, missing_files_only: bool = True) -> bool:
        """ Allows you to force verify game content on next launch

        :param missing_files_only: bool
        :return: bool
        """
        return self.steam.MarkContentCorrupt(missing_files_only)


    def GetAppInstallDir(self, app_id: int) -> str:
        """Gets the install folder for a specific AppID

        :param app_id: int
        :return: str install location
        """
        return self.steam.GetAppInstallDir(app_id).decode()


    def IsAppInstalled(self, app_id: int) -> bool:
        """Check if given application/game is installed, not necessarily owned

        :param app_id: int
        :return: bool
        """
        return self.steam.IsAppInstalled(app_id)


    def GetAppOwner(self) -> int:
        """ Gets the Steam ID of the original owner of the current app. If it's different from the current user then it is borrowed

        :return: int
        """
        return self.steam.GetAppOwner()


    def GetLaunchQueryParam(self, key: str) -> str:
        """Gets the associated launch parameter if the game is run via sdk://run/<appid>/?param1=value1;param2=value2;param3=value3 etc

        :param key: str
        :return: str
        """
        return self.steam.GetLaunchQueryParam(key)


    def GetAppBuildId(self) -> int:
        """Return the build ID for this app; will change based on backend updates

        :return: int build id
        """
        return self.steam.GetAppBuildId()


    def GetFileDetails(self, filename: str) -> None:
        """Asynchronously retrieves metadata details about a specific file in the depot manifest

        :param filename:
        :return: None
        """
        self.steam.GetFileDetails(filename)