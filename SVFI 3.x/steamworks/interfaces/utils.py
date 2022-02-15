from steamworks.enums 		import *
from steamworks.exceptions 	import *


class SteamUtils(object):
    def __init__(self, steam: object):
        self.steam = steam
        if not self.steam.loaded():
            raise SteamNotLoadedException('STEAMWORKS not yet loaded')


    def OverlayNeedsPresent(self) -> bool:
        """Checks if the Overlay needs a present. Only required if using event driven render updates.

        :return: bool
        """
        return self.steam.OverlayNeedsPresent()


    def GetAppID(self) -> int:
        """Get the Steam ID of the running application/game

        :return: int
        """
        return self.steam.GetAppID()


    def GetCurrentBatteryPower(self) -> int:
        """Get the amount of battery power, clearly for laptops

        :return: int [0-100] in % / 255 when on ac power
        """
        return self.steam.GetCurrentBatteryPower()


    def GetIPCCallCount(self) -> int:
        """Returns the number of IPC calls made since the last time this function was called

        :return: int
        """
        return self.steam.GetIPCCallCount()


    def GetIPCountry(self) -> str:
        """Get the user's country by IP

        :return: str
        """
        return self.steam.GetIPCountry() or ''


    def GetSecondsSinceAppActive(self) -> int:
        """Return amount of time, in seconds, user has spent in this session

        :return: int
        """
        return self.steam.GetSecondsSinceAppActive()


    def GetSecondsSinceComputerActive(self) -> int:
        """Returns the number of seconds since the user last moved the mouse

        :return: int
        """
        return self.steam.GetSecondsSinceComputerActive()

    def GetServerRealTime(self) -> int:
        """Returns the Steam server time in Unix epoch format. (Number of seconds since Jan 1, 1970 UTC)

        :return: int
        """
        return self.steam.GetServerRealTime()


    def GetSteamUILanguage(self) -> str:
        """Get the Steam user interface language

        :return: str
        """
        return self.steam.GetSteamUILanguage()


    def IsOverlayEnabled(self) -> bool:
        """Returns true/false if Steam overlay is enabled

        :return: bool
        """
        return self.steam.IsOverlayEnabled()


    def IsSteamInBigPictureMode(self) -> bool:
        """Returns true if Steam & the Steam Overlay are running in Big Picture mode

        :return: bool
        """
        return self.steam.IsSteamInBigPictureMode()


    def IsVRHeadsetStreamingEnabled(self) -> bool:
        """Is Steam running in VR

        :return: bool
        """
        return self.steam.IsVRHeadsetStreamingEnabled()


    def SetOverlayNotificationInset(self, horizontal: int, vertical: int) -> None:
        """Sets the inset of the overlay notification from the corner specified by SetOverlayNotificationPosition

        :param horizontal: int
        :param vertical: int
        :return: None
        """
        self.steam.SetOverlayNotificationInset(horizontal, vertical)


    def SetOverlayNotificationPosition(self, position: ENotificationPosition) -> None:
        """Set the position where overlay shows notifications

        :return: None
        """
        self.steam.SetOverlayNotificationPosition(position)


    def SetVRHeadsetStreamingEnabled(self, enabled: bool) -> None:
        """Set whether the HMD content will be streamed via Steam In-Home Streaming

        :param enabled: bool
        :return: None
        """

        self.steam.SetVRHeadsetStreamingEnabled(enabled)


    def ShowGamepadTextInput(self, \
                             input_mode: EGamepadTextInputLineMode, line_input_mode: EGamepadTextInputMode, \
                             description: str, max_characters: int, preset: str) -> bool:
        """Activates the Big Picture text input dialog which only supports gamepad input

        :param input_mode: EGamepadTextInputLineMode
        :param line_input_mode: EGamepadTextInputMode
        :param description: str
        :param max_characters: int
        :param preset: str
        :return: bool
        """
        return self.steam.ShowGamepadTextInput(input_mode, line_input_mode, description, max_characters, preset)


    def StartVRDashboard(self) -> None:
        """Ask SteamUI to create and render its OpenVR dashboard

        :return: None
        """
        self.steam.StartVRDashboard()