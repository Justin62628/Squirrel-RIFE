from steamworks.enums 		import *
from steamworks.exceptions 	import *


class SteamFriends(object):
    def __init__(self, steam: object):
        self.steam = steam
        if not self.steam.loaded():
            raise SteamNotLoadedException('STEAMWORKS not yet loaded')


    def GetFriendCount(self, flag: bytes = FriendFlags.ALL) -> int:
        """ Get number of friends user has

        :param flag: FriendFlags
        :return: int
        """
        return self.steam.GetFriendCount(flag.value)

    #
    def GetFriendByIndex(self, friend_index: int, flag: bytes = FriendFlags.ALL) -> int:
        """Get a friend by index

        :param friend_index: int position
        :param flag: FriendFlags
        :return: int steam64
        """
        return self.steam.GetFriendByIndex(friend_index, flag.value)


    def GetPlayerName(self) -> str:
        """Get the user's Steam username

        :return: str
        """
        return self.steam.GetPersonaName()


    def GetPlayerState(self) -> int:
        """Get the user's state on Steam

        :return: int
        """
        return self.steam.GetPersonaState()


    def GetFriendPersonaName(self, steam_id: int) -> str:
        """ Get given friend's Steam username

        :param steam_id: int
        :return: str
        """
        return self.steam.GetFriendPersonaName(steam_id)


    def SetGameInfo(self, server_key, server_value) -> None:
        """Set the game information in Steam; used in 'View Game Info'
        # Steamworks documentation is missing this method, still relevant?
        :param serverKey: str
        :param serverValue: str
        :return: None
        """
        self.steam.SetGameInfo(server_key, server_value)


    def ClearGameInfo(self) -> None:
        """Clear the game information in Steam; used in 'View Game Info'
        # Steamworks documentation is missing this method, still relevant?
        :return: None
        """
        self.steam.ClearGameInfo()


    def InviteFriend(self, steam_id: int, connection: str) -> None:
        """Invite friend to current game/lobby
        # Steamworks documentation is missing this function but "InviteUserToGame" is present, does this need an update?
        :param steam_id: int steam64
        :param connection: str connection string
        :return:
        """
        self.steam.InviteFriend(steam_id, connection)


    def SetPlayedWith(self, steam_id: int) -> None:
        """Set player as 'Played With' for game

        :param steam_id: int steam64
        :return: None
        """
        self.steam.SetPlayedWith(steam_id)


    def ActivateGameOverlay(self, dialog: str = '') -> None:
        """Activates the overlay with optional dialog

        :param dialog: str ["Friends", "Community", "Players", "Settings", "OfficialGameGroup", "Stats", "Achievements", "LobbyInvite"]
        :return: None
        """
        self.steam.ActivateGameOverlay(dialog.encode())


    def ActivateGameOverlayToUser(self, dialog: str, steam_id: int) -> None:
        """Activates the overlay to the specified dialog

        :param dialog: str ["steamid", "chat", "jointrade", "stats", "achievements", "friendadd", "friendremove", "friendrequestaccept", "friendrequestignore"]
        :param steam_id: int steam64
        :return: None
        """
        self.steam.ActivateGameOverlayToWebPage(dialog.encode(), steam_id)


    def ActivateGameOverlayToWebPage(self, url: str) -> None:
        """Activates the overlay with specified web address

        :param url: str
        :return: None
        """
        self.steam.ActivateGameOverlayToWebPage(url.encode())


    def ActivateGameOverlayToStore(self, app_id: int) -> None:
        """Activates the overlay with the application/game Steam store page

        :param app_id: int
        :return: None
        """
        self.steam.ActivateGameOverlayToWebPage(app_id)


    def ActivateGameOverlayInviteDialog(self, steam_lobby_id: int) -> None:
        """Activates game overlay to open the invite dialog. Invitations will be sent for the provided lobby

        :param steam_lobby_id:
        :return: None
        """
        self.steam.ActivateGameOverlayInviteDialog(steam_lobby_id)
