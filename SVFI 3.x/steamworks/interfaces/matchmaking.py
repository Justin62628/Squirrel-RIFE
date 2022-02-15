from steamworks.exceptions 	import *


class SteamMatchmaking(object):
    def __init__(self, steam: object):
        self.steam = steam
        if not self.steam.loaded():
            raise SteamNotLoadedException('STEAMWORKS not yet loaded')

    def CreateLobby(self, lobby_type: int, max_members: int) -> None:
        """Create a lobby on the Steam servers, if private the lobby will not be returned by any RequestLobbyList() call

        :param lobby_type: ELobbyType
        :param max_members: int count
        :return: None
        """
        self.steam.CreateLobby(lobby_type, max_members)


    def JoinLobby(self, steam_lobby_id: int) -> None:
        """Join an existing lobby

        :param steam_lobby_id: int
        :return: None
        """
        self.steam.JoinLobby(steam_lobby_id)


    def LeaveLobby(self, steam_lobby_id: int) -> None:
        """Leave a lobby, this will take effect immediately on the client side, other users will be notified by LobbyChatUpdate_t callback

        :param steam_lobby_id: int
        :return: None
        """
        self.steam.LeaveLobby(steam_lobby_id)

    #

    def InviteUserToLobby(self, steam_lobby_id: int, steam_id: int) -> bool:
        """Invite another user to the lobby, the target user will receive a LobbyInvite_t callback, will return true if the invite is successfully sent, whether or not the target responds

        :param steam_lobby_id: int
        :param steam_id: int
        :return: bool
        """
        return self.steam.InviteUserToLobby(steam_lobby_id, steam_id)
