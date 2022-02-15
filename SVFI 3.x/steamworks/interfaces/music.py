from steamworks.exceptions 	import *


class SteamMusic(object):
    def __init__(self, steam: object):
        self.steam = steam
        if not self.steam.loaded():
            raise SteamNotLoadedException('STEAMWORKS not yet loaded')


    def MusicIsEnabled(self) -> bool:
        """Is Steam music enabled

        :return: bool
        """
        return self.steam.MusicIsEnabled()


    def MusicIsPlaying(self) -> bool:
        """Is Steam music playing something

        :return: bool
        """
        return self.steam.MusicIsPlaying()


    def MusicGetVolume(self) -> float:
        """Get the volume level of the music.

        :return: float
        """
        return self.steam.MusicGetVolume()


    def MusicPause(self) -> None:
        """Pause whatever Steam music is playing

        :return: None
        """
        self.steam.MusicPause()


    def MusicPlay(self) -> None:
        """Play current track/album.

        :return: None
        """
        self.steam.MusicPlay()


    def MusicPlayNext(self) -> None:
        """Play next track/album.

        :return: None
        """
        self.steam.MusicPlayNext()


    def MusicPlayPrev(self) -> None:
        """Play previous track/album.

        :return: None
        """
        self.steam.MusicPlayPrev()


    def MusicSetVolume(self, volume: float) -> None:
        """Set the volume of Steam music

        :param volume: float 0,0 -> 1,0
        :return: None
        """
        self.steam.MusicSetVolume(volume)