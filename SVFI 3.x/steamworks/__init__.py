"""
Copyright (c) 2016 GP Garcia, CoaguCo Industries

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
__version__ = '2.0.0'
__author__  = 'GP Garcia'

import sys, os, time
import hashlib
import steamworks.util 		as steamworks_util
from steamworks.enums 		import *
from steamworks.structs 	import *
from steamworks.exceptions 	import *
from steamworks.methods 	import STEAMWORKS_METHODS

from steamworks.interfaces.apps         import SteamApps
from steamworks.interfaces.friends      import SteamFriends
from steamworks.interfaces.matchmaking  import SteamMatchmaking
from steamworks.interfaces.music        import SteamMusic
from steamworks.interfaces.screenshots  import SteamScreenshots
from steamworks.interfaces.users        import SteamUsers
from steamworks.interfaces.userstats    import SteamUserStats
from steamworks.interfaces.utils        import SteamUtils
from steamworks.interfaces.workshop     import SteamWorkshop

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.environ['LD_LIBRARY_PATH'] = os.getcwd()

class STEAMWORKS(object):
    """
        Primary STEAMWORKS class used for fundamental handling of the STEAMWORKS API
    """
    _arch = steamworks_util.get_arch()
    _native_supported_platforms = ['linux', 'linux2', 'darwin', 'win32']

    def __init__(self, app_id: int, supported_platforms: list = []) -> None:
        self._supported_platforms = supported_platforms
        self._loaded 	= False
        self._cdll 		= None

        self.app_id 	= app_id

        self._initialize()

    def _integrity_check(self, steamworks_api_path: str, pysteamworks_api_path: str) -> bool:
        with open(steamworks_api_path, 'rb') as fp1:
            steamworks_api_content = fp1.read()
        with open(pysteamworks_api_path, 'rb') as fp2:
            pysteamworks_api_content = fp2.read()
        steamworks_api_content_md5 = hashlib.md5(steamworks_api_content).hexdigest()
        pysteamworks_api_content_md5 = hashlib.md5(pysteamworks_api_content).hexdigest()
        if steamworks_api_content_md5 == 'a05dd414a7a394cae93d8face7572b1d' and pysteamworks_api_content_md5 == 'a6588197d54e1e0c9c93e5a69dc346b5':
            return True
        else:
            return False


    def _initialize(self) -> bool:
        """Initialize module by loading STEAMWORKS library

        :return: bool
        """
        platform = sys.platform
        if self._supported_platforms and platform not in self._supported_platforms:
            raise UnsupportedPlatformException(f'"{platform}" has been excluded')

        if platform not in STEAMWORKS._native_supported_platforms:
            raise UnsupportedPlatformException(f'"{platform}" is not being supported')

        pysteamworks_file_name = ''
        steamworks_file_name = ''
        if platform in ['linux', 'linux2']:
            pysteamworks_file_name = 'SteamworksPy.so'

        elif platform == 'darwin':
            pysteamworks_file_name = 'SteamworksPy.dylib'

        elif platform == 'win32':
            pysteamworks_file_name = 'SteamworksPy.dll' if STEAMWORKS._arch == Arch.x86 else 'SteamworksPy64.dll'
            steamworks_file_name = 'steam_api.dll' if STEAMWORKS._arch == Arch.x86 else 'steam_api64.dll'
        else:
            # This case is theoretically unreachable
            raise UnsupportedPlatformException(f'"{platform}" is not being supported')

        pysteamworks_path = os.path.join(dname, pysteamworks_file_name)
        steamworks_path = os.path.join(os.path.dirname(dname), steamworks_file_name)  # at app root dir
        if not os.path.exists(steamworks_path):
            steamworks_path = os.path.join(dname, steamworks_file_name)  # at steamworks dir

        os.chdir(dname)
        os.environ['LD_LIBRARY_PATH'] = os.getcwd()
        if not os.path.isfile(pysteamworks_path) or not os.path.isfile(steamworks_path):
            raise MissingSteamworksLibraryException(f'Missing library at {pysteamworks_path}')

        app_id_file = os.path.join(os.getcwd(), 'steam_appid.txt')
        if os.path.isfile(app_id_file):
            # check steam_appid in case fucker wants to shit in
            with open(app_id_file, 'r') as f:
                _app_id = int(f.read())
            if _app_id != self.app_id:
                raise SteamNotLoadedException(f'STEAM APPID Does not match SVFI when started')

        if not self._integrity_check(steamworks_path, pysteamworks_path):
            raise SteamNotLoadedException(f'STEAM API FILE HAS BEEN MODIFIED, PLS CHECK SOFTWARE INTEGRITY')

        self._cdll 		= CDLL(pysteamworks_path) # Throw native exception in case of error
        self._loaded 	= True

        self._load_steamworks_api()
        # os.chdir(last_cwd)
        return self._loaded


    def _load_steamworks_api(self) -> None:
        """Load all methods from steamworks api and assign their correct arg/res_len types based on method map

        :return: None
        """
        if not self._loaded:
            raise SteamNotLoadedException('STEAMWORKS not yet loaded')

        for method_name, attributes in STEAMWORKS_METHODS.items():
            f = getattr(self._cdll, method_name)

            if 'restype' in attributes:
                f.restype = attributes['restype']

            if 'argtypes' in attributes:
                f.argtypes = attributes['argtypes']

            setattr(self, method_name, f)

        self._reload_steamworks_interfaces()


    def _reload_steamworks_interfaces(self) -> None:
        """Reload all interface classes

        :return: None
        """
        self.Apps           = SteamApps(self)
        self.Friends        = SteamFriends(self)
        self.Matchmaking    = SteamMatchmaking(self)
        self.Music          = SteamMusic(self)
        self.Screenshots    = SteamScreenshots(self)
        self.Users          = SteamUsers(self)
        self.UserStats      = SteamUserStats(self)
        self.Utils          = SteamUtils(self)
        self.Workshop       = SteamWorkshop(self)


    def initialize(self) -> bool:
        """Initialize Steam API connection

        :return: bool
        """
        if not self.loaded():
            raise SteamNotLoadedException('STEAMWORKS not yet loaded')

        if not self.IsSteamRunning():
            raise SteamNotRunningException('Steam is not running')

        # Boot up the Steam API
        result = self._cdll.SteamInit()
        if result == 2:
            raise SteamNotRunningException('Steam is not running')

        elif result == 3:
            raise SteamConnectionException('Not logged on or connection to Steam client could not be established')

        elif result != 0:
            raise GenericSteamException('Failed to initialize STEAMWORKS API')

        return True

    def relaunch(self, app_id: int) -> bool:
        """

        :param app_id: int
        :return: None
        """
        return self._cdll.RestartAppIfNecessary()

    def unload(self) -> None:
        """Shuts down the Steamworks API, releases pointers and frees memory.

        :return: None
        """
        self._cdll.SteamShutdown()
        self._loaded    = False
        self._cdll      = None


    def loaded(self) -> bool:
        """Is library loaded and everything populated

        :return: bool
        """
        return (self._loaded and self._cdll)


    def run_callbacks(self) -> bool:
        """Execute all callbacks

        :return: bool
        """
        if not self.loaded():
            raise SteamNotLoadedException('STEAMWORKS not yet loaded')

        self._cdll.RunCallbacks()
        return True

    def run_forever(self, base_interval: float = 1.0) -> None:
        """Loop and call Steam.run_callbacks in specified interval

        :param base_interval: float
        :return: None
        """
        while True:
            self.run_callbacks()
            time.sleep(base_interval)
