import base64
import os
import pickle
from binascii import b2a_hex, a2b_hex

import wmi
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5, AES
from Crypto.PublicKey import RSA

import steamworks
from Utils.StaticParameters import appDir
from Utils.utils import ArgumentManager
from steamworks import GenericSteamException


class RSACipher(object):
    private_pem = None
    public_pem = None

    def __init__(self):
        self.private_pem = b""
        self.public_pem = b""

    def get_public_key(self):
        return self.public_pem

    def get_private_key(self):
        return self.private_pem

    def decrypt_with_private_key(self, _cipher_text):
        try:
            _rsa_key = RSA.importKey(self.private_pem)
            _cipher = Cipher_pkcs1_v1_5.new(_rsa_key)
            _text = _cipher.decrypt(base64.b64decode(_cipher_text), "ERROR")
            return _text.decode(encoding="utf-8")
        except:
            return ""

    def encrypt_with_public_key(self, _text):
        _rsa_key = RSA.importKey(self.public_pem)
        _cipher = Cipher_pkcs1_v1_5.new(_rsa_key)
        _cipher_text = base64.b64encode(_cipher.encrypt(_text.encode(encoding="utf-8")))
        return _cipher_text

    # encrypt with private key & decrypt with public key is not allowed in Python
    # although it is allowed in RSA
    def encrypt_with_private_key(self, _text):
        _rsa_key = RSA.importKey(self.private_pem)
        _cipher = Cipher_pkcs1_v1_5.new(_rsa_key)
        _cipher_text = base64.b64encode(_cipher.encrypt(_text.encode(encoding="utf-8")))
        return _cipher_text

    def decrypt_with_public_key(self, _cipher_text):
        _rsa_key = RSA.importKey(self.public_pem)
        _cipher = Cipher_pkcs1_v1_5.new(_rsa_key)
        _text = _cipher.decrypt(base64.b64decode(_cipher_text), "ERROR")
        return _text.decode(encoding="utf-8")


class AESCipher(object):
    def __init__(self):
        self.key = ''.encode('utf-8')
        self.mode = AES.MODE_CBC
        self.iv = b''  # TODO Notion default

    @staticmethod
    def _add_to_16(text: bytes):
        """
        如果text不足16位的倍数就用空格补足为16位
        :param text:
        :return:
        """
        if len(text) % 16:
            add = 16 - (len(text) % 16)
        else:
            add = 0
        text = text + (b'\0' * add)
        return text

    def _encrypt(self, text: bytes):
        """
        加密函数
        :return:
        """
        text = self._add_to_16(text)
        cryptos = AES.new(self.key, self.mode, self.iv)
        cipher_text = cryptos.encrypt(text)
        # 因为AES加密后的字符串不一定是ascii字符集的，输出保存可能存在问题，所以这里转为16进制字符串
        return b2a_hex(cipher_text)

    def _decrypt(self, text):
        """
        解密后，去掉补足的空格用strip() 去掉
        :return:
        """
        cryptos = AES.new(self.key, self.mode, self.iv)
        plain_text = cryptos.decrypt(a2b_hex(text))
        return plain_text


class ValidationBase:
    def __init__(self, logger):
        self.logger = logger
        self._is_validate_start = False
        self._validate_error = None
        pass

    def CheckValidateStart(self):
        return self._is_validate_start

    def CheckProDLC(self, pro_dlc_id: int):
        pass

    def GetStat(self, key: str, key_type: type):
        pass

    def GetAchv(self, key: str):
        pass

    def SetStat(self, key: str, value):
        pass

    def SetAchv(self, key: str, clear=False):
        pass

    def Store(self):
        pass

    def GetValidateError(self):
        return self._validate_error


class RetailValidation(ValidationBase):
    def __init__(self, logger):
        """
        Whether use steam for validation
        """
        super().__init__(logger)
        original_cwd = os.getcwd()
        self._rsa_worker = RSACipher()
        self._bin_path = os.path.join(appDir, 'license.dat')
        try:
            self._is_validate_start = self._regist()  # This method has to be called in order for the wrapper to become functional!
        except Exception as e:
            self._is_validate_start = False  # debug
            self._validate_error = e
            self.logger.error('Failed to initiate Retail License. Shutting down.')
            return
        os.chdir(original_cwd)

    def CheckValidateStart(self):
        return self._is_validate_start

    def _GetCVolumeSerialNumber(self):
        c = wmi.WMI()
        for physical_disk in c.Win32_DiskDrive():
            return physical_disk.SerialNumber
        else:
            return 0

    def _GenerateRegisterBin(self):
        bin_data = {'license_data': self._rsa_worker.encrypt_with_private_key(self._GetCVolumeSerialNumber())}
        pickle.dump(bin_data, open(self._bin_path, 'wb'))

    def _ReadRegisterBin(self):
        if not os.path.exists(self._bin_path):
            self._GenerateRegisterBin()
            raise OSError("Could not find License File. The system had generated a .dat file at the root dir "
                          "of this app for license, please send this to administrator "
                          "and replace it with the one that was sent you")
        bin_data = pickle.load(open(self._bin_path, 'rb'))
        assert type(bin_data) is dict, "Type of License Data is not correct, " \
                                       "please consult administrator for further support"
        license_key = bin_data.get('license_key', "")
        return license_key

    def _regist(self):
        license_key = self._ReadRegisterBin()
        volume_serial = self._GetCVolumeSerialNumber()
        key_decrypted = self._rsa_worker.decrypt_with_private_key(license_key)
        if volume_serial != key_decrypted:
            self._GenerateRegisterBin()
            raise OSError("Wrong Register code, please check your license with your administrator")
        elif volume_serial == key_decrypted:
            return True

    def CheckProDLC(self, pro_dlc_id: int):
        """All DLC Purchased as default"""
        return True

    def GetStat(self, key: str, key_type: type):
        return False

    def GetAchv(self, key: str):
        return False

    def SetStat(self, key: str, value):
        return

    def SetAchv(self, key: str, clear=False):
        return

    def Store(self):
        return False


class SteamValidation(ValidationBase):
    def __init__(self, logger):
        """
        Whether use steam for validation
        """
        super().__init__(logger)
        original_cwd = os.getcwd()
        self.steamworks = None
        self.steamworks = steamworks.STEAMWORKS(ArgumentManager.app_id)
        try:
            self.steamworks.initialize()  # This method has to be called in order for the wrapper to become functional!
        except:
            self._is_validate_start = False
            self._validate_error = GenericSteamException(
                'Failed to Load Steam Status, Please Make Sure this game is purchased')
            self.logger.error('Failed to initiate Steam API. Shutting down.')
            return
        self._is_validate_start = True
        if self.steamworks.UserStats.RequestCurrentStats():
            self.logger.info('Steam Stats successfully retrieved!')
        else:
            self._is_validate_start = False
            self._validate_error = GenericSteamException('Failed to get Stats Error, Please Make Sure Steam is On')
            self.logger.error('Failed to get Steam stats. Shutting down.')
        os.chdir(original_cwd)

    def _CheckPurchaseStatus(self):
        steam_64id = self.steamworks.Users.GetSteamID()
        valid_response = self.steamworks.Users.GetAuthSessionTicket()
        self.logger.debug(f'Steam User Logged on as {steam_64id}, auth: {valid_response}')
        if valid_response != 0:  # Abnormal Purchase
            self._is_validate_start = False
            self._validate_error = GenericSteamException("Abnormal Start, Please Check Software's Purchase Status, "
                                                         f"Response: {valid_response}")

    def CheckValidateStart(self):
        return self._is_validate_start

    def CheckProDLC(self, dlc_id: int) -> bool:
        """

        :param dlc_id: DLC for SVFI, start from 0
        0: Pro
        :return:
        """
        purchase_pro = self.steamworks.Apps.IsDLCInstalled(ArgumentManager.pro_dlc_id[dlc_id])
        self.logger.info(f'Steam User Purchase Pro DLC Status: {purchase_pro}')
        return purchase_pro

    def GetStat(self, key: str, key_type: type):
        if key_type is int:
            return self.steamworks.UserStats.GetStatInt(key)
        elif key_type is float:
            return self.steamworks.UserStats.GetStatFloat(key)

    def GetAchv(self, key: str):
        return self.steamworks.UserStats.GetAchievement(key)

    def SetStat(self, key: str, value):
        return self.steamworks.UserStats.SetStat(key, value)

    def SetAchv(self, key: str, clear=False):
        if clear:
            return self.steamworks.UserStats.ClearAchievement(key)
        return self.steamworks.UserStats.SetAchievement(key)

    def Store(self):
        return self.steamworks.UserStats.StoreStats()


class EULAWriter:
    eula_hi = """
    EULA
    
    重要须知——请仔细阅读：请确保仔细阅读并理解《最终用户许可协议》（简称“协议”）中描述的所有权利与限制。
    
    协议
    本协议是您与SDT Core及其附属公司（简称“公司”）之间达成的协议。仅在您接受本协议中包含的所有条件的情况下，您方可使用软件及任何附属印刷材料。
    安装或使用软件即表明，您同意接受本《协议》各项条款的约束。如果您不同意本《协议》中的条款：(i)请勿安装软件, (ii)如果您已经购买软件，请立即凭购买凭证将其退回购买处，并获得退款。
    在您安装软件时，会被要求预览并通过点击“我接受”按钮决定接受或不接受本《协议》的所有条款。点击“我接受”按钮，即表明您承认已经阅读过本《协议》，并且理解并同意受其条款与条件的约束。
    版权
    软件受版权法、国际协约条例以及其他知识产权法和条例的保护。软件（包括但不限于软件中含有的任何图片、照片、动画、视频、音乐、文字和小型应用程序）及其附属于软件的任何印刷材料的版权均由公司及其许可者拥有。
    
    许可证的授予
    软件的授权与使用须遵从本《协议》。公司授予您有限的、个人的、非独占的许可证，允许您使用软件，并且以将其安装在您的手机上为唯一目的。公司保留一切未在本《协议》中授予您的权利。
    
    授权使用
    1. 如果软件配置为在一个硬盘驱动器上运行，您可以将软件安装在单一电脑上，以便在您的手机上安装和使用它。
    2. 您可以制作和保留软件的一个副本用于备份和存档，条件是软件及副本归属于您。
    3. 您可以将您在本《协议》项下的所有权利永久转让，转让的条件是您不得保留副本，转让软件（包括全部组件、媒体、印刷材料及任何升级版本），并且受让人接受本《协议》的各项条款。
    
    限制
    1. 您不得删除或掩盖软件或附属印刷材料注明的版权、商标或其他所有权。
    2. 您不得对软件进行反编译、修改、逆向工程、反汇编或重制。
    3. 您不得复制、租赁、发布、散布或公开展示软件，不得制作软件的衍生产品（除非编辑器和本协议最终用户变更部分或其他附属于软件的文件明确许可），或是以商业目的对软件进行开发。
    4. 您不得通过电子方式或网络将软件从一台电脑、控制台或其他平台传送到另一个上。
    5. 您不得将软件的备份或存档副本用作其他用途，只可在原始副本被损坏或残缺的情况下，用其替换原始副本。
    6. 您不得将软件的输出结果用于商业用途
    
    试用版本
    如果提供给您的软件为试用版，其使用期限或使用数量有限制，您同意在试用期结束后停止使用软件。您知晓并同意软件可能包含用于避免您突破这些限制的代码，并且这些代码会在您删除软件后仍保留在您的电脑上，避免您下载其他副本并重复利用试用期。
    
    编辑器和最终用户变更
    如果软件允许您进行修改或创建新内容（“编辑器”），您可以使用该编辑器修改或优化软件，包括创建新内容（统称“变更”），但必须遵守下列限制。您的变更(i)必须符合已注册的完整版软件；(ii)不得对执行文件进行改动；(iii)不得包含任何诽谤、中伤、违法、损害他人或公众利益的内容；(iv)不得包含任何商标、著作权保护内容或第三方的所有权内容；(v)不得用作商业目的，包括但不限于，出售变更内容、按次计费或分时服务。
    
    终止
    本协议在终止前都是有效的。您可以随时卸载软件来终止该协议。如果您违反了协议的任何条款或条件，本协议将自动终止，恕不另行通知。本协议中涉及到的保证、责任限制和损失赔偿的部分在协议终止后仍然有效。
    
    有限保修及免责条款
    您知道并同意因使用该软件及其记录该软件的媒体所产生的风险由您自行承担。该软件和媒体“照原样”发布。除非有适用法律规定，本公司向此产品的原始购买人保证，在正常使用的情况，该软件媒体存储介质在30天内（自购买之日算起）无缺陷。对于因意外、滥用、疏忽或误用引起的缺陷，该保证无效。如果软件没有达到保证要求，您可能会单方面获得补偿，如果您退回有缺陷的软件，您可以免费获得替换产品。本公司不保证该软件及其操作和功能达到您的要求，也不保证软件的使用不会出现中断或错误。
    在适用法律许可的最大范围下，除了上述的明确保证之外，本公司不做其他任何保证，包括但不限于暗含性的适销保证、特殊用途保证及非侵权保证。除了上述的明确保证之外，本公司不对软件使用和软件使用结果在正确性、准确性、可靠性、通用性和其他方面做出保证、担保或陈述。部分司法管辖区不允许排除或限制暗含性保证，因此上面的例外和限制情况可能对您不适用。
    
    责任范围
    在任何情况下，本公司及其员工和授权商都不对任何由软件使用或无法使用软件而引起的任何附带、间接、特殊、偶然或惩罚性伤害以及其他伤害（包括但不限于对人身或财产的伤害，利益损失，运营中断，商业信息丢失，隐私侵犯，履行职责失败及疏忽）负责，即使公司或公司授权代表已知悉了存在这种伤害的可能性。部分司法管辖区不允许排除附带或间接伤害，因此，上述例外情况可能对您不适用。
    
    在任何情况下，公司承担的和软件伤害相关的费用都不超过您对该软件实际支付的数额。
    
    其他
    如果发现此最终用户许可协议的任意条款或规定违法、无效或因某些原因无法强制执行，该条款和部分将被自动舍弃，不会影响本协议其余规定的有效性和可执行性。
    本协议包含软您和本软件公司之间的所有协议及其使用方法。
    
    eula = true
    """

    def __init__(self):
        self.eula_dir = os.path.join(appDir, 'train_log')
        os.makedirs(self.eula_dir, exist_ok=True)
        self.eula_path = os.path.join(self.eula_dir, 'md5.svfi')

    def boom(self):
        with open(self.eula_path, 'w', encoding='utf-8') as w:
            w.write(self.eula_hi)
