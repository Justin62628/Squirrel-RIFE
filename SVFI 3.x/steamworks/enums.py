from enum import Enum, IntFlag


class Arch(Enum):
    """ Limited list of processor architectures """
    x86 = 0
    x64 = 1


class FriendFlags(Enum):
    """ EFriendFlags """
    NONE                    = 0x00
    BLOCKED                 = 0x01
    FRIENDSHIP_REQUESTED    = 0x02
    IMMEDIATE               = 0x04
    CLAN_MEMBER             = 0x08
    ON_GAME_SERVER          = 0x10
    REQUESTING_FRIENDSHIP   = 0x80
    REQUESTING_INFO         = 0x100
    IGNORED                 = 0x200
    IGNORED_FRIEND          = 0x400
    SUGGESTED               = 0x800
    ALL                     = 0xFFFF


class EWorkshopFileType(Enum):
    COMMUNITY               = 0
    MICRO_TRANSACTION       = 1
    COLLECTION              = 2
    ART                     = 3
    VIDEO                   = 4
    SCREENSHOT              = 5
    GAME                    = 6
    SOFTWARE                = 7
    CONCEPT                 = 8
    WEB_GUIDE               = 9
    INTEGRATED_GUIDE        = 10
    MERCH                   = 11
    CONTROLLER_BINDING      = 12
    STEAMWORKS_ACCESS_INVITE= 13
    STEAM_VIDEO             = 14
    GAME_MANAGED_ITEM       = 15
    MAX                     = 16


class EResult(Enum):
    OK                      = 1
    FAIL                    = 2     # Unknown error
    NO_CONNECTION           = 3
                                    # 4 is absent
    INVALID_PASSWORD        = 5
    LOGGED_IN_ELSEWHERE     = 6
    INVALID_PROTOCOL_VER    = 7
    INVALID_PARAM           = 8
    FILE_NOT_FOUND          = 9
    BUSY                    = 10
    INVALID_STATE           = 11
    INVALID_NAME            = 12
    INVALID_EMAIL           = 13
    DUPLICATE_NAME          = 14
    ACCESS_DENIED           = 15
    TIMEOUT                 = 16
    BANNED                  = 17    # User is VAC banned
    ACCOUNT_NOT_FOUND       = 18
    INVALID_STEAM_ID        = 19
    SERVICE_UNAVAILABLE     = 20
    NOT_LOGGED_ON           = 21
    PENDING                 = 22    # In process or waiting on third party
    INSUFFICIENT_PRIVILEGE  = 24
    LIMIT_EXCEEDED          = 25
    REVOKED                 = 26    # Guest pass access revoked
    EXPIRED                 = 27    # Guest pass access expired
    ALREADY_REDEEMED        = 28    # Guest pass already used
    DUPLICATE_REQUEST       = 29
    ALREADY_OWNED           = 30
    IP_NOT_FOUND            = 31
    PERSIST_FAILED          = 32
    LOCKING_FAILED          = 33
    LOGON_SESSION_REPLACED  = 34
    CONNECT_FAILED          = 35
    HANDSHAKE_FAILED        = 36
    IO_FAILURE              = 37
    REMOTE_DISCONNECT       = 38
    SHOPPING_CART_NOT_FOUND = 39
    BLOCKED                 = 40
    IGNORED                 = 41
    NO_MATCH                = 42
    ACCOUNT_DISABLED        = 43
    SERVICE_READ_ONLY       = 44    # Account is too new to upload content
    ACCOUNT_NOT_FEATURED    = 45
    ADMINISTRATOR_OK        = 46
    CONTENT_VERSION         = 47
    TRY_ANOTHER_CM          = 48
    PASSWORD_REQUIRED_TO_KICK_SESSION = 49
    ALREADY_LOGGED_IN_ELSEWHERE = 50
    SUSPENDED               = 51
    CANCELLED               = 52
    DATA_CORRUPTION         = 53
    DISK_FULL               = 54
    REMOTE_CALL_FAILED      = 55
    PASSWORD_UNSET          = 56
    EXTERNAL_ACCOUNT_UNLINKED = 57
    PSN_TICKET_INVALID      = 58
    EXTERNAL_ACCOUNT_ALREADY_LINKED = 59
    REMOTE_FILE_CONFLICT    = 60
    ILLEGAL_PASSWORD        = 61
    SAME_AS_PREVIOUS_VALUE  = 62
    ACCOUNT_LOGON_DENIED    = 63
    CANNOT_USE_OLD_PASSWORD = 64
    INVALID_LOGIN_AUTH_CODE = 65
    ACCOUNT_LOGON_DENIED_NO_MAIL = 66
    HARDWARE_NOT_CAPABLE_OF_IPT = 67
    IPT_INIT_ERROR          = 68
    PARENTAL_CONTROL_RESTRICTED = 69
    FACEBOOK_QUERY_ERROR    = 70
    EXPIRED_LOGIN_AUTHCODE  = 71
    IP_LOGIN_RESTRICTION_FAILED = 72
    ACCOUNT_LOCKED_DOWN     = 73
    ACCOUNT_LOGON_DENIED_VERIFIED_EMAIL_REQUIRED = 74
    NO_MATCHING_URL         = 75
    BAD_RESPONSE            = 76
    REQUIRE_PASSWORD_REENTRY = 77
    VALUE_OUT_OF_RANGE      = 78
    UNEXPECTED_ERROR        = 79
    DISABLED                = 80
    INVALID_CEG_SUBMISSION  = 81
    RESTRICTED_DEVICE       = 82
    REGION_LOCKED           = 83
    RATE_LIMIT_EXCEEDED     = 84
    ACCOUNT_LOGIN_DENIED_NEED_TWO_FACTOR = 85
    ITEM_DELETED            = 86
    ACCOUNT_LOGIN_DENIED_THROTTLE = 87
    TWO_FACTOR_CODE_MISMATCH = 88
    TWO_FACTOR_ACTIVATION_CODE_MISMATCH = 89
    ACCOUNT_ASSOCIATED_TO_MULTIPLE_PARTNERS = 90
    NOT_MODIFIED            = 91
    NO_MOBILE_DEVICE        = 92
    TIME_NOT_SYNCED         = 93
    SMS_CODE_FAILED         = 94
    ACCOUNT_LIMIT_EXCEEDED  = 95
    ACCOUNT_ACTIVITY_LIMIT_EXCEEDED = 96
    PHONE_ACTIVITY_LIMIT_EXCEEDED = 97
    REFUND_TO_WALLET        = 98
    EMAIL_SEND_FAILURE      = 99
    NOT_SETTLED             = 100
    NEED_CAPTCHA            = 101
    GSLT_DENIED             = 102
    GS_OWNER_DENIED         = 103
    INVALID_ITEM_TYPE       = 104
    IP_BANNED               = 105
    GSLT_EXPIRED            = 106
    INSUFFICIENT_FUNDS      = 107
    TOO_MANY_PENDING        = 108


class EItemState(IntFlag):
    """ EItemState """
    NONE                = 0
    SUBSCRIBED          = 1
    LEGACY_ITEM         = 2
    INSTALLED           = 4
    NEEDS_UPDATE        = 8
    DOWNLOADING         = 16
    DOWNLOAD_PENDING    = 32


class ERemoteStoragePublishedFileVisibility(Enum):
    PUBLIC              = 0
    FRIENDS_ONLY        = 1
    PRIVATE             = 2


class ENotificationPosition(Enum):
    TOP_LEFT        = 0
    TOP_RIGHT       = 1
    BOTTOM_LEFT     = 2
    BOTTOM_RIGHT    = 3


class EGamepadTextInputLineMode(Enum):
    SINGLE_LINE     = 0
    MULTIPLE_LINES  = 1


class EGamepadTextInputMode(Enum):
    NORMAL      = 0
    PASSWORD    = 1


class EItemUpdateStatus(Enum):
    INVALID                 = 0
    PREPARING_CONFIG        = 1
    PREPARING_CONTENT       = 2
    UPLOADING_CONTENT       = 3
    UPLOADING_PREVIEW_FILE  = 4
    COMMITTING_CHANGES      = 5
