from ctypes import *


class FindLeaderboardResult_t(Structure):
    """ Represents the STEAMWORKS LeaderboardFindResult_t call result type """
    _fields_ = [
        ("leaderboardHandle", c_uint64),
        ("leaderboardFound", c_uint32)
    ]


class CreateItemResult_t(Structure):
    _fields_ = [
        ("result", c_int),
        ("publishedFileId", c_uint64),
        ("userNeedsToAcceptWorkshopLegalAgreement", c_bool)
    ]


class SubmitItemUpdateResult_t(Structure):
    _fields_ = [
        ("result", c_int),
        ("userNeedsToAcceptWorkshopLegalAgreement", c_bool)
    ]


class ItemInstalled_t(Structure):
    _fields_ = [
        ("appId", c_uint32),
        ("publishedFileId", c_uint64)
    ]


class RemoteStorageSubscribePublishedFileResult_t(Structure):
    _fields_ = [
        ("result", c_int),
        ("publishedFileId", c_uint64)
    ]


class RemoteStorageUnsubscribePublishedFileResult_t(Structure):
    _fields_ = [
        ("result", c_int),
        ("publishedFileId", c_uint64)
    ]