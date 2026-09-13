"""Exception classes for MusicMining"""


class MusicMiningError(Exception):
    """MusicMining Base Error"""

    pass


class ParameterError(MusicMiningError):
    """Error raised for invalid parameters"""

    pass
