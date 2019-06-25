from enum import Enum


class EExecutionType(Enum):
    TRAIN = 0
    TEST = 1
    BOTH = 2
    VIZ = 3