# -*- coding: utf-8 -*-
import random
import secrets

import numpy as np


def make_rngs(seed=None):
    """
    seed=None -> криптостойкие источники (SystemRandom + энтропия ОС):
                 перестановку нельзя восстановить по seed.
    seed=int  -> воспроизводимый режим (только отладка).
    Глобальные random / np.random не используются.
    """
    if seed is None:
        return secrets.SystemRandom(), np.random.default_rng()
    return random.Random(seed), np.random.default_rng(seed)
