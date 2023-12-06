import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
from sgfmill import boards, sgf
from tqdm.auto import tqdm
