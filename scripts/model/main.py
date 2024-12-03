# %%
import sys
from pathlib import Path

import optimize
import simulate
import utils
import visualize

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query

# %%
