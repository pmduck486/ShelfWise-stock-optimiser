"""One place that builds the CBC solver, so the warm start works on every OS.

Why: PuLP writes the warm-start file (.mst) to an absolute temp path. On
Windows, CBC puts ".\\" in front of that path, cannot open the file and
silently starts from scratch (found 2026-10-31 on the team laptop: all three
full test days fell back to the start plan). A RELATIVE scratch folder avoids
that, and PuLP still deletes its files after each solve. Each solve uses a
random file name, so parallel DC solves and parallel runs cannot clash.
"""

import os
from pathlib import Path

import pulp

SCRATCH_DIR = '.cbc_scratch'   # created under the current working directory


def cbc_solver(**kwargs):
    solver = pulp.PULP_CBC_CMD(**kwargs)
    if kwargs.get('warmStart') and os.name == 'nt':
        Path(SCRATCH_DIR).mkdir(exist_ok=True)
        solver.tmpDir = SCRATCH_DIR          # relative on purpose (see above)
    return solver
