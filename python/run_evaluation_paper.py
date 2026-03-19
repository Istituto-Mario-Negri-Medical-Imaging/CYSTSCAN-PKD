"""run_evaluation_paper.py
Batch evaluation on the Evaluation set using the published parameters.

This script can be run without prior optimisation.
Results are written to results/evaluation_paper/.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _evaluation_core import run_evaluation_core

# Published parameters (identical to those in sensitivity_analysis.m)
PUBLISHED_PARAMS = np.array([
    4.27269938837114,
    18.2632731580751,
    42144.3492985436,
    14.9647743085524,
    0.111299008136904,
    0.918315428393029,
])


def main():
    repo_root = Path(__file__).resolve().parent.parent

    print('Using published parameters:')
    print(f'  {PUBLISHED_PARAMS.tolist()}\n')

    run_evaluation_core(
        params=PUBLISHED_PARAMS,
        output_dir=repo_root / 'results' / 'evaluation_paper',
        label='Published parameters',
    )


if __name__ == '__main__':
    main()
