"""run_evaluation.py
Batch evaluation on the Evaluation set using GA-generated parameters.

Reads results/optimization/BestParams.txt produced by run_optimization.py.
To use the published parameters instead, run run_evaluation_paper.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _evaluation_core import run_evaluation_core


def main():
    repo_root  = Path(__file__).resolve().parent.parent
    params_file = repo_root / 'results' / 'optimization' / 'BestParams.txt'

    if not params_file.is_file():
        sys.exit(
            f'BestParams.txt not found:\n  {params_file}\n'
            'Run run_optimization.py first, or use run_evaluation_paper.py '
            'to evaluate with the published parameters.'
        )

    params = np.loadtxt(params_file)
    print(f'Parameters loaded from: {params_file}')

    run_evaluation_core(
        params=params,
        output_dir=repo_root / 'results' / 'evaluation',
        label='GA-generated parameters',
    )


if __name__ == '__main__':
    main()
