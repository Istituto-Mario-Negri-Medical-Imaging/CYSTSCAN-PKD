"""run_evaluation_paper.py
Batch evaluation on the Evaluation set using the published parameters.

This script can be run without prior optimisation.
Results are written to results/evaluation_paper/.
"""

from pathlib import Path

from cystscan.counting import PUBLISHED_PARAMS
from cystscan.evaluation import run_evaluation_core


def main():
    repo_root = Path(__file__).resolve().parent.parent

    print('Using published parameters:')
    print(f'  {PUBLISHED_PARAMS.tolist()}\n')

    run_evaluation_core(
        params=PUBLISHED_PARAMS,
        output_dir=repo_root / 'results' / 'evaluation_paper',
        label='Published parameters',
        data_dir=repo_root / 'data' / '2_Cyst_counting_pipeline' / 'Evaluation set',
    )


if __name__ == '__main__':
    main()
