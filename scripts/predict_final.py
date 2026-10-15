"""Rebuild the archived experimental blend from private, ordered prediction arrays.

The two inventories are deliberately separate: the retained joint uses 12 runs,
while the later hurdle uses 14. No test labels or filename-derived rules enter
either prediction function. This fits saved hyperparameters, not a new search.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research_hurdle_rmse_20261015 import predict as hurdle_predict
from optimize_all_metadata_ensembles_20261015 import raw_prediction as joint_predict

ROOT = Path(__file__).resolve().parents[1]


def load_inputs(path, width):
    """Validate each split independently; a filename may occur in both splits."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ('x', 'test', 'y', 'names', 'test_names')}
    if data['x'].shape != (769, width) or data['test'].shape != (216, width):
        raise ValueError(f'Expected 769/216 rows and {width} ordered model columns')
    if data['y'].shape != (769,):
        raise ValueError('Training labels must be a length-769 vector')
    for key, count in [('names', 769), ('test_names', 216)]:
        if data[key].shape != (count,) or len(set(data[key])) != count:
            raise ValueError(f'Invalid or duplicate {key}')
    for key in ('x', 'test', 'y'):
        if not np.isfinite(data[key]).all() or not ((data[key] >= 0) & (data[key] <= 5)).all():
            raise ValueError(f'Nonfinite or out-of-range {key}')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hurdle-inputs', type=Path, required=True)
    parser.add_argument('--joint-inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Optional private archived CSV for exact verification')
    args = parser.parse_args()
    hurdle, joint = load_inputs(args.hurdle_inputs, 14), load_inputs(args.joint_inputs, 12)
    for key in ('names', 'test_names', 'y'):
        if not np.array_equal(hurdle[key], joint[key]):
            raise ValueError(f'The two inventories disagree on {key}')
    hfit = json.loads((ROOT / 'reports/hurdle_fits.json').read_text())['hurdle_regression']['full_fit']['selected']
    jfit = json.loads((ROOT / 'reports/joint_fits.json').read_text())['full_fits']['joint_quality']['selected']['params']
    with threadpool_limits(limits=1):
        h = hurdle_predict(hurdle['x'], hurdle['y'], hurdle['test'], hfit)
        j = joint_predict(joint['x'], joint['y'], joint['test'], jfit)
    # Fixed, public-informed historical recipe; not claimed to be independently optimal.
    prediction = 0.60 * h + 0.40 * j
    if not np.isfinite(prediction).all() or not ((prediction >= 0) & (prediction <= 5)).all():
        raise ValueError('Invalid final predictions')
    frame = pd.DataFrame({'filename': hurdle['test_names'], 'label': prediction})
    if args.reference:
        ref = pd.read_csv(args.reference)
        if ref.filename.tolist() != frame.filename.tolist():
            raise ValueError('Reference test ordering differs')
        np.testing.assert_allclose(ref.label, prediction, atol=1e-10, rtol=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    receipt = {'rows': len(frame), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
               'recipe': {'hurdle': 0.6, 'prior_joint_quality': 0.4},
               'note': 'Floating-point/platform differences can change CSV bytes; use numeric reference verification.'}
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
