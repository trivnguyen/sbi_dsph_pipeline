"""Simulate training data for any registered model spec.

    python simulate_batch.py --model 8p_ZhaoPlumCOM_v4 \
        --n-sims 100000 --output-dir /scratch/.../8p_ZhaoPlumCOM_v5

    python simulate_batch.py --list

Through SLURM (extra args are forwarded):

    sbatch ../../npe/slurm/simulate.sbatch \
        ../dsph_sims/scripts/simulate_batch.py <output_dir> <n_sims> \
        --model 8p_ZhaoPlumCOM_v4
"""

import sys

from dsph_sims import SPECS, get_spec
from dsph_sims.runner import main

if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--list' in argv:
        for spec in SPECS.values():
            print(f'{spec.name:28s} {spec.description}')
            print(f'{"":28s} datasets: {", ".join(spec.datasets) or "-"}')
        sys.exit(0)
    if '--model' not in argv:
        sys.exit('usage: simulate_batch.py --model <spec> [runner args]; '
                 '--list shows the specs')
    i = argv.index('--model')
    spec = get_spec(argv[i + 1])
    main(spec, argv[:i] + argv[i + 2:])
