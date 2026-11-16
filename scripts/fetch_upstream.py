"""Retrieve upstream source/data from the original authors at a pinned commit."""
import argparse
import subprocess
from pathlib import Path

COMMIT = '633bbd7294e5b67ae537b88c5f75eb8c53abd454'
URL = 'https://github.com/bmil-jnu/OS-Carcinogenicity.git'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('upstream'))
    a = p.parse_args()
    if a.output.exists():
        head = subprocess.check_output(['git', '-C', str(a.output), 'rev-parse', 'HEAD'], text=True).strip()
        if head != COMMIT:
            raise ValueError('Existing directory is not at the pinned upstream commit; use a new output directory.')
    else:
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', URL, str(a.output)], check=True)
        subprocess.run(['git', '-C', str(a.output), 'checkout', '--detach', COMMIT], check=True)
    print(f'Upstream ready: {a.output} at {COMMIT}')
