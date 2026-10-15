"""Check the public export without loading competition data or training models."""
import ast
import hashlib
import json
from pathlib import Path
import re

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    counts = {'python_files': 0, 'notebooks': 0, 'local_links': 0, 'source_hashes': 0}
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
        counts['python_files'] += 1
    for path in ROOT.rglob('*.ipynb'):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if cell.outputs or cell.execution_count is not None:
                    raise ValueError(f'Notebook outputs not cleared: {path}')
        counts['notebooks'] += 1
    for path in ROOT.rglob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            target = target.split('#')[0]
            if target and not (path.parent / target).exists():
                raise ValueError(f'Broken link in {path}: {target}')
            counts['local_links'] += 1
    for item in json.loads((ROOT / 'reports/source_manifest.json').read_text()):
        digest = hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest()
        if digest != item['published_sha256']:
            raise ValueError(f"Source manifest mismatch: {item['path']}")
        counts['source_hashes'] += 1
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
