"""Verify all sealed run artifacts after an independent download."""
import argparse
import json
from pathlib import Path
from sweep_utils import file_sha256
from telemetry import atomic_json


def verify(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'backup_manifest.json').read_text())
    total=0
    for name,digest in manifest['files'].items():
        path=(directory/name).resolve()
        if directory not in path.parents:
            raise ValueError('Invalid manifest path')
        if not path.is_file() or file_sha256(path)!=digest:
            raise ValueError(f'Missing/changed backup artifact: {name}')
        total+=path.stat().st_size
    receipt={'verified':True,'files':len(manifest['files']),'bytes':total,
             'manifest_sha256':file_sha256(directory/'backup_manifest.json'),
             'destination':str(directory),'independent_local_backup':True}
    atomic_json(directory/'local_backup_receipt.json',receipt)
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    verify(parser.parse_args().directory)
