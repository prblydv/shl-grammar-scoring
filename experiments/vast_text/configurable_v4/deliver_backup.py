"""Lossless compressed checkpoint delivery in independently verified SSH chunks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import zstandard as zstd
from sweep_utils import file_sha256
from telemetry import atomic_json


class ChunkWriter:
    def __init__(self,directory,limit=64*2**20):
        self.directory=directory
        self.limit=limit
        self.stream=None
        self.size=0
        self.digest=None
        self.chunks=[]
    def writable(self): return True
    def flush(self):
        if self.stream: self.stream.flush()
    def write(self,data):
        original=len(data)
        view=memoryview(data)
        while view:
            if self.stream is None:
                self.name=f'chunk_{len(self.chunks):05d}.zstpart'
                self.stream=(self.directory/self.name).open('wb')
                self.size=0
                self.digest=hashlib.sha256()
            amount=min(len(view),self.limit-self.size)
            self.stream.write(view[:amount])
            self.digest.update(view[:amount])
            self.size+=amount
            view=view[amount:]
            if self.size==self.limit: self.finish_chunk()
        return original
    def finish_chunk(self):
        if self.stream:
            self.stream.close()
            self.chunks.append({'name':self.name,'bytes':self.size,'sha256':self.digest.hexdigest()})
            self.stream=None
    def close(self): self.finish_chunk()


def pack(source,directory):
    source,directory=Path(source).resolve(),Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'backup_manifest.json').read_text())
    names=sorted(manifest['files'])+['backup_manifest.json']
    chunks=ChunkWriter(directory)
    original=0
    with zstd.ZstdCompressor(level=5,threads=4).stream_writer(chunks,closefd=False) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as archive:
            for name in names:
                path=source/name
                if name in manifest['files'] and file_sha256(path)!=manifest['files'][name]:
                    raise ValueError(f'Artifact changed before packaging: {name}')
                archive.add(path,arcname=name,recursive=False)
                original+=path.stat().st_size
                print(f'PACKED {name} ({original/2**30:.2f} GiB original)',flush=True)
    chunks.close()
    atomic_json(directory/'chunks.json',{'chunks':chunks.chunks,'original_bytes':original,
        'compressed_bytes':sum(c['bytes'] for c in chunks.chunks),
        'backup_manifest_sha256':file_sha256(source/'backup_manifest.json')})
    print(f'PACK COMPLETE: {len(chunks.chunks)} chunks',flush=True)


def fetch(remote,directory,output,key,port,workers):
    output=Path(output).resolve()
    cache=output.parent/(output.name+'-transfer')
    cache.mkdir(parents=True,exist_ok=True)
    transport=['scp','-q','-i',str(key),'-P',str(port)]
    subprocess.run(transport+[f'{remote}:{directory}/chunks.json',str(cache/'chunks.json')],check=True)
    manifest=json.loads((cache/'chunks.json').read_text())
    def download(chunk):
        target=cache/chunk['name']
        if target.exists() and file_sha256(target)==chunk['sha256']: return chunk['bytes']
        subprocess.run(transport+[f'{remote}:{directory}/{chunk["name"]}',str(target)],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        if file_sha256(target)!=chunk['sha256']: raise ValueError(f'Chunk checksum failed: {target}')
        return chunk['bytes']
    total=0
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(download,c) for c in manifest['chunks']]
        for finished,future in enumerate(as_completed(futures),1):
            total+=future.result()
            print(f'DOWNLOAD {finished}/{len(futures)} chunks | {total/2**30:.2f}/{manifest["compressed_bytes"]/2**30:.2f} GiB | {total/max(.01,time.monotonic()-started)/2**20:.2f} MiB/s',flush=True)
    archive=cache/'backup.tar.zst'
    with archive.open('wb') as stream:
        for chunk in manifest['chunks']:
            with (cache/chunk['name']).open('rb') as part: shutil.copyfileobj(part,stream,8*2**20)
    output.mkdir(parents=True,exist_ok=True)
    with archive.open('rb') as stream,zstd.ZstdDecompressor().stream_reader(stream) as decoded:
        with tarfile.open(fileobj=decoded,mode='r|') as contents:
            for entry in contents:
                target=(output/entry.name).resolve()
                if output not in target.parents or not entry.isfile():
                    raise ValueError(f'Unexpected archive member: {entry.name}')
                target.parent.mkdir(parents=True,exist_ok=True)
                with contents.extractfile(entry) as source,target.open('wb') as destination:
                    shutil.copyfileobj(source,destination,8*2**20)
                print(f'EXTRACTED {entry.name}',flush=True)
    if file_sha256(output/'backup_manifest.json')!=manifest['backup_manifest_sha256']:
        raise ValueError('Backup manifest differs after extraction')
    from verify_backup import verify
    verify(output)
    # Delete only delivery cache files listed in the checked chunk manifest after full verification.
    for chunk in manifest['chunks']: (cache/chunk['name']).unlink()
    archive.unlink()
    atomic_json(cache/'delivery_receipt.json',{'verified':True,'output':str(output),
        'compressed_bytes':manifest['compressed_bytes'],'original_bytes':manifest['original_bytes']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['pack','fetch'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--remote',default='user@example.invalid')
    parser.add_argument('--port',type=int,default=26153)
    parser.add_argument('--key',type=Path,default=Path.home()/'.ssh/vast_ai_ed25519')
    parser.add_argument('--output',type=Path,default=Path('artifacts/vast_deberta_large_first'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='pack':
        if args.source is None: parser.error('--source required')
        pack(args.source,args.directory)
    else: fetch(args.remote,args.directory,args.output,args.key,args.port,args.workers)


if __name__=='__main__': main()
