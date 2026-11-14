"""Explicit, checksummed downloads; no automatic execution of upstream code."""
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath


def fetch(spec,destination):
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        h=hashlib.sha256(dest.read_bytes()).hexdigest()
        if h==spec['sha256']:return str(dest)
        raise ValueError('Existing file has a different checksum; choose another destination')
    if not spec['url'].startswith('https://'):raise ValueError('HTTPS required')
    temp=dest.with_suffix(dest.suffix+'.partial'); h=hashlib.sha256();size=0
    try:
        with urllib.request.urlopen(spec['url'],timeout=60) as response,temp.open('wb') as out:
            while chunk:=response.read(1024*1024):
                size+=len(chunk)
                if size>spec['bytes']:raise ValueError('Download larger than pinned artifact')
                out.write(chunk);h.update(chunk)
        if size!=spec['bytes'] or h.hexdigest()!=spec['sha256']:
            raise ValueError('Checksum/size mismatch: upstream latest may have changed')
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
    return str(dest)


def extract_scifact(archive,destination):
    dest=Path(destination)
    allowed={'data/claims.jsonl','data/claims_metadata.jsonl','data/corpus_candidates.jsonl','prediction/model_predictions.parqet'}
    found=[]
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            name=PurePosixPath(m.name)
            if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe archive path')
            if m.name not in allowed:continue
            if not m.isfile() or m.size>50_000_000:raise ValueError('Unexpected archive member')
            output=dest/m.name
            if output.exists():raise FileExistsError(output)
            output.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as source,output.open('wb') as out:shutil.copyfileobj(source,out)
            found.append(m.name)
    if set(found)!=allowed:raise ValueError('Incomplete SciFact-Open release')
    return found
