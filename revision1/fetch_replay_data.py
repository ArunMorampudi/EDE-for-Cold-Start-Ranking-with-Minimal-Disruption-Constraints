"""Download a reproducible prefix of the labeled Yandex training stream.

The upstream gzip file is 5.6 GB. Stop after 200,000 complete sessions rather
than downloading the whole corpus; store decoded original lines unchanged.
Credentials are read only for the Kaggle request and never written to outputs.
"""
import gzip, io, json, hashlib, zipfile, os
from pathlib import Path
import requests

out=Path('revision1/data'); out.mkdir(exist_ok=True)
key=json.loads((Path(os.environ.get('KAGGLE_CONFIG_DIR',str(Path.home()/'.kaggle')))/'kaggle.json').read_text())
url='https://www.kaggle.com/api/v1/competitions/data/download/yandex-personalized-web-search-challenge/train.gz'
with requests.get(url,auth=(key['username'],key['key']),stream=True,timeout=120) as response:
    response.raise_for_status()
    raw=gzip.GzipFile(fileobj=response.raw)
    count=0; sid=None; lines=0; digest=hashlib.sha256()
    with (out/'train_first_200000_sessions.txt').open('wb') as dest:
        for line in raw:
            parts=line.split(b'\t',2)
            if parts[0]!=sid:
                if count==200000: break
                sid=parts[0]; count+=1
            dest.write(line); digest.update(line); lines+=1
            if lines%500000==0: print('Lines',lines,'sessions',count,flush=True)
    (out/'source.json').write_text(json.dumps(dict(source=url,sessions=count,lines=lines,sha256=digest.hexdigest(),
        selection='First 200000 contiguous sessions from train.gz, complete through final session; no filtering on outcomes'),indent=2))
print('Training prefix saved',flush=True)
