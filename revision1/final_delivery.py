"""Create the two user-facing delivery folders from explicit file lists."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import zipfile

root=Path(__file__).resolve().parents[1]
rev=root/'revision1'
final=rev/'FINAL'
zenodo=final/'01_Zenodo_upload'
peerj=final/'02_PeerJ_upload'
zenodo.mkdir(parents=True,exist_ok=True)
peerj.mkdir(parents=True,exist_ok=True)
uploads=sorted((root/'submitted_peerj_docs').glob('*.docx'))+sorted((root/'submitted_peerj_docs').glob('*.png'))
assert len(uploads)==15
for path in uploads:
    shutil.copy2(path,peerj/path.name)
assert len(list(peerj.iterdir()))==15

files=[p for folder in ('src','tests') for p in (root/folder).rglob('*.py')]
files += [root/'LICENSE',rev/'requirements-revision.txt',rev/'fetch_replay_data.py']
files += list((rev/'results').glob('*.csv'))
files += [rev/p for p in ('results/protocol.json','results/replay/summary.csv','results/replay/manifest.json','data/source.json')]
archive_path=zenodo/'EDE_revision_one_code_and_results.zip'
readme='''# EDE revision one code and numerical results

These corrected results supersede the original simulation and replay estimates.
The archive contains code, tests, dependencies, per-seed results and source-selection
metadata. Raw Yandex logs and peer-review correspondence are excluded.

Run these commands from this archive root with Python 3.12:

```
python -m pip install -r revision1/requirements-revision.txt
python -m pytest tests -q
python -m src.ede_sanity.revision_experiments --workers 4
python revision1/fetch_replay_data.py
python -m src.ede_real_replay.revision_replay --log revision1/data/train_first_200000_sessions.txt
python revision1/build_figures_tables.py
```

The acquisition script requires your own Kaggle credentials and access to the
Yandex Personalized Web Search Challenge. The saved source manifest specifies the
first 200,000 complete training sessions and their SHA-256 hash. These yield
372,282 separate result pages; this is not the historical 61,105-session sample.

Validation uses seeds 100–109; held-out tests use 1000–1019. Lambda is selected
on validation data and frozen for the test conditions. The 100 validation and
3,900 test runs, full summary statistics and protocol are included.

The corrected policy preserves the original prefix before novelty gating and
retains the same penalized base score in both policies. The replay maps clicks
by page and document and updates only actual logged events. Set/click preservation
is structural on ten logged candidates, not evidence of causal engagement or
real-world cold-start discovery. Comparators include explicitly defined heuristics,
not exact reproductions of full published safe-ranking algorithms.

Figure/table regeneration uses saved result CSVs and writes separate files to
submitted_peerj_docs. It does not regenerate or rewrite the manuscript.
'''
# Reuse the validated figure/table functions, omitting the superseded manuscript builder.
builder=(rev/'build_submission.py').read_text(encoding='utf-8')
lines=builder.splitlines(keepends=True)
removed=set()
for node in ast.parse(builder).body:
    if isinstance(node,ast.FunctionDef) and node.name=='manuscript' or isinstance(node,ast.If):
        removed.update(range(node.lineno-1,node.end_lineno))
builder=''.join(line for i,line in enumerate(lines) if i not in removed)
builder=builder.replace('Section 3.5','Section 3.4')
builder+='\nif __name__ == "__main__":\n    figures()\n    tables()\n'
entries={}
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(files)):
        name=path.relative_to(root).as_posix()
        data=path.read_bytes()
        archive.writestr(name,data)
        entries[name]=hashlib.sha256(data).hexdigest()
    for name,data in [('README.md',readme.encode()),('revision1/build_figures_tables.py',builder.encode())]:
        archive.writestr(name,data)
        entries[name]=hashlib.sha256(data).hexdigest()
    archive.writestr('SHA256.json',json.dumps(entries,indent=2))
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    for name,digest in entries.items():
        assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    assert not any('kaggle.json' in n or 'per_page.csv' in n or 'feedback' in n for n in archive.namelist())
assert len(list(zenodo.iterdir()))==1
manifest={str(p.relative_to(final)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [zenodo,peerj] for p in folder.iterdir()}
(rev/'work/final_delivery_manifest.json').write_text(json.dumps(manifest,indent=2))
print('FINAL contains exactly two folders: 1 Zenodo ZIP and 15 PeerJ upload/reference files.')
