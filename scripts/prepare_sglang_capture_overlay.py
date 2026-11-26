#!/usr/bin/env python3
"""Build a verified training-only capture view without editing pinned SGLang.

Unmodified files are symlinks to the selected source. Only patch targets are
copied, and git apply is constrained to that new view. No dependencies or git
revisions are changed. The JSON manifest records both base and actual overrides.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def prepare(base, cache, patch_dir):
    base=base.resolve();cache=cache.resolve();patch_dir=patch_dir.resolve()
    expected=json.loads((patch_dir/'base.json').read_text())
    patch=patch_dir/'spec-capture.patch'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip()
    diff_sha=hashlib.sha256(subprocess.check_output(['git','diff'],cwd=base)).hexdigest()
    if head!=expected['commit'] or diff_sha!=expected['patch_sha256']:
        raise ValueError('SGLang base revision/patch differs from the recorded training-capture base')
    for name,digest in {**expected['files'],**expected['capture_base_files']}.items():
        if sha(base/name)!=digest:raise ValueError(f'SGLang base file mismatch: {name}')
    names=re.findall(r'^\+\+\+ b/(.+)$',patch.read_text(),re.M)
    if not names or any(not name.startswith('python/sglang/') or '..' in Path(name).parts for name in names):
        raise ValueError('Capture patch contains unexpected paths')
    key=hashlib.sha256((str(base)+head+diff_sha+sha(patch)).encode()).hexdigest()
    cache.mkdir(parents=True,exist_ok=True)
    view=cache/key
    with (cache/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not view.exists():
            stage=Path(tempfile.mkdtemp(prefix=key+'.building-',dir=cache))
            for root,dirs,files in os.walk(base/'python/sglang'):
                dirs[:]=[d for d in dirs if d not in ('__pycache__','.git')]
                relative=Path(root).relative_to(base)
                destination=stage/relative;destination.mkdir(parents=True,exist_ok=True)
                for name in files:
                    source=Path(root)/name
                    if name.endswith('.pyc') or name=='.spec_capture_patch.applied':continue
                    target=destination/name
                    if (relative/name).as_posix() in names:shutil.copy2(source,target)
                    else:target.symlink_to(source)
            env=dict(os.environ,GIT_CEILING_DIRECTORIES=str(cache))
            subprocess.run(['git','apply','--check',str(patch)],cwd=stage,env=env,check=True)
            subprocess.run(['git','apply',str(patch)],cwd=stage,env=env,check=True)
            # Prove no patch followed a link into the baseline source.
            for name,digest in expected['capture_base_files'].items():
                if sha(base/name)!=digest:raise RuntimeError(f'Base unexpectedly changed: {name}')
            manifest=dict(base=str(base),commit=head,base_diff_sha256=diff_sha,
                capture_patch_sha256=sha(patch),files={name:sha(stage/name) for name in names})
            (stage/'capture-overlay.json').write_text(json.dumps(manifest,indent=2))
            stage.rename(view)
        manifest=json.loads((view/'capture-overlay.json').read_text())
        if manifest['capture_patch_sha256']!=sha(patch) or manifest['base']!=str(base):
            raise ValueError('Capture overlay provenance mismatch')
        for name,digest in manifest['files'].items():
            target=view/name
            if target.is_symlink() or sha(target)!=digest:
                raise ValueError(f'Capture overlay modified: {name}')
        print(json.dumps({'training_capture_overlay':str(view),'base':str(base),
            'base_commit':head,'base_diff_sha256':diff_sha,'capture_patch_sha256':sha(patch)}),file=sys.stderr)
        return view/'python'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--cache',type=Path,default=Path(tempfile.gettempdir())/f'xj-specforge-capture-{os.getuid()}')
    parser.add_argument('--patch-dir',type=Path,default=Path(__file__).resolve().parents[1]/'patches/sglang/hn47-01889693')
    args=parser.parse_args()
    print(prepare(args.base,args.cache,args.patch_dir))
