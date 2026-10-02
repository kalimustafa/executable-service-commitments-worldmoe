"""Download the two official JSSPP Frontier archives and retain exact provenance."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import requests,zipfile,hashlib,json,shutil
from paths import BASE
PAGE='https://jsspp.org/workload/index.php?page=frontier25'
def main():
    (BASE/'data').mkdir(parents=True,exist_ok=True)
    r=requests.get(PAGE,timeout=30);r.raise_for_status()
    (BASE/'data/frontier_archive_description.html').write_bytes(r.content)
    def one(year):
        out=BASE/'data'/f'frontier_{year}';out.mkdir(parents=True,exist_ok=True)
        url=f'https://jsspp.org/workload/logs/2026/frontier-{year}.zip';p=out/f'frontier-{year}.zip'
        if not p.exists():
            with requests.get(url,timeout=(20,60),stream=True) as r:
                r.raise_for_status()
                with p.with_suffix('.part').open('wb') as f:
                    for chunk in r.iter_content(1024*1024):f.write(chunk)
            p.with_suffix('.part').replace(p)
        with zipfile.ZipFile(p) as z:
            names=z.namelist()
            for name in names:
                target=(out/name).resolve()
                if not target.is_relative_to(out.resolve()):raise ValueError('Unsafe archive member')
                if not name.endswith('/'):
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(name) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
        rec=dict(year=year,url=url,description_url=PAGE,downloaded_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,members=names)
        (out/'download_manifest.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
        print(json.dumps(rec),flush=True)
        for path in out.rglob('*'):
            if path.is_file() and path.suffix.lower() not in ['.zip','.json']:
                with path.open(encoding='utf-8',errors='replace') as f:lines=[next(f,'').rstrip() for _ in range(50)]
                print(path.name+'\n'+'\n'.join(lines),flush=True)
        return rec
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,[2024,2025]))
if __name__=='__main__':main()
