from pathlib import Path
import hashlib
root=Path(__file__).resolve().parent
count=0
for line in (root/'SHA256SUMS.txt').read_text().splitlines():
    expected,name=line.split('  ',1);path=root/name
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):digest.update(block)
    if digest.hexdigest()!=expected:raise SystemExit('SHA-256 mismatch: '+name)
    count+=1
print('Verified',count,'package files')
