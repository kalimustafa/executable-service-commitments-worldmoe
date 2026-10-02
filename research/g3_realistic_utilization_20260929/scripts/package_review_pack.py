from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
ARCHIVE=WORKSPACE/"G3_REALISTIC_UTILIZATION_REVIEW_PACK_20260929.zip"
TOP=["PLAN.md","README.md"]
DIRS=["configs","scripts","outputs","figures","tables","manuscript_patch","reports","palette"]
SKIP_PARTS={"__pycache__","numba_cache","cache","raw","raw_nyiso","trajectories"}
SKIP_SUFFIXES={".pyc",".pt",".pth",".ckpt",".npz",".npy",".parquet"}

def main():
    members=[]
    for rel in TOP:
        p=ROOT/rel
        if p.is_file():members.append(p)
    for folder in DIRS:
        base=ROOT/folder
        for p in base.rglob("*"):
            if not p.is_file():continue
            rel=p.relative_to(ROOT)
            if any(part.lower() in SKIP_PARTS for part in rel.parts):continue
            if p.suffix.lower() in SKIP_SUFFIXES:continue
            members.append(p)
    with ZipFile(ARCHIVE,"w",compression=ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(members)):
            z.write(p,Path(ROOT.name)/p.relative_to(ROOT))
    print(f"{ARCHIVE}\nfiles={len(set(members))}\nbytes={ARCHIVE.stat().st_size}")

if __name__=="__main__":main()
