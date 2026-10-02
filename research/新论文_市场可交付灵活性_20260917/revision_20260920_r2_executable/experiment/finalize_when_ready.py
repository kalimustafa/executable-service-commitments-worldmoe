"""Finish dependent outputs after the ongoing, continuous old-R replay."""
from pathlib import Path
import subprocess,time,json,sys
R=Path(__file__).resolve().parents[1]
scientific=sys.executable
document=Path('C:/Users/Kali/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
began=time.monotonic();last=0
while not (R/'neural_2025/completed.json').exists():
    elapsed=time.monotonic()-began
    if elapsed>1800:raise TimeoutError('Matched neural replay not yet complete; inspect its progress and log')
    if elapsed-last>60:
        p=R/'neural_2025/progress.json'
        print('WAITING',p.read_text() if p.exists() else 'loading',flush=True);last=elapsed
    time.sleep(5)
for script in ['analyze_economics.py','report_ood.py','build_figures.py','build_supplement.py','translate_supplement.py','assemble_manuscript.py','final_report.py']:
    print('RUN',script,flush=True);subprocess.run([scientific,str(R/'experiment'/script)],check=True)
print('BUILD DOCUMENTS',flush=True)
subprocess.run([str(document),str(R/'manuscript/build_documents.py')],check=True)
print('RENDER WORD',flush=True)
subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(R/'manuscript/render_word.ps1')],check=True)
print('FINALIZATION GENERATED; VISUAL QA REQUIRED',flush=True)
