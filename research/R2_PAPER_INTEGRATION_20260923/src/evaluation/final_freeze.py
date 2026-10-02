"""Verify immutable learned artifacts before the authorized final-year test."""
from src.runtime import ROOT
from pathlib import Path
import hashlib
import json

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_freeze():
    manifest=json.loads((ROOT/'frozen_candidate/EVALUATION_FREEZE.json').read_text())
    assert manifest['status']=='frozen'
    assert manifest['user_authorized_final_year_test'] is True
    assert manifest['post_evaluation_parameter_changes_allowed'] is False
    assert manifest['seeds']==[20260920,20260921,20260922]
    for record in manifest['files']:
        assert digest(record['path'])==record['sha256'], 'Frozen artifact changed: '+record['path']
    return manifest
