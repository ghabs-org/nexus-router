import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_guarded_gating(tmp_path, monkeypatch):
    # create minimal report with low samples
    report = {"ok": True, "model_task_signals_top": [{"task":"t1","model":"m1","samples":2,"centered_signal":0.5}], "critical_drift": False}
    repfile = tmp_path / "report.json"
    repfile.write_text(json.dumps(report))

    # monkeypatch report script to output our report
    fake = SCRIPTS / "fake_report.py"
    fake.write_text(f"import json;print({json.dumps(report)})")
    # run auto_tune with guarded apply; should fail gates due to samples
    res = subprocess.run([str(SCRIPTS / "auto_tune.py"), "--guarded-apply", "--min-total-samples", "10"], check=False, capture_output=True, text=True)
    assert "gates_pass=False" in res.stdout


def test_unknown_policy_require_higher(tmp_path):
    report = {"ok": True, "model_task_signals_top": [{"task":"unknown","model":"m1","samples":5,"centered_signal":0.6}]}
    repfile = tmp_path / "r2.json"
    repfile.write_text(json.dumps(report))
    fake = SCRIPTS / "fake_report2.py"
    fake.write_text(f"import json;print({json.dumps(report)})")
    res = subprocess.run([str(SCRIPTS / "auto_tune.py"), "--guarded-apply", "--unknown-policy", "require_higher", "--min-samples", "5"], check=False, capture_output=True, text=True)
    # when requiring higher, the unknown should be filtered and no adjustments found
    assert "No actionable adjustments found." in res.stdout


def test_rollback_no_snapshots(tmp_path):
    # run rollback --latest when no snapshots exist
    res = subprocess.run([str(SCRIPTS / "auto_tune_rollback.py"), "--latest"], check=False, capture_output=True, text=True)
    assert "no snapshots available" in res.stdout
