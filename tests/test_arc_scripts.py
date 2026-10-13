from pathlib import Path


def test_arc_jobs_are_batch_only_and_use_beegfs():
    arc = Path("arc")
    jobs = list(arc.glob("*.sbatch"))
    assert {p.name for p in jobs} >= {"setup.sbatch", "preflight-4060.sbatch", "smoke-4060.sbatch", "full-4060.sbatch", "full-a6000.sbatch"}
    all_text = "\n".join(p.read_text() for p in list(arc.glob("*")) if p.is_file())
    assert "salloc" not in all_text
    assert "srun" not in all_text
    assert "/mnt/beegfs/ssrini27/failfast-step0" in all_text
    assert "GITHUB_TOKEN" not in all_text
    assert "#SBATCH --mem=" not in all_text
    assert "#SBATCH --gres=" not in all_text
    assert "/home/ssrini27/failfast-arc/gates" in all_text


def test_full_submit_checks_gate_markers():
    text = Path("arc/submit_full.sh").read_text()
    assert "PREFLIGHT_OK" in text
    assert "SMOKE_OK" in text
    assert "git ls-remote" in text


def test_smoke_exports_partial_evidence_on_failure():
    text = Path("arc/smoke-4060.sbatch").read_text()
    assert "trap export_evidence EXIT" in text
    assert text.index("smoke_run_path.txt") < text.index("for CONFIG in")


def test_experiment_jobs_force_hugging_face_offline():
    text = Path("arc/common.sh").read_text()
    assert 'export HF_HUB_OFFLINE=1' in text
    assert 'export TRANSFORMERS_OFFLINE=1' in text
