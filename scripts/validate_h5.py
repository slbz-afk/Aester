"""Read a finished job and save a new H5 report without editing its old records."""
import argparse
import datetime as dt
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack.h5_job_validation import evaluate_job
from starstack.acceptance_status import render_checks
from starstack.records import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "h5-validation-20260919"))
    parser.add_argument("--sample-step", type=int, default=1)
    args = parser.parse_args(argv)
    out = Path(args.output_root).resolve() / ("h5-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ-")+uuid.uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    report = evaluate_job(Path(args.job), sample_step=args.sample_step)
    atomic_json(out / "h5.json", report)
    text = "# H5 同位置合成验收\n\n"+render_checks(report["judgement"])
    text += "\n\n本报告检查实际记录权重是否正确应用；天空可见性、插值支持与摄影接边质量仍需分别验证。\n"
    (out / "H5.md").write_text(text, encoding="utf-8")
    print(f"{report['judgement']['status']}: {out / 'H5.md'}")
    return {"passed": 0, "failed": 2, "insufficient_data": 3, "not_applicable": 4}[report["judgement"]["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
