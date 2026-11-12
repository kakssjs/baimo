"""Command line: python -m pipeline VIDEO [--crop W:H:X:Y]"""
import argparse
import sys

from .job import run_job

parser = argparse.ArgumentParser(prog="python -m pipeline", description="Run one video through the pipeline.")
parser.add_argument("video")
parser.add_argument("--crop", help='ffmpeg crop "w:h:x:y" applied to the input, e.g. "iw:ih/2:0:0" for the top half')
args = parser.parse_args()

job_dir = run_job(args.video, args.crop)
print((job_dir / "job.json").read_text(encoding="utf-8"))
print(job_dir)
sys.exit(0 if (job_dir / "final.mp4").exists() else 1)
