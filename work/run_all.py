"""Re-run the whole pipeline in order from the project root."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
(ROOT / "outputs" / "eda").mkdir(parents=True, exist_ok=True)
with open(ROOT / "outputs" / "eda" / "00_data_exploration_stdout.txt", "w", encoding="utf-8") as log:
    subprocess.run([sys.executable, "00_data_exploration.py"], cwd=ROOT, env=env, stdout=log,
                   stderr=subprocess.STDOUT, check=True)
for script in ["01_eda.py", "02_train_shap.py", "03_task3_local.py", "04_task4_global.py", "05_task6.py",
               "06_background_repeat.py", "08_build_slides.py", "07_build_report.py"]:
    print("running", script, flush=True)
    subprocess.run([sys.executable, script], cwd=ROOT / "work", env=env, stdout=subprocess.DEVNULL, check=True)
print("all done")
