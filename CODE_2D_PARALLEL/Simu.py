import json
import subprocess
from pathlib import Path
import gui  # adjust if needed
import sys

def main():
    
    params = gui.run_gui()  # opens tkinter, returns dict after Launch
    proc = int(params["proc"])

    cfg = Path("simulation_input.json")
    cfg.write_text(json.dumps(params, indent=2))
    
    T = float(params["Tsim"])
    
    script_dir = Path(__file__).parent
    monitor_script = script_dir / "progress_monitor.py"

    monitor = subprocess.Popen([sys.executable, "-u", str(monitor_script), "--max", str(T), "--proc", str(proc)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    import time
    time.sleep(0.1)

    if monitor.poll() is not None:
        out, err = monitor.communicate()
        raise RuntimeError(f"progress_monitor.py exited.\nSTDOUT:\n{out}\nSTDERR:\n{err}")

    try:
        subprocess.run(
            [
                "mpiexec", "-n", str(proc), "python3",
                "./CODE_2D_PARALLEL/Collision_model_parallel.py",
                "--config", str(cfg)
            ],
            check=True
            )
    finally:
        monitor.terminate()

if __name__ == "__main__":
    main()