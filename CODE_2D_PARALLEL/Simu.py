import json
import subprocess
from pathlib import Path
import gui  # adjust if needed

def main():
    
    
    params = gui.run_gui()  # opens tkinter, returns dict after Launch
    proc = params["proc"]

    cfg = Path("simulation_input.json")
    cfg.write_text(json.dumps(params, indent=2))

    subprocess.run(
        [
            "mpiexec", "-n", str(proc), "python",
            "./CNRS-EM2C-MODELE-FILM-NANO-PARTICULE/CODE_2D_PARALLEL/Collision_model_parallel.py",
            "--config", str(cfg)
        ],
        check=True
    )

if __name__ == "__main__":
    main()