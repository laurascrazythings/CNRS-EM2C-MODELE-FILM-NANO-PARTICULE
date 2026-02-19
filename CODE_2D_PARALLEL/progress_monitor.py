# progress_monitor.py  (NEW FILE)
from pathlib import Path
import argparse
from Progress_GUI import Progression_Window  # <-- reuses your class

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=float, required=True)
    ap.add_argument("--proc", type=int, required=True)
    args = ap.parse_args()

    pb = Progression_Window(maximum=args.max, proc_num = args.proc)  # <-- reuse

    progress_path = Path("progress.txt").resolve()
    progress_path.write_text("0.0")
    
    merge_path = Path("merge_progress.txt").resolve()
    merge_path.write_text("0.0")

    def poll():
        try:
            val = float(progress_path.read_text().strip())
            pb.set(val, f"{val:.2f}/{args.max:.2f} s")
            if val >= args.max:
                pb.set(args.max, "Multiproc Done.")
        except Exception:
            pass
        
        # ---- Bar 2: merge progress (0..proc) ----
        try:
            m = int(float(merge_path.read_text().strip()))
            pb.set_2(m, f"Merging {m}/{args.proc}")
            if m >= args.proc:
                pb.set(args.proc, "Merge Proc Done")
        except Exception:
            pass

        pb.after(500, poll)  # <-- IMPORTANT: schedule next poll (Tk-friendly)

    poll()
    pb.mainloop()

if __name__ == "__main__":
    main()