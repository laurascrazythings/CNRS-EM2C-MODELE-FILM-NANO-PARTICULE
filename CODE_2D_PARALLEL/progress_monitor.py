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
    
    video_path = Path("video_progress.txt").resolve()
    video_path.write_text("0.0")

    def poll():
         # ---- Bar 1: simulation time progress ----
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
                pb.set_2(args.proc, "Merge Proc Done")
        except Exception:
            pass
        
        # ---- Bar 3: video progress ----
        try:
            v = float(video_path.read_text().strip())
            pb.set_3(v, f"Video Progress {v}%")
            if v >= 100:
                pb.set_3(100, "Video Compilation Done")
        except Exception:
            pass
        

        pb.after(500, poll)

    poll()
    pb.mainloop()

if __name__ == "__main__":
    main()