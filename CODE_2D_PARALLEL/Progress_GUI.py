import tkinter as tk
from tkinter import ttk

# def main():
#     pw = Progression_Window()
#     pw.mainloop()
    
class Progression_Window(tk.Tk):
    def __init__(self, maximum, proc_num):
        super().__init__()
        self.title("Progress Window")
        self.lift() #put in front                   
        self.attributes("-topmost", True)  # force on top (temporarily)
        self.after(200, lambda: self.attributes("-topmost", False))  # release topmost
        self.focus_force()                 # focus it
        self.update_idletasks()
        w, h = 900, 400 # width x height
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")
                
        # self.protocol("WM_DELETE_WINDOW", lambda: None)# preventing it form being closed
        
        # #configure row columns
        Nb_rows = 6
        for r in range(Nb_rows):
            self.rowconfigure(r, weight = 1)    
        Nb_columns = 1
        for c in range(Nb_columns):
            self.columnconfigure(c, weight = 1)
        
        self.msg = tk.StringVar(value="Starting multi processor calculation") 
        self.msg_2 = tk.StringVar(value="Starting merging processors") 
        self.msg_3 = tk.StringVar(value="The video compilation is starting")
    
        #-----------    
        #Header Label 1 - Simulation Progress
        header_1 = ttk.Label(self, textvariable=self.msg, anchor = "center")
        header_1.grid(row = 0, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        # thicker progressbar via style
        style = ttk.Style(self)
        style.configure("Thick.Horizontal.TProgressbar", thickness=25)

        #bar 1 - multiproc progress bar
        self.bar_1 = ttk.Progressbar(self, mode = "determinate", maximum = maximum, style = "Thick.Horizontal.TProgressbar")
        self.bar_1.grid(row = 1, column = 0, rowspan= 1, sticky= "nsew", padx= 5, pady= (0, 12))
        
        #-----------    
        #Header Label 2 - Processor Merging
        header_2 = ttk.Label(self, textvariable=self.msg_2, anchor = "center")
        header_2.grid(row = 2, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))

        #bar 2 - multiproc progress bar
        self.bar_2 = ttk.Progressbar(self, mode = "determinate", maximum = proc_num, style = "Thick.Horizontal.TProgressbar")
        self.bar_2.grid(row = 3, column = 0, rowspan= 1, sticky= "nsew", padx= 5, pady= (0, 12))
        
        #-----------    
        #Header Label 3 - Video Compilation
        header_3 = ttk.Label(self, textvariable = self.msg_3 , anchor = "center")
        header_3.grid(row = 4, column = 0, columnspan = 1, rowspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #bar 3 - Video expected progress bar
        self.bar_3 = ttk.Progressbar(self, mode = "determinate", maximum = 100, style = "Thick.Horizontal.TProgressbar")
        self.bar_3.grid(row = 5, column = 0, rowspan= 1, sticky= "nsew", padx= 5, pady= (0, 12))
        
        
        self.update()
    
    def set(self, value, text = None):
        self.bar_1["value"] = value
        if text is not None:
            self.msg.set(text)
        self.update_idletasks()
    
    def set_2(self, value, text = None):
        self.bar_2["value"] = value
        if text is not None:
            self.msg_2.set(text)
        self.update_idletasks()
        
    def set_3(self, value, text = None):
        self.bar_3["value"] = value
        if text is not None:
            self.msg_3.set(text)
        self.update_idletasks()
         
    def close(self):
        self.destroy()
        
    
        
#running the script
# if __name__ == "__main__":
#     main()