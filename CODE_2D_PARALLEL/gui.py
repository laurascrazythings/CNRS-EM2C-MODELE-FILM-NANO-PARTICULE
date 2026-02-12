import tkinter as tk
from tkinter import ttk, messagebox
# make sure you have downloaded it into your conda or homebrew
# tk._test() #to test on your software, ensurign its downloaded

def main():
    app = Application()
    app.mainloop()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("User Initialisation")
        # #configure row columns
        Nb_rows = 25
        for r in range(Nb_rows):
            self.rowconfigure(r, weight = 1)    
        Nb_columns = 2
        for c in range(Nb_columns):
            self.columnconfigure(c, weight = 1)
        
        #-----------    
        #Header Label 1 - Time config
        header_1 = ttk.Label(self, text = "Time Inputs", anchor = "center")
        header_1.grid(row = 0, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 1 - Simulation Time
        self.text_frame_1 = tk.StringVar(value = " Simulation Time : float (in Seconds):")
        self.default_value_1 = tk.DoubleVar(value = 100)
        frame_1 = InputForm_float(self, text_var = self.text_frame_1, default_value = self.default_value_1) #create a frame
        frame_1.grid(row = 1, column = 0, sticky="nsew", padx=5, pady = 5)  
        #frame 2 - Time adding particles
        self.text_frame_2 = tk.StringVar(value = "Time particules are added : float (in Seconds):")
        self.default_value_2 = tk.DoubleVar(value = 1)
        frame_2 = InputForm_float(self, text_var = self.text_frame_2, default_value = self.default_value_2) #create a frame
        frame_2.grid(row = 2, column = 0, sticky="nsew", padx=5, pady = 5) 
        #frame 3 - dt
        self.text_frame_3 = tk.StringVar(value = "Delta Time / Time Step : float (in Seconds):")
        self.default_value_3 = tk.DoubleVar(value = 0.05)
        frame_3 = InputForm_float(self, text_var = self.text_frame_3, default_value = self.default_value_3) #create a frame
        frame_3.grid(row = 3, column = 0, sticky="nsew", padx=5, pady = 5) 
        
        #-----------
        #Header Label 2 - Mesh config
        header_2 = ttk.Label(self, text = "Mesh Inputs", anchor = "center")
        header_2.grid(row = 4, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 4 - Domain - x
        self.text_frame_4 = tk.StringVar(value = " Domain in X : integer (in Micrometer)")
        self.default_value_4 = tk.IntVar(value = 100)
        frame_4 = InputForm_int(self, text_var = self.text_frame_4, default_value = self.default_value_4) #create a frame
        frame_4.grid(row = 5, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 5 - Domain - y
        self.text_frame_5 = tk.StringVar(value = " Domain in Y : integer (in Micrometer)")
        self.default_value_5 = tk.IntVar(value = 100)
        frame_5 = InputForm_int(self, text_var = self.text_frame_5, default_value = self.default_value_5) #create a frame
        frame_5.grid(row = 6, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #-----------
        #Header Label 3 - Particles config
        header_3 = ttk.Label(self, text = "Particle Inputs", anchor = "center")
        header_3.grid(row = 7, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 6 - Number of particles
        self.text_frame_6 = tk.StringVar(value = " Number of particles : integer")
        self.default_value_6 = tk.IntVar(value = 1000)
        frame_6 = InputForm_int(self, text_var = self.text_frame_6, default_value = self.default_value_6) #create a frame
        frame_6.grid(row = 8, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 7 - Number of additional particles per dt
        self.text_frame_7 = tk.StringVar(value = " Number of particles added per dt : integer")
        self.default_value_7 = tk.IntVar(value = 1)
        frame_7 = InputForm_int(self, text_var = self.text_frame_7, default_value = self.default_value_7) #create a frame
        frame_7.grid(row = 9, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #-----------
        #Header Label 4 - Particle config
        header_4 = ttk.Label(self, text = "Particle configuration", anchor = "center")
        header_4.grid(row = 10, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 8 - Hamaker Constant
        self.text_frame_8 = tk.StringVar(value = " Hamaker constant : float (example: 6e-20)")
        self.default_value_8 = tk.DoubleVar(value = 6e-20)
        frame_8 = InputForm_float(self, text_var = self.text_frame_8, default_value = self.default_value_8) #create a frame
        frame_8.grid(row = 11, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 9 - Radius
        self.text_frame_9 = tk.StringVar(value = "Particle Radius : float (in micrometer)")
        self.default_value_9 = tk.DoubleVar(value = 0.01)
        frame_9 = InputForm_float(self, text_var = self.text_frame_9, default_value = self.default_value_9) #create a frame
        frame_9.grid(row = 12, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 10 - Density
        self.text_frame_10 = tk.StringVar(value = "Density of the particles : float (in g/m^3)")
        self.default_value_10 = tk.DoubleVar(value = 4500000)
        frame_10 = InputForm_float(self, text_var = self.text_frame_10, default_value = self.default_value_10) #create a frame
        frame_10.grid(row = 13, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 11 - Molar Mass
        self.text_frame_11 = tk.StringVar(value = "Molar Mass of the particles : float (in g/mol)")
        self.default_value_11 = tk.DoubleVar(value = 79.9)
        frame_11 = InputForm_float(self, text_var = self.text_frame_11, default_value = self.default_value_11) #create a frame
        frame_11.grid(row = 14, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #-----------
        #Header Label 5 - Velocity
        header_5 = ttk.Label(self, text = "Particle Velocity", anchor = "center")
        header_5.grid(row = 15, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 12 - Lowest X Velocity
        self.text_frame_12 = tk.StringVar(value = " Lowest Velocity in X : float (in micrometers / s)")
        self.default_value_12 = tk.DoubleVar(value = -1)
        frame_12 = InputForm_float(self, text_var = self.text_frame_12, default_value = self.default_value_12) #create a frame
        frame_12.grid(row = 16, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 13 - Highest X Velocity
        self.text_frame_13 = tk.StringVar(value = " Highest Velocity in X : float (in micrometers / s)")
        self.default_value_13 = tk.DoubleVar(value = 1)
        frame_13 = InputForm_float(self, text_var = self.text_frame_13, default_value = self.default_value_13) #create a frame
        frame_13.grid(row = 17, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 14 - Lowest Y Velocity
        self.text_frame_14 = tk.StringVar(value = " Lowest Velocity in Y : float (in micrometers / s)")
        self.default_value_14 = tk.DoubleVar(value = 0.01)
        frame_14 = InputForm_float(self, text_var = self.text_frame_14, default_value = self.default_value_14) #create a frame
        frame_14.grid(row = 18, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 15 - Highest Y Velocity
        self.text_frame_15 = tk.StringVar(value = " Highest Velocity in Y : float (in micrometers / s)")
        self.default_value_15 = tk.DoubleVar(value = 1)
        frame_15 = InputForm_float(self, text_var = self.text_frame_15, default_value = self.default_value_15) #create a frame
        frame_15.grid(row = 19, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #-----------
        #Header Label 6 - Brownian
        header_6 = ttk.Label(self, text = "Brownian", anchor = "center")
        header_6.grid(row = 20, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 16 - Relaxation Time
        self.text_frame_16 = tk.StringVar(value = " Relaxation Time : float (in s)")
        self.default_value_16 = tk.DoubleVar(value = 1)
        frame_16 = InputForm_float(self, text_var = self.text_frame_16, default_value = self.default_value_16) #create a frame
        frame_16.grid(row = 21, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 17 - Noise Diffusion
        self.text_frame_17 = tk.StringVar(value = " Noise Diffusion : float")
        self.default_value_17 = tk.DoubleVar(value = 0.5)
        frame_17 = InputForm_float(self, text_var = self.text_frame_17, default_value = self.default_value_17) #create a frame
        frame_17.grid(row = 22, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 18 - X Air Velocity
        self.text_frame_18 = tk.StringVar(value = "X Air velocity : float (in micrometers / s)")
        self.default_value_18 = tk.DoubleVar(value = 0.0)
        frame_18 = InputForm_float(self, text_var = self.text_frame_18, default_value = self.default_value_18) #create a frame
        frame_18.grid(row = 23, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #frame 19 - Y Air Velocity
        self.text_frame_19 = tk.StringVar(value = "Y Air velocity : float (in micrometers / s)")
        self.default_value_19 = tk.DoubleVar(value = 1)
        frame_19 = InputForm_float(self, text_var = self.text_frame_19, default_value = self.default_value_19) #create a frame
        frame_19.grid(row = 24, column = 0, sticky="nsew", padx=5, pady = 5)
        
        #-----------
        #Header Label 7 - Walls
        header_7 = ttk.Label(self, text = "Walls: if False then periodical condition", anchor = "center")
        header_7.grid(row = 0, column = 1, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 20 - Right Wall
        self.text_frame_20 = tk.StringVar(value = "Right? : boolean")
        self.default_value_20 = tk.BooleanVar(value = "False")
        frame_20 = InputForm_bool(self, text_var = self.text_frame_20, default_value = self.default_value_20) #create a frame
        frame_20.grid(row = 1, column = 1, sticky="nsew", padx=5, pady = 5)
        
        
        
        
        
        
        
        
        
        
        
 #imputing floats       
class InputForm_float(ttk.Frame):
    def __init__(self, parent, text_var: tk.StringVar, default_value : tk.DoubleVar):
        super().__init__(parent)
        
        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight = 1)
        
        #add a label
        self.text_var = text_var
        self.default_value = default_value
        self.label = ttk.Label(self, textvariable = text_var)
        self.label.grid(row = 0, column = 0, sticky = "ew")

        self.entry = ttk.Entry(self, textvariable=self.default_value) #create en entry area
        self.entry.grid(row = 0, column = 1, sticky= "ew")

        #return is like add
        self.entry.bind("<Return>", self.add_to_list_float)
        
        self.text_list = tk.Listbox(self, height = 1)#text box area
        self.text_list.grid(row = 0, column= 2, rowspan=1, sticky="ew")
        #self.text_list.insert(tk.END, default_value)

        self.entry_btn = ttk.Button(self, text= "Add", command= self.add_to_list_float)#entry button
        self.entry_btn.grid(row=0, column= 3)
        
        self.entry_btn_2 = ttk.Button(self, text= "Clear", command= self.clear_list)#entry button
        self.entry_btn_2.grid(row=0, column= 4)

        
        
    def add_to_list_float(self, event = None ):
        text = self.entry.get().strip() # strip takes the space away
        if not text:
            return
        
        #ensure it is an int
        try:
            text = float(text)
        except ValueError:
            messagebox.showerror("Float expected")
            return
        #add the int
        self.text_list.delete(0, tk.END)
        self.text_list.insert(tk.END, text)
        self.entry.delete(0, tk.END)
    
    def clear_list(self):
        self.text_list.delete(0, tk.END)

#inputing ints
class InputForm_int(ttk.Frame,):
    def __init__(self, parent, text_var: tk.StringVar, default_value : tk.IntVar):
        super().__init__(parent)
        
        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight = 1)
        
        #add a label
        self.text_var = text_var
        self.default_value = default_value
        self.label = ttk.Label(self, textvariable = text_var)
        self.label.grid(row = 0, column = 0, sticky = "ew")

        self.entry = ttk.Entry(self, textvariable=self.default_value) #create en entry area
        self.entry.grid(row = 0, column = 1, sticky= "ew")
        
        #return is like add
        self.entry.bind("<Return>", self.add_to_list_int)
        
        self.text_list = tk.Listbox(self, height = 1)#text box area
        self.text_list.grid(row = 0, column= 2, rowspan=1, sticky="ew")

        self.entry_btn = ttk.Button(self, text= "Add", command= self.add_to_list_int)#entry button
        self.entry_btn.grid(row=0, column= 3)
        
        self.entry_btn_2 = ttk.Button(self, text= "Clear", command= self.clear_list)#entry button
        self.entry_btn_2.grid(row=0, column= 4)

        
        
    def add_to_list_int(self, event = None ):
        text = self.entry.get().strip() # strip takes the space away
        if not text:
            return
        
        #ensure it is an int
        try:
            text = int(text)
        except ValueError:
            messagebox.showerror("Integer expected")
            return
        #add the int
        self.text_list.delete(0, tk.END)
        self.text_list.insert(tk.END, text)
        self.entry.delete(0, tk.END)
    
    def clear_list(self):
        self.text_list.delete(0, tk.END)
        
#imputing booleans      
class InputForm_bool(ttk.Frame):
    def __init__(self, parent, text_var: tk.StringVar, default_value : tk.BooleanVar):
        super().__init__(parent)
        
        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight = 1)
        
        #add a label
        self.text_var = text_var
        self.default_value = default_value
        self.label = ttk.Label(self, textvariable = text_var)
        self.label.grid(row = 0, column = 0, sticky = "ew")

        self.entry = ttk.Entry(self, textvariable=self.default_value) #create en entry area
        self.entry.grid(row = 0, column = 1, sticky= "ew")

        #return is like add
        self.entry.bind("<Return>", self.add_to_list_bool)
        
        self.text_list = tk.Listbox(self, height = 1)#text box area
        self.text_list.grid(row = 0, column= 2, rowspan=1, sticky="ew")
        #self.text_list.insert(tk.END, default_value)

        self.entry_btn = ttk.Button(self, text= "Add", command= self.add_to_list_bool)#entry button
        self.entry_btn.grid(row=0, column= 3)
        
        self.entry_btn_2 = ttk.Button(self, text= "Clear", command= self.clear_list)#entry button
        self.entry_btn_2.grid(row=0, column= 4)

        
        
    def add_to_list_bool(self, event = None ):
        text = self.entry.get().strip() # strip takes the space away
        if not text:
            return
        
        #ensure it is an int
        try:
            text = bool(text)
        except ValueError:
            messagebox.showerror("Boolean expected")
            return
        #add the int
        self.text_list.delete(0, tk.END)
        self.text_list.insert(tk.END, text)
        self.entry.delete(0, tk.END)
    
    def clear_list(self):
        self.text_list.delete(0, tk.END)   
#running the script
if __name__ == "__main__":
    main()

