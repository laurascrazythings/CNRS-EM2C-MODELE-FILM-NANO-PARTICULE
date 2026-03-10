import tkinter as tk
from tkinter import ttk, messagebox
import json
from pathlib import Path
# make sure you have downloaded it into your conda or homebrew
# tk._test() #to test on your software, ensurign its downloaded

def main():
    app = Application()
    app.mainloop()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.params = None
        self.title("User Initialisation")

        # #configure row columns
        Nb_rows = 25
        for r in range(Nb_rows):
            self.rowconfigure(r, weight = 1)    
        Nb_columns = 2
        for c in range(Nb_columns):
            self.columnconfigure(c, weight = 1)
        
        self.forms = [] #Store 
        
        #-----------    
        #Header Label 1 - Time config
        header_1 = ttk.Label(self, text = "Time Inputs", anchor = "center")
        header_1.grid(row = 0, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 1 - Simulation Time
        self.text_frame_1 = tk.StringVar(value = " Simulation Time : float (in Seconds):")
        self.default_value_1 = tk.DoubleVar(value = 100)
        self.fav_value_1 = tk.DoubleVar(value = 100)
        frame_1 = InputForm_float(self, key = "Tsim", text_var = self.text_frame_1, default_value = self.default_value_1, fav_1_value = self.fav_value_1) #create a frame
        frame_1.grid(row = 1, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_1)  
        #frame 2 - Time adding particles
        self.text_frame_2 = tk.StringVar(value = "Time particules are added : float (in Seconds):")
        self.default_value_2 = tk.DoubleVar(value = 1)
        self.fav_value_2 = tk.DoubleVar(value = 1)
        frame_2 = InputForm_float(self, key = "T_add", text_var = self.text_frame_2, default_value = self.default_value_2, fav_1_value = self.fav_value_2) #create a frame
        frame_2.grid(row = 2, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_2)  
        #frame 3 - dt
        self.text_frame_3 = tk.StringVar(value = "Delta Time / Time Step : float (in Seconds):")
        self.default_value_3 = tk.DoubleVar(value = 0.05)
        self.fav_value_3 = tk.DoubleVar(value = 0.05)
        frame_3 = InputForm_float(self, key = "dt", text_var = self.text_frame_3, default_value = self.default_value_3, fav_1_value = self.fav_value_3) #create a frame
        frame_3.grid(row = 3, column = 0, sticky="nsew", padx=5, pady = 5) 
        self.forms.append(frame_3) 
        
        #-----------
        #Header Label 2 - Mesh config
        header_2 = ttk.Label(self, text = "Mesh Inputs", anchor = "center")
        header_2.grid(row = 4, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 4 - Domain - x
        self.text_frame_4 = tk.StringVar(value = " Domain in X : integer (in Micrometer)")
        self.default_value_4 = tk.IntVar(value = 100)
        self.fav_value_4 = tk.IntVar(value = 100)
        frame_4 = InputForm_int(self, key = "Lx", text_var = self.text_frame_4, default_value = self.default_value_4, fav_1_value = self.fav_value_4) #create a frame
        frame_4.grid(row = 5, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_4) 
        
        #frame 5 - Domain - y
        self.text_frame_5 = tk.StringVar(value = " Domain in Y : integer (in Micrometer)")
        self.default_value_5 = tk.IntVar(value = 100)
        self.fav_value_5 = tk.IntVar(value = 100)
        frame_5 = InputForm_int(self, key = "Ly", text_var = self.text_frame_5, default_value = self.default_value_5, fav_1_value = self.fav_value_5) #create a frame
        frame_5.grid(row = 6, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_5) 
        
        #-----------
        #Header Label 3 - Particles config
        header_3 = ttk.Label(self, text = "Particle Inputs", anchor = "center")
        header_3.grid(row = 7, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 6 - Number of particles
        self.text_frame_6 = tk.StringVar(value = " Number of particles : integer")
        self.default_value_6 = tk.IntVar(value = 1000)
        self.fav_value_6 = tk.IntVar(value = 1000)
        frame_6 = InputForm_int(self, key = "Np", text_var = self.text_frame_6, default_value = self.default_value_6, fav_1_value = self.fav_value_6) #create a frame
        frame_6.grid(row = 8, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_6) 
        
        #frame 7 - Number of additional particles per dt
        self.text_frame_7 = tk.StringVar(value = " Number of particles added per dt : integer")
        self.default_value_7 = tk.IntVar(value = 1)
        self.fav_value_7 = tk.IntVar(value = 1)
        frame_7 = InputForm_int(self, key = "Np_dt", text_var = self.text_frame_7, default_value = self.default_value_7, fav_1_value = self.fav_value_7) #create a frame
        frame_7.grid(row = 9, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_7) 
        
        #-----------
        #Header Label 4 - Particle config
        header_4 = ttk.Label(self, text = "Particle configuration", anchor = "center")
        header_4.grid(row = 10, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 8 - Hamaker Constant
        self.text_frame_8 = tk.StringVar(value = " Hamaker constant : float (example: 6e-20)")
        self.default_value_8 = tk.DoubleVar(value = 6e-20)
        self.fav_value_8 = tk.DoubleVar(value = 6e-20)
        frame_8 = InputForm_float(self, key = "Hamaker", text_var = self.text_frame_8, default_value = self.default_value_8, fav_1_value = self.fav_value_8) #create a frame
        frame_8.grid(row = 11, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_8) 
        
        #frame 9 - Radius
        self.text_frame_9 = tk.StringVar(value = "Particle Radius : float (in micrometer)")
        self.default_value_9 = tk.DoubleVar(value = 0.01)
        self.fav_value_9 = tk.DoubleVar(value = 0.01)
        frame_9 = InputForm_float(self, key = "Rad_particle", text_var = self.text_frame_9, default_value = self.default_value_9, fav_1_value = self.fav_value_9) #create a frame
        frame_9.grid(row = 12, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_9) 
        
        #frame 10 - Density
        self.text_frame_10 = tk.StringVar(value = "Density of the particles : float (in g/m^3)")
        self.default_value_10 = tk.DoubleVar(value = 4500000)
        self.fav_value_10 = tk.DoubleVar(value = 4500000)
        frame_10 = InputForm_float(self, key = "Density_particle", text_var = self.text_frame_10, default_value = self.default_value_10, fav_1_value = self.fav_value_10) #create a frame
        frame_10.grid(row = 13, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_10) 
        
        #frame 11 - Molar Mass
        self.text_frame_11 = tk.StringVar(value = "Molar Mass of the particles : float (in g/mol)")
        self.default_value_11 = tk.DoubleVar(value = 79.9)
        self.fav_value_11 = tk.DoubleVar(value = 79.9)
        frame_11 = InputForm_float(self, key = "Molar_mass_particle", text_var = self.text_frame_11, default_value = self.default_value_11, fav_1_value = self.fav_value_11) #create a frame
        frame_11.grid(row = 14, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_11) 
        
        #-----------
        #Header Label 5 - Velocity
        header_5 = ttk.Label(self, text = "Particle Velocity", anchor = "center")
        header_5.grid(row = 15, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 12 - Lowest X Velocity
        self.text_frame_12 = tk.StringVar(value = " Lowest Velocity in X : float (in micrometers / s)")
        self.default_value_12 = tk.DoubleVar(value = -1)
        self.fav_value_12 = tk.DoubleVar(value = -1)
        frame_12 = InputForm_float(self, key = "Low_Particle_X_Velocity", text_var = self.text_frame_12, default_value = self.default_value_12, fav_1_value = self.fav_value_12) #create a frame
        frame_12.grid(row = 16, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_12)
        
        #frame 13 - Highest X Velocity
        self.text_frame_13 = tk.StringVar(value = " Highest Velocity in X : float (in micrometers / s)")
        self.default_value_13 = tk.DoubleVar(value = 1)
        self.fav_value_13 = tk.DoubleVar(value = 1)
        frame_13 = InputForm_float(self, key = "High_Particle_X_Velocity", text_var = self.text_frame_13, default_value = self.default_value_13, fav_1_value = self.fav_value_13) #create a frame
        frame_13.grid(row = 17, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_13) 
        
        #frame 14 - Lowest Y Velocity
        self.text_frame_14 = tk.StringVar(value = " Lowest Velocity in Y : float (in micrometers / s)")
        self.default_value_14 = tk.DoubleVar(value = 0.01)
        self.fav_value_14 = tk.DoubleVar(value = 0.01)
        frame_14 = InputForm_float(self, key = "Low_Particle_Y_Velocity", text_var = self.text_frame_14, default_value = self.default_value_14, fav_1_value = self.fav_value_14) #create a frame
        frame_14.grid(row = 18, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_14) 
        
        #frame 15 - Highest Y Velocity
        self.text_frame_15 = tk.StringVar(value = " Highest Velocity in Y : float (in micrometers / s)")
        self.default_value_15 = tk.DoubleVar(value = 1)
        self.fav_value_15 = tk.DoubleVar(value = 1)
        frame_15 = InputForm_float(self, key = "High_Particle_Y_Velocity", text_var = self.text_frame_15, default_value = self.default_value_15, fav_1_value = self.fav_value_15) #create a frame
        frame_15.grid(row = 19, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_15) 
        
        #-----------
        #Header Label 6 - Brownian
        header_6 = ttk.Label(self, text = "Brownian", anchor = "center")
        header_6.grid(row = 20, column = 0, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10)) 
        
        #frame 16 - Relaxation Time
        self.text_frame_16 = tk.StringVar(value = " Relaxation Time : float (in s)")
        self.default_value_16 = tk.DoubleVar(value = 1)
        self.fav_value_16 = tk.DoubleVar(value = 1)
        frame_16 = InputForm_float(self, key = "Tau", text_var = self.text_frame_16, default_value = self.default_value_16, fav_1_value = self.fav_value_16) #create a frame
        frame_16.grid(row = 21, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_16)
        
        #frame 17 - Noise Diffusion
        self.text_frame_17 = tk.StringVar(value = " Noise Diffusion : float")
        self.default_value_17 = tk.DoubleVar(value = 0.5)
        self.fav_value_17 = tk.DoubleVar(value = 0.5)
        frame_17 = InputForm_float(self, key = "B", text_var = self.text_frame_17, default_value = self.default_value_17, fav_1_value = self.fav_value_17) #create a frame
        frame_17.grid(row = 22, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_17)
        
        #frame 18 - X Air Velocity
        self.text_frame_18 = tk.StringVar(value = "X Air velocity : float (in micrometers / s)")
        self.default_value_18 = tk.DoubleVar(value = 0.0)
        self.fav_value_18 = tk.DoubleVar(value = 0.0)
        frame_18 = InputForm_float(self, key = "Air_X_Velocity", text_var = self.text_frame_18, default_value = self.default_value_18, fav_1_value = self.fav_value_18) #create a frame
        frame_18.grid(row = 23, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_18)
        
        #frame 19 - Y Air Velocity
        self.text_frame_19 = tk.StringVar(value = "Y Air velocity : float (in micrometers / s)")
        self.default_value_19 = tk.DoubleVar(value = 1)
        self.fav_value_19 = tk.DoubleVar(value = 1)
        frame_19 = InputForm_float(self, key = "Air_Y_Velocity", text_var = self.text_frame_19, default_value = self.default_value_19, fav_1_value = self.fav_value_19) #create a frame
        frame_19.grid(row = 24, column = 0, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_19)
        
        #-----------
        #Header Label 7 - Walls
        header_7 = ttk.Label(self, text = "Walls: if False then periodical condition", anchor = "center")
        header_7.grid(row = 0, column = 1, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 20 - Right Wall
        self.text_frame_20 = tk.StringVar(value = "Right? : boolean")
        self.default_value_20 = tk.BooleanVar(value = False)
        self.fav_value_20 = tk.BooleanVar(value = False)
        frame_20 = InputForm_bool(self, key = "Right_W", text_var = self.text_frame_20, default_value = self.default_value_20, fav_1_value = self.fav_value_20) #create a frame
        frame_20.grid(row = 1, column = 1, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_20)
        
        #frame 21 - Left Wall
        self.text_frame_21 = tk.StringVar(value = "Left? : boolean")
        self.default_value_21 = tk.BooleanVar(value = False)
        self.fav_value_21 = tk.BooleanVar(value = False)
        frame_21 = InputForm_bool(self, key = "Left_W", text_var = self.text_frame_21, default_value = self.default_value_21, fav_1_value = self.fav_value_21) #create a frame
        frame_21.grid(row = 2, column = 1, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_21)
        
        #frame 22 - Up Wall
        self.text_frame_22 = tk.StringVar(value = "Up? : boolean")
        self.default_value_22 = tk.BooleanVar(value = True)
        self.fav_value_22 = tk.BooleanVar(value = True)
        frame_22 = InputForm_bool(self, key = "Up_W", text_var = self.text_frame_22, default_value = self.default_value_22, fav_1_value = self.fav_value_22) #create a frame
        frame_22.grid(row = 3, column = 1, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_22)
        
        #frame 23 - Down Wall
        self.text_frame_23 = tk.StringVar(value = "Down? : boolean")
        self.default_value_23 = tk.BooleanVar(value = True)
        self.fav_value_23 = tk.BooleanVar(value = True)
        frame_23 = InputForm_bool(self, key = "Down_W", text_var = self.text_frame_23, default_value = self.default_value_23, fav_1_value = self.fav_value_23) #create a frame
        frame_23.grid(row = 4, column = 1, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_23)
    
        #-----------
        #Header Label 8 - Adhering Walls
        header_8 = ttk.Label(self, text = "Adhering Walls: only if the up wall exists:", anchor = "center")
        header_8.grid(row = 5, column = 1, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 24 - Up Wall Adhesion
        self.text_frame_24 = tk.StringVar(value = "Up Wall Adhering? : boolean")
        self.default_value_24 = tk.BooleanVar(value = True)
        self.fav_value_24 = tk.BooleanVar(value = True)
        frame_24 = InputForm_bool(self, key = "Up_A", text_var = self.text_frame_24, default_value = self.default_value_24, fav_1_value = self.fav_value_24) #create a frame
        frame_24.grid(row = 6, column = 1, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_24)
        
        #-----------
        #Header Label 9 - Plots
        header_9 = ttk.Label(self, text = "Plots saving", anchor = "center")
        header_9.grid(row = 7, column = 1, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 25 - Video Save
        self.text_frame_25 = tk.StringVar(value = "Save the video simulation? : boolean")
        self.default_value_25 = tk.BooleanVar(value = True)
        self.fav_value_25 = tk.BooleanVar(value = True)
        frame_25 = InputForm_bool(self, key = "Video", text_var = self.text_frame_25, default_value = self.default_value_25, fav_1_value = self.fav_value_25) #create a frame
        frame_25.grid(row = 8, column = 1, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_25)
        
        #frame 26 - End plot size
        self.text_frame_26 = tk.StringVar(value = "Full size end plot? no then zoomed  : boolean")
        self.default_value_26 = tk.BooleanVar(value = False)
        self.fav_value_26 = tk.BooleanVar(value = False)
        frame_26 = InputForm_bool(self, key = "End_Plot", text_var = self.text_frame_26, default_value = self.default_value_26, fav_1_value = self.fav_value_26) #create a frame
        frame_26.grid(row = 9, column = 1, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_26)
        
        #-----------
        #Header Label 10 - Proc
        header_10 = ttk.Label(self, text = "Number of procs", anchor = "center")
        header_10.grid(row = 10, column = 1, columnspan = 1, sticky = "ew", padx= 5, pady= (5, 10))
        
        #frame 27 - Number of processors
        self.text_frame_27 = tk.StringVar(value = " Number of processors : integer")
        self.default_value_27 = tk.IntVar(value = 6)
        self.fav_value_27 = tk.IntVar(value = 6)
        frame_27 = InputForm_int(self, key = "proc", text_var = self.text_frame_27, default_value = self.default_value_27, fav_1_value = self.fav_value_27) #create a frame
        frame_27.grid(row = 11, column = 1, sticky="nsew", padx=5, pady = 5)
        self.forms.append(frame_27) 
        
        #------------
        #button add all
        btn_add_all = ttk.Button(self, text = "Add All", command = self.add_all)
        btn_add_all.grid(row = 12, column = 1, sticky = "e", padx = 10, pady = 10)
        
        #button default all
        btn_default_all = ttk.Button(self, text = "Default All", command= self.def_all)
        btn_default_all.grid(row = 12, column = 1, sticky = "w", padx = 10, pady = 10)
        
        #button clear all
        btn_clear_all = ttk.Button(self, text = "Clear All", command= self.clear_all)
        btn_clear_all.grid(row = 12, column = 1, sticky = "", padx = 10, pady = 10)

        #button fav 1 all
        btn_fav_1_all = ttk.Button(self, text = "Favorite 1", command= self.fav_1_all)
        btn_fav_1_all.grid(row = 13, column = 1, sticky = "", padx = 10, pady = 10)
        
        
        #-----------
        #Header Label 11 - Check everything
        header_11 = ttk.Label(self, text = "Check everything is full before launching!", anchor = "center", background="#1F9F70")
        header_11.grid(row = 14, column = 1, columnspan = 1, rowspan = 2, sticky = "ew", padx= 5, pady= (5, 10))
        
        #Button launch
        btn_launch_sim = ttk.Button(self, text = "Launch Simulation", command = self.launch_sim)
        btn_launch_sim.grid(row = 15, column = 1,columnspan=1, sticky = "nswe", padx = 10, pady = 10)

        #-----------
        #Header Label 12 - Save favorites
        header_12 = ttk.Label(self, text = "Save your current set up as favorite?", anchor = "center", background="#1F9F70")
        header_12.grid(row = 16, column = 1, columnspan = 1, rowspan = 2, sticky = "ew", padx= 5, pady= (5, 10))
        
        #Button Save as Favorite 1
        btn_favorite_1 = ttk.Button(self, text = "Save as Favorite 1", command = self.save_fav_1)
        btn_favorite_1.grid(row = 17, column = 1, columnspan = 1, sticky = "nswe", padx = 10, pady = 10)
        
        
    def add_all(self):
        for form in self.forms:
            form.add_to_list()
    
    def def_all(self):
        for form in self.forms:
            form.default_to_entry()
        
    def clear_all(self):
        for form in self.forms:
            form.clear_list()
    
    def launch_sim(self):
        #collect input
        params = {}
        for form in self.forms:
            params[form.key] = form.get_value()
        
        #save to json
        out_path = Path("simulation_input.json")
        out_path.write_text(json.dumps(params, indent = 2))
        self.params = params
        self.destroy()
    
    def save_fav_1(self):
        #collect favorite 1 data
        favorite_1_data = {}
        for form in self.forms:
            favorite_1_data[form.key] = form.get_value()

        #save to json file
        out_path = Path("favorite_1_input.json")
        out_path.write_text(json.dumps(favorite_1_data, indent = 2))


    def fav_1_all(self):
        in_path = Path("favorite_1_input.json")
        # check if file exists
        if not in_path.exists():
            print("Favorite 1 file not found")
            return
        # read json
        favorite_1_data = json.loads(in_path.read_text())
        # populate the GUI fields
        for form in self.forms:
            if form.key in favorite_1_data:
                form.set_value(favorite_1_data[form.key]) 

        
 #imputing floats
class InputForm_float(ttk.Frame):
    def __init__(self, parent, key: str, text_var: tk.StringVar, default_value : tk.DoubleVar, fav_1_value : tk.DoubleVar):
        super().__init__(parent)
        self.key = key
        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight = 1)
        
        ##add a label
        #int values
        self.text_var = text_var
        self.default_value = default_value
        self.fav_1_value = fav_1_value
        self._initial_default = float(default_value.get()) #get the actual value not the pointer
        self._initial_fav_1 = float(fav_1_value.get())

        self.label = ttk.Label(self, textvariable = text_var)
        self.label.grid(row = 0, column = 0, sticky = "ew")

        self.entry = ttk.Entry(self, textvariable=self.default_value) #create en entry area
        self.entry.grid(row = 0, column = 1, sticky= "ew")

        #return is like add
        self.entry.bind("<Return>", self.add_to_list)
        
        self.text_list = tk.Listbox(self, height = 1)#text box area
        self.text_list.grid(row = 0, column= 2, rowspan=1, sticky="ew")
        #self.text_list.insert(tk.END, default_value)

        self.entry_btn = ttk.Button(self, text= "Add", command= self.add_to_list)#entry button
        self.entry_btn.grid(row=0, column= 3)
        
        self.entry_btn_2 = ttk.Button(self, text= "Clear", command= self.clear_list)#entry button
        self.entry_btn_2.grid(row=0, column= 4)

        
        
    def add_to_list(self, event = None ):
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
        
    def default_to_entry(self):
        self.default_value.set(self._initial_default)
        self.text_list.delete(0, tk.END)
        self.text_list.insert(tk.END, str(self._initial_default))

    def set_value(self, value):
        self.fav_1_value.set(value)
        self.text_list.delete(0, tk.END)
        self.text_list.insert(tk.END, str(value))
        
    def get_value(self) -> float:
        if self.text_list.size() > 0:
            return float(self.text_list.get(0))
        return 
    

#inputing ints
class InputForm_int(ttk.Frame,):
    def __init__(self, parent, key: str, text_var: tk.StringVar, default_value : tk.IntVar, fav_1_value : tk.IntVar):
        super().__init__(parent)
        self.key = key
        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight = 1)
        
        #add a label
        self.text_var = text_var
        self.default_value = default_value
        self.fav_1_value = fav_1_value
        self._initial_default = int(default_value.get())
        self._initial_fav_1 = int(fav_1_value.get())
        self.label = ttk.Label(self, textvariable = text_var)
        self.label.grid(row = 0, column = 0, sticky = "ew")

        self.entry = ttk.Entry(self, textvariable=self.default_value) #create en entry area
        self.entry.grid(row = 0, column = 1, sticky= "ew")
        
        #return is like add
        self.entry.bind("<Return>", self.add_to_list)
        
        self.text_list = tk.Listbox(self, height = 1)#text box area
        self.text_list.grid(row = 0, column= 2, rowspan=1, sticky="ew")

        self.entry_btn = ttk.Button(self, text= "Add", command= self.add_to_list)#entry button
        self.entry_btn.grid(row=0, column= 3)
        
        self.entry_btn_2 = ttk.Button(self, text= "Clear", command= self.clear_list)#entry button
        self.entry_btn_2.grid(row=0, column= 4)

        
        
    def add_to_list(self, event = None ):
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
        
    def default_to_entry(self):
        self.default_value.set(self._initial_default)
        self.text_list.delete(0, tk.END)
        self.text_list.insert(tk.END, str(self._initial_default))

    def set_value(self, value):
        self.fav_1_value.set(value)
        self.text_list.delete(0, tk.END)
        self.text_list.insert(tk.END, str(value))
        
    def get_value(self) -> int:
        if self.text_list.size() > 0:
            return int(self.text_list.get(0))
        return 
        
#imputing booleans      
class InputForm_bool(ttk.Frame):
    def __init__(self, parent, key: str, text_var: tk.StringVar, default_value : tk.BooleanVar, fav_1_value : tk.BooleanVar):
        super().__init__(parent)
        self.key = key
        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight = 1)
        
        #add a label
        self.text_var = text_var
        self.default_value = default_value
        self.fav_1_value = fav_1_value
        self._initial_default = bool(default_value.get())
        self._initial_fav_1 = bool(fav_1_value.get())
        self.label = ttk.Label(self, textvariable = text_var)
        self.label.grid(row = 0, column = 0, sticky = "ew")

        self.entry = ttk.Entry(self, textvariable=self.default_value) #create en entry area
        self.entry.grid(row = 0, column = 1, sticky= "ew")

        #return is like add
        self.entry.bind("<Return>", self.add_to_list)
        
        self.text_list = tk.Listbox(self, height = 1)#text box area
        self.text_list.grid(row = 0, column= 2, rowspan=1, sticky="ew")
        #self.text_list.insert(tk.END, default_value)

        self.entry_btn = ttk.Button(self, text= "Add", command= self.add_to_list)#entry button
        self.entry_btn.grid(row=0, column= 3)
        
        self.entry_btn_2 = ttk.Button(self, text= "Clear", command= self.clear_list)#entry button
        self.entry_btn_2.grid(row=0, column= 4)

        
        
    def add_to_list(self, event = None ):
        text = self.entry.get().strip() # strip takes the space away
        if not text:
            return
        
        if text in {"true", "t", "1", "yes", "True"}:
            text = True
        elif text in {"false", "f", "0", "no", "False"}:
            text = False
        else:
            messagebox.showerror("Boolean expected: True/ False / yes/ no")
            return
        
        #add the bool
        self.text_list.delete(0, tk.END)
        self.text_list.insert(tk.END, text)
        self.entry.delete(0, tk.END)
    
    def clear_list(self):
        self.text_list.delete(0, tk.END)   
        
    def default_to_entry(self):
        self.default_value.set(self._initial_default)
        self.text_list.delete(0, tk.END)
        self.text_list.insert(tk.END, str(self._initial_default))
    
    def set_value(self, value):
        self.fav_1_value.set(value)
        self.text_list.delete(0, tk.END)
        self.text_list.insert(tk.END, str(value))
          
    def get_value(self) -> bool:
        if self.text_list.size() > 0:
            value = str(self.text_list.get(0)).strip().lower()
            return value
        return 
                              
#running the script
if __name__ == "__main__":
    main()
    
def run_gui() -> dict:
    app = Application()
    app.mainloop()
    return app.params