#!/usr/bin/python3
# IMPORTS:
from tkinter import *
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk

# GLOBALS:
main_window = Tk()

# methods
def print_welcome(version):
    print("==============================================")
    print("=                                            =")
    print("=         UGA SSRL Computer Vision           =")
    print("=             Version: " + version + "                 =")
    print("=         Smallsat.uga.edu/research          =")
    print("=                                            =")
    print("==============================================")

# classes
# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):

        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        #reference to the master widget, which is the tk window
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("UGA SSRL Computer Vision")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a menu instance
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # ======================================================================
        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Load Images", command=self.client_load_imgs)
        file.add_command(label="clear Images", command=self.client_clear_imgs)
        file.add_command(label="Save Features", command=self.client_save_features)
        file.add_command(label="Save Matches", command=self.client_save_matches)
        file.add_separator()
        file.add_command(label="Exit", command=self.client_exit)

        #added "file" to our menu
        menu.add_cascade(label="File", menu=file)

        # ======================================================================
        features = Menu(menu)

        features.add_command(label="Generate Dense SIFT", command=self.client_dsift)
        features.add_command(label="Generate SIFT", command=self.client_sift)
        features.add_separator()
        features.add_command(label="Generate Dense SURF", command=self.client_dsurf)
        features.add_command(label="Generate SURF", command=self.client_surf)
        features.add_separator()
        features.add_command(label="Generate HARRIS", command=self.client_harris)
        features.add_separator()
        features.add_command(label="Generate FAST", command=self.client_fast)

        menu.add_cascade(label="Features", menu=features)

        # ======================================================================
        matching = Menu(menu)

        matching.add_command(label="Brute Force",command=self.client_brute_force)
        matching.add_separator()
        matching.add_command(label="Filter: Distance Cutoff",command=self.client_distance_cut)
        matching.add_command(label="filter: Top Percentile",command=self.client_top_percentile)

        menu.add_cascade(label="Matching", menu=matching)

        # ======================================================================
        # create the file object)
        about = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        about.add_command(label="About", command=self.client_about)

        #added "file" to our menu
        menu.add_cascade(label="About", menu=about)

    # ==========================================================================
    # file menu

    def client_exit(self):
        print("Exiting...")
        exit()

    def client_load_imgs(self):
        print("Loading Images...")

    def client_clear_imgs(self):
        print("Clearing Images...")
        messagebox.showerror("ERROR", "There no images to clear")

    def client_save_features(self):
        print("Saving Features...")
        messagebox.showerror("ERROR", "There are no features to save")

    def client_save_matches(self):
        print("Saving Matches...")
        messagebox.showerror("ERROR", "There are no matches to save")

    # ==========================================================================
    # matches menu

    def client_brute_force(self):
        print("TODO")

    def client_distance_cut(self):
        print("TODO")

    def client_top_percentile(self):
        print("TODO")

    # ==========================================================================
    # features menu

    def client_dsift(self):
        print("TODO")

    def client_sift(self):
        print("TODO")

    def client_dsurf(self):
        print("TODO")

    def client_surf(self):
        print("TODO")

    def client_harris(self):
        print("TODO")

    def client_fast(self):
        print("TODO")

    # ==========================================================================
    #about

    def client_about(self):
        jackson = "Jackson Parker - something@something.com"
        caleb = "Caleb Adams - CalebAshmoreAdams@gmail.com"
        messagebox.showinfo("ABOUT", "Authors: \n\n Feature Detection: \n" + jackson + "\n\n 3D reconstruction: \n" + caleb + "\n\n GUI: \n" + caleb + "\n\n Surface reconstruction \n" + jackson)


# main
if __name__ == "__main__":
    print_welcome("0.0.1")
    main_window.geometry("800x600");
    instance = Window(main_window);
    main_window.mainloop()








































# ==
