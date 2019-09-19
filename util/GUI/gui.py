#!/usr/bin/python3
# IMPORTS:
import os
import glob
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk

# GLOBALS:
main_window = Tk()
IMG_EXIST = False
IMG_POS = [0,0]
IMG_LIST = []
IMG_FILE_LIST = []
FEATURE_EXIST = False
MTACHES_EXIST = False

# methods
def print_welcome(version):
    print("==============================================")
    print("=                                            =")
    print("=         UGA SSRL Computer Vision           =")
    print("=            GUI Version: " + version + "              =")
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
        file.add_command(label="Load Image", command=self.client_load_img)
        file.add_command(label="Load Image Folder", command=self.client_load_img_folder)
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
        pointcloud = Menu(menu)

        pointcloud.add_command(label="Stereo Disparity", command=self.client_stereo_disp)
        pointcloud.add_command(label="2 View Reproject", command=self.client_2view)
        pointcloud.add_command(label="N View Reproject", command=self.client_nview)
        pointcloud.add_command(label="Bundle Adjustment", command=self.client_bundle_adjust)

        menu.add_cascade(label="PointCloud", menu=pointcloud)

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

    def client_load_img(self):
        filename =  filedialog.askopenfile(initialdir = "~",title = "Select Single Image",filetypes = (("all files","*.*"),("jpeg files","*.jpg"),("png files","*.png")))
        if (filename != None):
            if (str(filename.name).lower()[-3:] == 'jpg' or str(filename.name).lower()[-3:] == 'png' or str(filename.name).lower()[-4:] == 'jpeg'):
                print("loading: " + str(filename.name) + " ...")
                self.showImg(str(filename.name))
            else:
                messagebox.showerror("ERROR", "Not a valid file extension")

    def client_load_img_folder(self):
        directory = filedialog.askdirectory(initialdir = "~",title = "Select Image Folder")
        if (directory != None):
            print("")

    def client_clear_imgs(self):
        global IMG_EXIST, IMG_POS, IMG_LIST, IMG_FILE_LIST
        print("Clearing Images...")
        if (IMG_EXIST):
            for i in IMG_LIST:
                i.place_forget()
            IMG_EXIST = False
            IMG_LIST = []
            IMG_POS = [0,0]
            IMG_FILE_LIST = []
        else:
            messagebox.showerror("ERROR", "There no images to clear")

    def client_save_features(self):
        print("Saving Features...")
        messagebox.showerror("ERROR", "There are no features to save")
        # filename =  filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

    def client_save_matches(self):
        print("Saving Matches...")
        messagebox.showerror("ERROR", "There are no matches to save")
        # filename =  filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

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
    # point cloud menu

    def client_stereo_disp(self):
        print("TODO")

    def client_2view(self):
        print("TODO")

    def client_nview(self):
        print("TODO")

    def client_bundle_adjust(self):
        print("TODO")

    # ==========================================================================
    # about

    def client_about(self):
        jackson = "Jackson Parker - something@something.com"
        caleb = "Caleb Adams - CalebAshmoreAdams@gmail.com"
        messagebox.showinfo("ABOUT", "Authors: \n\n Feature Detection: \n" + jackson + "\n\n 3D reconstruction: \n" + caleb + "\n\n GUI: \n" + caleb + "\n\n Surface reconstruction \n" + jackson)

    # ==========================================================================
    # ==========================================================================
    # ==========================================================================
    # ==========================================================================
    # ==========================================================================
    # non-menu methods

    def showImg(self,image_path):
        global IMG_EXIST, IMG_POS, IMG_LIST, IMG_FILE_LIST
        # all stubs are 100x100
        load = Image.open(image_path)
        if (load.size[0] != load.size[1]):
            if (load.size[0] > load.size[1]):
                scale = load.size[0] / 100
                load = load.resize((int(load.size[0]/scale),int(load.size[1]/scale)),Image.ANTIALIAS)
            else:
                scale = load.size[1] / 100
                load = load.resize((int(load.size[0]/scale),int(load.size[1]/scale)),Image.ANTIALIAS)
        else:
            load = load.resize((100,100),Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=IMG_POS[0], y=IMG_POS[1])
        if (IMG_POS[1] == 700):
            IMG_POS[1] += 100
            IMG_POS[0] = 0
        else:
            IMG_POS[0] += 100
        IMG_EXIST = True
        IMG_LIST.append(img)
        IMG_FILE_LIST.append(image_path)



# main
if __name__ == "__main__":
    print_welcome("0.0.1")
    main_window.geometry("800x600");
    instance = Window(main_window);
    main_window.mainloop()








































# ==
