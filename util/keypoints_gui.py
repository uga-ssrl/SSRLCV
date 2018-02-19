from Tkinter import *
from PIL import ImageTk, Image
import os
import Tkinter as tk
import shutil
import math
import numpy as np

####

global scan_lines_loaded
global scan_lines_current
scan_lines_loaded  = []
scan_lines_current = []

camera_position = [6, 8, 6.5]
camera_unit     = [-0.48,-0.64,-0.6] # this can be a constant?

verts  = [[-1, 1,-1], # 0 A
          [ 1, 1,-1], # 1 B
          [ 1,-1,-1], # 2 C
          [-1,-1,-1], # 3 D
          [-1, 1, 1], # 4 E
          [ 1, 1, 1], # 5 F
          [ 1,-1, 1], # 6 G
          [-1,-1, 1]] # 7 H

# AB, BC, CD, DA, EF, FG, GH, HE, AE, BF, CG, and DH.
edges  = [[0,1], # AB
          [1,2], # BC
          [2,3], # CD
          [3,0], # DA
          [4,5], # EF
          [5,6], # FG
          [6,7], # GH
          [7,4], # HE
          [0,4], # AE
          [1,5], # BF
          [2,6], # CG
          [3,7]] # DH

window_size = 1.0 #cm
view_dist   = 1.0 #cm
screen_size = 1024 #pixs

cm_per_pix  = window_size/screen_size

####

file_name = 'bolbi.jpg'
image = Image.open(file_name)
shutil.copy(file_name,file_name + '.temp.jpg')
image_temp = Image.open(file_name + '.temp.jpg')

root = tk.Tk()

imgFrame = Frame(width=screen_size, height=screen_size+200)
imgFrame.pack()

img = ImageTk.PhotoImage(image)
panel = tk.Label(imgFrame, image=img)
panel.pack(side="top", fill="both", expand="yes")


####

def brz(x0, y0, x1, y1, r, g, b):
    print "Bresenham's line algorithm"
    pixels = image_temp.load()

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            try:
                pixels[x,y] = (r,g,b)
            except Exception:
                print 'pixels ' + str(x) + ' ' + str(y) + ' out of bounds'
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            try:
                pixels[x,y] = (r,g,b)
            except Exception:
                print 'pixels ' + str(x) + ' ' + str(y) + ' out of bounds'
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    try:
        pixels[x,y] = (r,g,b)
    except Exception:
        print 'pixels ' + str(x) + ' ' + str(y) + ' out of bounds'

    wline = ImageTk.PhotoImage(image_temp)
    panel.configure(image=wline)
    panel.image = wline

def updatescanlines():
    global scan_lines_current
    global image
    global image_temp
    image = Image.open(file_name)
    shutil.copy(file_name,file_name + '.temp.jpg')
    image_temp = Image.open(file_name + '.temp.jpg')
    img3 = ImageTk.PhotoImage(image_temp)
    panel.configure(image=img3)
    panel.image = img3
    print 'updating scanlines... '
    print scan_lines_current
    for points in scan_lines_current:
        x_1 = int(points[0])
        y_1 = int(points[1])
        x_2 = int(points[2])
        y_2 = int(points[3])
        brz(x_1,y_1,x_2,y_2, 255, 0, 0)
    return 0

def print_mat(M):
    line = ''
    for v in M:
        line = line + '|'
        for x in v:
            line = line + '\t' + str(x) + ', '
        line = line + '|'
        print line
        line = ''
    print ''

def matrix_multiply(X, Y, col_num):
    if (col_num == 1):
        result = [[0.0,0.0,0.0,0.0]]
        for i in range(0,4):
            result[0][0] += X[0][i]*Y[i][0]
        for i in range(0,4):
            result[0][1] += X[0][i]*Y[i][0]
        for i in range(0,4):
            result[0][2] += X[0][i]*Y[i][0]
        for i in range(0,4):
            result[0][3] += X[0][i]*Y[i][0]
    else:
        result = [[0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0]]

        result[0][0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0] + X[0][2]*Y[2][0] + X[0][3]*Y[3][0]
        result[0][1] = X[0][0]*Y[0][1] + X[0][1]*Y[1][1] + X[0][2]*Y[2][1] + X[0][3]*Y[3][1]
        result[0][2] = X[0][0]*Y[0][2] + X[0][1]*Y[1][2] + X[0][2]*Y[2][2] + X[0][3]*Y[3][2]
        result[0][3] = X[0][0]*Y[0][3] + X[0][1]*Y[1][3] + X[0][2]*Y[2][3] + X[0][3]*Y[3][3]

        result[1][0] = X[1][0]*Y[0][0] + X[1][1]*Y[1][0] + X[1][2]*Y[2][0] + X[1][3]*Y[3][0]
        result[1][1] = X[1][0]*Y[0][1] + X[1][1]*Y[1][1] + X[1][2]*Y[2][1] + X[1][3]*Y[3][1]
        result[1][2] = X[1][0]*Y[0][2] + X[1][1]*Y[1][2] + X[1][2]*Y[2][2] + X[1][3]*Y[3][2]
        result[1][3] = X[1][0]*Y[0][3] + X[1][1]*Y[1][3] + X[1][2]*Y[2][3] + X[1][3]*Y[3][3]

        result[2][0] = X[2][0]*Y[0][0] + X[2][1]*Y[1][0] + X[2][2]*Y[2][0] + X[2][3]*Y[3][0]
        result[2][1] = X[2][0]*Y[0][1] + X[2][1]*Y[1][1] + X[2][2]*Y[2][1] + X[2][3]*Y[3][1]
        result[2][2] = X[2][0]*Y[0][2] + X[2][1]*Y[1][2] + X[2][2]*Y[2][2] + X[2][3]*Y[3][2]
        result[2][3] = X[2][0]*Y[0][3] + X[2][1]*Y[1][3] + X[2][2]*Y[2][3] + X[2][3]*Y[3][3]

        result[3][0] = X[3][0]*Y[0][0] + X[3][1]*Y[1][0] + X[3][2]*Y[2][0] + X[3][3]*Y[3][0]
        result[3][1] = X[3][0]*Y[0][1] + X[3][1]*Y[1][1] + X[3][2]*Y[2][1] + X[3][3]*Y[3][1]
        result[3][2] = X[3][0]*Y[0][2] + X[3][1]*Y[1][2] + X[3][2]*Y[2][2] + X[3][3]*Y[3][2]
        result[3][3] = X[3][0]*Y[0][3] + X[3][1]*Y[1][3] + X[3][2]*Y[2][3] + X[3][3]*Y[3][3]


    return result

def find_proj_matrix():
    # Translate the coordinate system
    T_1 = [[ 1.0, 0.0, 0.0, 0.0],
           [ 0.0, 1.0, 0.0, 0.0],
           [ 0.0, 0.0, 1.0, 0.0],
           [-1.0 * camera_position[0], -1.0 * camera_position[1], -1.0 * camera_position[2], 1.0]]
    print_mat(T_1)
    T_2 = [[1.0,0.0,0.0,0.0],
           [0.0,0.0,-1.0,0.0],
           [0.0,1.0,0.0,0.0],
           [0.0,0.0,0.0,1.0]]
    print_mat(T_2);
    # rotate around the y-axis
    cos_t3 = camera_position[1]/ math.sqrt(camera_position[0]**2.0 + camera_position[1]**2.0)
    print 'cos: ' +str(cos_t3)
    sin_t3 = camera_position[0]/ math.sqrt(camera_position[0]**2.0 + camera_position[1]**2.0)
    print 'sin: ' + str(sin_t3) + '\n'
    T_3 = [[-1* cos_t3, 0.0, sin_t3, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [-1* cos_t3, 0.0, -1*sin_t3, 0.0],
           [0.0,0.0,0.0,1.0]]
    print_mat(T_3)
    # rotate even more!
    cos_t4 = math.sqrt(camera_position[0]**2.0 + camera_position[1]**2.0)/math.sqrt(camera_position[0]**2.0 + camera_position[1]**2 + camera_position[2]**2.0 )
    print 'cos: ' +str(cos_t4)
    sin_t4 = camera_position[2]/math.sqrt(camera_position[0]**2.0 + camera_position[1]**2.0 + camera_position[2]**2.0 )
    print 'sin: ' + str(sin_t4) + '\n'
    T_4 = [[1.0, 0.0, 0.0, 0.0],
           [0.0, cos_t4, sin_t4, 0.0],
           [0.0, -1*sin_t4, cos_t4,0.0],
           [0.0,0.0,0.0,1.0]]
    print_mat(T_4)
    # final transform matrix
    T_5 = [[1.0,0.0,0.0,0.0],
           [0.0,1.0,0.0,0.0],
           [0.0,0.0,-1.0,0.0],
           [0.0,0.0,0.0,1.0]]
    print_mat(T_5);

    #### Multiply all the T's together
    print '======V======'
    #V = matrix_multiply(matrix_multiply(matrix_multiply(matrix_multiply(T_1, T_2,4), T_3,4), T_4,4), T_5,4)
    t1 = np.mat(T_1)
    t2 = np.mat(T_2)
    t3 = np.mat(T_3)
    t4 = np.mat(T_4)
    t5 = np.mat(T_5)
    temp1 = np.matmul(t1, t2)
    temp2 = np.matmul(temp1, t3)
    temp3 = np.matmul(temp2, t4)
    V = np.matmul(temp3, t5)
    print V
    # V_test = matrix_multiply(matrix_multiply(matrix_multiply(T_2, T_3,4), T_4,4), T_5,4)
    #print_mat(V)

    ##### N matrix time
    print '======N======'
    N   = [[view_dist/(window_size/2.0),0.0,0.0,0.0],
           [0.0,view_dist/(window_size/2.0),0.0,0.0],
           [0.0,0.0,1.0,0.0],
           [0.0,0.0,0.0,1.0]]
    N = np.mat(N)
    print N
    #print_mat(N);

    ##### Both T and N
    print '======V*N======'
    #V_N = matrix_multiply(V,N,4)
    V_N = np.matmul(V, N)
    print V_N
    #print_mat(V_N)

    #print '======(v correct)======'
    # VN_manual = [[-3.2, -1.4, -0.5, 0.0],
    #             [ 2.4, -1.9, -0.6, 0.0],
    #             [ 0.0,  3.2, -0.6, 0.0],
    #             [ 0.0,  0.0, 12.5, 1.0]]

    # R = matrix_multiply(V_test,N,4)
    # print_mat(R)

    print '=======attempt======='
    new_guy = []
    for point in verts:
        # TODO add a transformation matrix in here
        p = np.mat([point[0],point[1],point[2],1.0])
        coords = np.mat(np.matmul(p,V_N))
        # get screen point here:
        x_s = (coords.item(0)/coords.item(2))*screen_size + screen_size
        y_s = (coords.item(1)/coords.item(2))*screen_size + screen_size
        #print '(' + str(x_s) + ',' + str(y_s) + ')'
        print str(p) + ' -> ' + str(coords) + '\t : : \t' + '(' + str(x_s) + ',' + str(y_s) + ')'
        new_guy.append([x_s,y_s])
    global scan_lines_current
    scan_lines_current = []
    for pair in edges:
        new_dude = [ new_guy[pair[0]][0],  new_guy[pair[0]][1],  new_guy[pair[1]][0],  new_guy[pair[1]][1] ]
        print new_dude
        scan_lines_current.append([ new_guy[pair[0]][0],  new_guy[pair[0]][1],  new_guy[pair[1]][0],  new_guy[pair[1]][1] ])


########## just to run the GUI at this point

### load frame
loadFrame = Frame(width=screen_size)
loadFrame.pack()

def redraw():
    print 'drawing...'
    find_proj_matrix()
    updatescanlines()


btnEnter = Button(loadFrame, text='Redraw / Load', command=redraw)
btnEnter.pack(side=LEFT)


######### Steg Frame

stegFrame = Frame(width=screen_size)
stegFrame.pack()

trans = Entry(stegFrame, text='trans')
trans.insert(END,'translate')
trans.pack(side=LEFT)

def px():
    print 'x++'
    for coord in verts:
        coord[0] += float(trans.get())
    redraw()

def nx():
    print 'x--'
    for coord in verts:
        coord[0] -= float(trans.get())
    redraw()

def py():
    print 'y++'
    for coord in verts:
        coord[1] += float(trans.get())
    redraw()

def ny():
    print 'y--'
    for coord in verts:
        coord[1] -= float(trans.get())
    redraw()

def pz():
    print 'z++'
    for coord in verts:
        coord[2] += float(trans.get())
    redraw()

def nz():
    print 'z--'
    for coord in verts:
        coord[2] -= float(trans.get())
    redraw()


btnUp = Button(stegFrame, text='+x', command=px)
btnUp.pack(side=LEFT)

btnDn = Button(stegFrame, text='-x', command=nx)
btnDn.pack(side=LEFT)

btnLf = Button(stegFrame, text='+y', command=py)
btnLf.pack(side=LEFT)

btnRt = Button(stegFrame, text='-y', command=ny)
btnRt.pack(side=LEFT)

btnLf = Button(stegFrame, text='+z', command=pz)
btnLf.pack(side=LEFT)

btnRt = Button(stegFrame, text='-z', command=nz)
btnRt.pack(side=LEFT)

### rot frame
asdfFrame = Frame(width=screen_size)
asdfFrame.pack()

asdfx = Entry(asdfFrame, text='')
asdfx.insert(END,'0.0')
asdfx.pack(side=LEFT)
asdfy = Entry(asdfFrame, text='')
asdfy.insert(END,'0.0')
asdfy.pack(side=LEFT)
asdfz = Entry(asdfFrame, text='')
asdfz.insert(END,'0.0')
asdfz.pack(side=LEFT)

rotFrame = Frame(width=screen_size)
rotFrame.pack()

angle = Entry(rotFrame, text='angle')
angle.insert(END,'angle')
angle.pack(side=LEFT)

def rx():
    r   = [[1.0,0.0,0.0],
           [0.0,math.cos(math.radians(float(angle.get()))),-1.0*math.sin(math.radians(float(angle.get())))],
           [0.0,math.sin(math.radians(float(angle.get()))),math.cos(math.radians(float(angle.get())))]]
    r = np.mat(r)
    for coord in verts:
        coord[0] -= float(asdfx.get())
        coord[1] -= float(asdfy.get())
        coord[2] -= float(asdfz.get())
        x = np.mat(coord)
        y = np.matmul(x,r)
        print y
        coord[0] = y.item(0) + float(asdfx.get())
        coord[1] = y.item(1) + float(asdfy.get())
        coord[2] = y.item(2) + float(asdfz.get())
    redraw()

def ry():
    r   = [[math.cos(math.radians(float(angle.get()))),0.0,math.sin(math.radians(float(angle.get())))],
           [0.0,1.0,0.0],
           [-1.0*math.sin(math.radians(float(angle.get()))),0.0,math.cos(math.radians(float(angle.get())))]]
    r = np.mat(r)
    for coord in verts:
        coord[0] -= float(asdfx.get())
        coord[1] -= float(asdfy.get())
        coord[2] -= float(asdfz.get())
        x = np.mat(coord)
        y = np.matmul(x,r)
        print y
        coord[0] = y.item(0) + float(asdfx.get())
        coord[1] = y.item(1) + float(asdfy.get())
        coord[2] = y.item(2) + float(asdfz.get())
    redraw()

def rz():
    r   = [[math.cos(math.radians(float(angle.get()))),-1.0*math.sin(math.radians(float(angle.get()))),0.0],
           [math.sin(math.radians(float(angle.get()))),math.cos(math.radians(float(angle.get()))),0.0],
           [0.0,0.0,1.0]]
    r = np.mat(r)
    for coord in verts:
        coord[0] -= float(asdfx.get())
        coord[1] -= float(asdfy.get())
        coord[2] -= float(asdfz.get())
        x = np.mat(coord)
        y = np.matmul(x,r)
        print y
        coord[0] = y.item(0) + float(asdfx.get())
        coord[1] = y.item(1) + float(asdfy.get())
        coord[2] = y.item(2) + float(asdfz.get())
    redraw()

btnrx = Button(rotFrame, text='rotate x', command=rx)
btnrx.pack(side=LEFT)

btnry = Button(rotFrame, text='rotate y', command=ry)
btnry.pack(side=LEFT)

btnrz = Button(rotFrame, text='rotate z', command=rz)
btnrz.pack(side=LEFT)

### scale frame
scaleFrame = Frame(width=screen_size)
scaleFrame.pack()

scaler = Entry(scaleFrame, text='scaler')
scaler.insert(END,'scaler')
scaler.pack(side=LEFT)

def sx():
    print 'scaling x'
    for coord in verts:
        coord[0] *= float(scaler.get())
    redraw()

def sy():
    print 'scaling y'
    for coord in verts:
        coord[1] *= float(scaler.get())
    redraw()

def sz():
    print 'scaling z'
    for coord in verts:
        coord[2] *= float(scaler.get())
    redraw()

btnsx = Button(scaleFrame, text='scale x', command=sx)
btnsx.pack(side=LEFT)

btnsy = Button(scaleFrame, text='scale y', command=sy)
btnsy.pack(side=LEFT)

btnsz = Button(scaleFrame, text='scale z', command=sz)
btnsz.pack(side=LEFT)

#########
root.bind("<Return>", redraw)
root.mainloop()
