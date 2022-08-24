from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import



fig = plt.figure()
ax = plt.axes(projection='3d')

mm_to_pix_2 = {
   '0,160,0':[407,1613], '0,128,19.2':[623,1503], '0,64,38.4':[1074,1431], '0,32,19.2':[1310,1635], '0,64,57.6':[1077,1271], '0,96,96':[850,912],
   '0,128,96':[626,879],'0,32,9.6':[1310,1715],'0,16,48':[1434,1413],'0,32,96':[1320,980],'0,0,105.6':[1564,936],'0,0,115.2':[1566,850],'0,64,115.2':[1082,782],
   '0,128,105.6':[625,799],'16,0,28.8':[1621,1566],'64,0,9.6':[1813,1625],'96,0,19.2':[1941,1486],'144,0,19.2':[2116,1398],'144,0,76.8':[2128,948],'16,0,105.6':[1633,913],
   '208,0,105.6':[2352,641],'112,0,115.2':[2023,687],'48,0,115.2':[1772,776],'16,0,115.2':[1637,829],'240,0,9.6':[2430,1307],'240,0,115.2':[2456,531],'128,0,96':[2074,821],'192,0,96':[2296,737]
}

corners = {
    '0,0,124.8': [1000,800], '0,192,124.8':[248,580], '0,192,0':[244, 1448], '240,0,124.8':[2450,630], '240,0,0':[2424, 1510], '0,0,0':[992,1772]
}

def get_coords(ax_target):
    coords = []
    for key in mm_to_pix_2:
        coords.append(float(key.split(',')[ax_target]))
    return np.array(coords)

def plot_camera(ax):
    factor = 300
    camera1_loc = np.array([-687.14554115, -1034.97175416,   362.48207674])
    camera2_loc = np.array([-923.99874656, -514.65675181,  345.93903943])

    camera1_x = camera1_loc - factor*np.array([-0.82157973,  0.56998084, -0.01133987])
    camera1_y = camera1_loc - -factor*np.array([0.05759187,  0.1027705,   0.99303645])
    camera1_z = camera1_loc - factor*np.array([-0.56717716, -0.81520554,  0.11726042])

    camera2_x = camera2_loc - -factor*np.array([ 0.53175792, -0.84683533,  0.01017066])
    camera2_y = camera2_loc - factor*np.array([-0.13294922, -0.09533191, -0.98652741])
    camera2_z = camera2_loc - -factor*np.array([ 0.83639585,  0.52324158, -0.16327959])
    
    camsize = 80
    # plot camera1
    ax.scatter(camera1_loc[0], camera1_loc[1], camera1_loc[2], marker='d', color='r', label='Camera 1', s=camsize, alpha=0.6)
    # plot camera1 x
    ax.plot([camera1_loc[0], camera1_x[0]], [camera1_loc[1], camera1_x[1]], [camera1_loc[2], camera1_x[2]], color='green', label='x1')
    # plot camera1 y
    ax.plot([camera1_loc[0], camera1_y[0]], [camera1_loc[1], camera1_y[1]], [camera1_loc[2], camera1_y[2]], color='blue', label='y1')
    # plot camera1 z
    ax.plot([camera1_loc[0], camera1_z[0]], [camera1_loc[1], camera1_z[1]], [camera1_loc[2], camera1_z[2]], color='black', label='z1')


    # plot camera1
    ax.scatter(camera2_loc[0], camera2_loc[1], camera2_loc[2], marker='d', color='g', label='Camera 2', s=camsize, alpha=0.6)
    # plot camera1 x
    ax.plot([camera2_loc[0], camera2_x[0]], [camera2_loc[1], camera2_x[1]], [camera2_loc[2], camera2_x[2]], color='yellowgreen', label='x2')
    # plot camera1 y
    ax.plot([camera2_loc[0], camera2_y[0]], [camera2_loc[1], camera2_y[1]], [camera2_loc[2], camera2_y[2]], color='dodgerblue',label='y2')
    # plot camera1 z
    ax.plot([camera2_loc[0], camera2_z[0]], [camera2_loc[1], camera2_z[1]], [camera2_loc[2], camera2_z[2]], color='midnightblue', label='z2')



def plot_structure(ax):
    # a to b
    ax.plot([0,16],[192,192],[124.8,124.8], color='firebrick')
    # a to c
    ax.plot([0,0],[192,192],[124.8,0], color='firebrick')
    # b to d
    ax.plot([16,16],[192,192],[124.8,0], color='firebrick')
    # c to d
    ax.plot([0,16],[192,192],[0,0], color='firebrick')
    # c to e
    ax.plot([0,0],[192,0],[0,0], color='firebrick')
    # d to l
    ax.plot([16,16],[192,16],[0,0], color='firebrick')
    # l to k
    ax.plot([16,240],[16,16],[0,0], color='firebrick')
    # e to j
    ax.plot([0,240],[0,0],[0,0], color='firebrick')
    # k to j
    ax.plot([240,240],[16,0],[0,0], color='firebrick')
    # k to i
    ax.plot([240,240],[16,16],[0,124.8], color='firebrick')
    # j to h
    ax.plot([240,240],[0,0],[0,124.8], color='firebrick')
    # i to h
    ax.plot([240,240],[16,0],[124.8,124.8], color='firebrick')
    # h to f
    ax.plot([240,0],[0,0],[124.8,124.8], color='firebrick')
    # i to g    
    ax.plot([240,16],[16,16],[124.8,124.8], color='firebrick')
    # g to b
    ax.plot([16,16],[16,192],[124.8,124.8], color='firebrick')
    # f to a
    ax.plot([0,0],[0,192],[124.8,124.8], color='firebrick')
    # f to e
    ax.plot([0,0],[0,0],[124.8,0], color='firebrick')
    # g to l
    # ax.plot([16,16],[16,16],[124.8,0], color='r')

    # fill between a, b, g, i, h, f
    coords = [(0,192,124.8), (16,192,124.8), (16,16,124.8), (240,16,124.8), (240,0,124.8), (0,0,124.8)]
    ax.add_collection3d(Poly3DCollection([coords],color='firebrick', alpha=0.4)) 
    # fill between a, b, g, i, h, f
    coords = [(0,192,0), (16,192,0), (16,16,0), (240,16,0), (240,0,0), (0,0,0)]
    ax.add_collection3d(Poly3DCollection([coords],color='grey', alpha=0.4)) 
    # fill between a, f, h, j, e, c
    coords = [(0,192,124.8), (0,0,124.8), (240,0,124.8), (240,0,0), (0,0,0), (0,192,0)]
    ax.add_collection3d(Poly3DCollection([coords],color='firebrick', alpha=0.4)) 

ax.scatter3D(get_coords(0), get_coords(1), get_coords(2), s=3)
plot_structure(ax)
plot_camera(ax)
plt.legend()
plt.xlabel("X-Axis (mm)")
plt.ylabel("Y-Axis (mm)")

plt.show()