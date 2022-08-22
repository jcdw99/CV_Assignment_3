from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

pic = Image.open('resources/lego1.jpg')


mm_to_pix = {
    "0,32,9.6": [854,1641], "0,32,19.2": [854,1569], "0,64,9.6": [720,1582], "0,96,9.6": [594,1533], "0,64,28.8": [721,1443], "144,0,9.6": [1885,1539],
    "0,0,9.6": [991,1696], "48,0,9.6": [1300,1645], "80,0,9.6": [1501,1610], "112,0,19.2": [1698,1503], "0,128,19.2": [471,1407], "112,0,38.4": [1700,1363], "0,128,38.4": [472,1270],
    "0,160,9.6": [355,1425], "0,0,9.6": [991,1698], "176,0,9.6": [2069,1506], "0,32,28.8": [722,1444], "0,32,38.4": [721,1371], "0,32,48.0": [721,1299], "0,32,57.6": [722,1227], "0,32,67.2": [722,1156],
    "0,32,76.8": [723,1082], "80,0,57.6": [1505,1250], "80,0,67.2": [1508,1176], "80,0,48.0": [1505,1322], "144,0,48.0": [1892,1259], "0,160,48.0": [355,1157], "0,96,57.6": [594,1179]
}

corners = {
    '0,0,124.8': [1000,800], '0,192,124.8':[248,580], '0,192,0':[244, 1448], '240,0,124.8':[2450,630], '240,0,0':[2424, 1510], '0,0,0':[992,1772]
}

def get_A_matrix(print_mode = False):

    x_s = []
    y_s = []
    z_s = []
    xprime = []
    yprime = []
    mm_to_pix = corners
    for key in mm_to_pix:
        key_data = key.split(",")
        x_s.append(float(key_data[0]))
        y_s.append(float(key_data[1]))
        z_s.append(float(key_data[2]))
        val_data = mm_to_pix.get(key)
        xprime.append(val_data[0])
        yprime.append(val_data[1])


    #TL TR BR BL
    x_primes = [xprime[i] for i in range(len(xprime))]
    y_primes = [yprime[i] for i in range(len(yprime))]
    col_1 = [x_s[i//2] if i % 2 == 0 else 0 for i in range(2*len(x_s))]
    col_2 = [y_s[i//2] if i % 2 == 0 else 0 for i in range(2*len(y_s))]
    col_3 = [z_s[i//2] if i % 2 == 0 else 0 for i in range(2*len(z_s))]
    col_4 = [1 if i % 2 == 0 else 0 for i in range(2*len(y_s))]
    col_5 = [x_s[i//2] if i % 2 != 0 else 0 for i in range(2*len(x_s))]
    col_6 = [y_s[i//2] if i % 2 != 0 else 0 for i in range(2*len(y_s))]
    col_7 = [z_s[i//2] if i % 2 != 0 else 0 for i in range(2*len(z_s))]
    col_8 = [1-i for i in col_4]
    col_9 = np.array([x_s[i//2] for i in range(2*len(x_s))]) * np.array([-x_primes[i//2] if i % 2 == 0 else -y_primes[i//2] for i in range(2*len(y_primes))])
    col_10 = np.array([y_s[i//2] for i in range(2*len(y_s))]) * np.array([-x_primes[i//2] if i % 2 == 0 else -y_primes[i//2] for i in range(2*len(y_primes))])
    col_11 = np.array([z_s[i//2] for i in range(2*len(z_s))]) * np.array([-x_primes[i//2] if i % 2 == 0 else -y_primes[i//2] for i in range(2*len(y_primes))])
    col_12 = [-x_primes[i//2] if i % 2 == 0 else -y_primes[i//2] for i in range(2*len(x_primes))]
    matrix = np.array([col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12]).transpose()
    if print_mode:
        df = pd.DataFrame(matrix)
        df.columns = [1,2,3,4,5,6,7,8,9,10,11,12]
        print(df)
    return matrix

def do_forward_map(world_coords):
    A = get_A_matrix()
    U, S, V = np.linalg.svd(A)
    H = (V[-1]).reshape((3,4))
    pixels = np.dot(H, np.array([world_coords[0], world_coords[1], world_coords[2], 1]))
    pixels = (pixels / pixels[2])[:-1]
    pixels = tuple(pixels.astype(int))
    return pixels

A = get_A_matrix(True)
# U, S, V = np.linalg.svd(A)
# H = (V[11]).reshape((3,4))

print(do_forward_map([16, 0, 76.8]))
