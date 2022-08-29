from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2



mm2_to_pix = {
   '0,160,0':[407,1613], '0,128,19.2':[623,1503], '0,64,38.4':[1074,1431], '0,32,19.2':[1310,1635], '0,64,57.6':[1077,1271], '0,96,96':[850,912],
   '0,128,96':[626,879],'0,32,9.6':[1310,1715],'0,16,48':[1434,1413],'0,32,96':[1320,980],'0,0,105.6':[1564,936],'0,0,115.2':[1566,850],'0,64,115.2':[1082,782],
   '0,128,105.6':[625,799],'16,0,28.8':[1621,1566],'64,0,9.6':[1813,1625],'96,0,19.2':[1941,1486],'144,0,19.2':[2116,1398],'144,0,76.8':[2128,948],'16,0,105.6':[1633,913],
   '208,0,105.6':[2352,641],'112,0,115.2':[2023,687],'48,0,115.2':[1772,776],'16,0,115.2':[1637,829],'240,0,9.6':[2430,1307],'240,0,115.2':[2456,531],'128,0,96':[2074,821],'192,0,96':[2296,737]
}

mm1_to_pix = {
    '0,0,124.8': [1000,800], '0,192,124.8':[248,580], '0,192,0':[244, 1448], '240,0,124.8':[2450,630], '240,0,0':[2424, 1510], '0,0,0':[992,1772]
}

def draw_on_1():
    pic1 = cv2.imread('resources/lego1.jpg')
    p1 = get_P_matrix(mm1_to_pix)

    orig = p1[:,3]
    x = 240*p1[:,0] + orig
    y = 192*p1[:,1] + orig
    z = 124.8*p1[:,2] + orig

    orig /= orig[2]; orig = orig[:-1]
    x /= x[2]; x = x[:-1]
    y /= y[2]; y = y[:-1]
    z /= z[2]; z = z[:-1]

   

    pic1 = cv2.arrowedLine(pic1, tuple(orig.astype(int)), tuple(x.astype(int)), (0,255,0), thickness=9)
    pic1 = cv2.arrowedLine(pic1, tuple(orig.astype(int)), tuple(z.astype(int)),  (255,0,0), thickness=9)
    pic1 = cv2.arrowedLine(pic1, tuple(orig.astype(int)), tuple(y.astype(int)), (0,0,255), thickness=9)

    pic1 = cv2.putText(pic1, 'X', tuple((np.array(x) - np.array([0, 20])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4, cv2.LINE_AA, False)
    pic1 = cv2.putText(pic1, 'Y', tuple((np.array(y) - np.array([0, 30])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, cv2.LINE_AA, False)
    pic1 = cv2.putText(pic1, 'Z', tuple((np.array(z) - np.array([0, 20])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA, False)

    pic1 = cv2.circle(pic1, tuple(orig.astype(int)), radius=20, color=(0, 0, 0), thickness=-1)
    pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
    Image.fromarray(pic1).show()


def draw_on_2():
    pic2 = cv2.imread('resources/lego2.jpg')
    p2 = get_P_matrix(mm2_to_pix)

    orig = p2[:,3]

    x = 240*p2[:,0] + orig
    y = 192*p2[:,1] + orig
    z = 124.8*p2[:,2] + orig

    orig /= orig[2]; orig = orig[:-1]
    x /= x[2]; x = x[:-1]
    y /= y[2]; y = y[:-1]
    z /= z[2]; z = z[:-1]

    pic2 = cv2.arrowedLine(pic2, tuple(orig.astype(int)), tuple(x.astype(int)), (0,255,0), thickness=9)
    pic2 = cv2.arrowedLine(pic2, tuple(orig.astype(int)), tuple(z.astype(int)),  (255,0,0), thickness=9)
    pic2 = cv2.arrowedLine(pic2, tuple(orig.astype(int)), tuple(y.astype(int)), (0,0,255), thickness=9)


    pic2 = cv2.putText(pic2, 'X', tuple((np.array(x) - np.array([0, 20])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4, cv2.LINE_AA, False)
    pic2 = cv2.putText(pic2, 'Y', tuple((np.array(y) - np.array([0, 30])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, cv2.LINE_AA, False)
    pic2 = cv2.putText(pic2, 'Z', tuple((np.array(z) - np.array([-80, -20])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA, False)


    pic2 = cv2.circle(pic2, tuple(orig.astype(int)), radius=20, color=(0, 0, 0), thickness=-1)
    pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2RGB)
    Image.fromarray(pic2).show()

def get_P_matrix(points):
    A = get_A_matrix(points)
    U, S, V = np.linalg.svd(A)
    P = (V[-1]).reshape((3,4))
    return P

def get_A_matrix(points, print_mode = False):
    x_s = []
    y_s = []
    z_s = []
    xprime = []
    yprime = []
    # mm_to_pix = corners
    for key in points:
        key_data = key.split(",")
        x_s.append(float(key_data[0]))
        y_s.append(float(key_data[1]))
        z_s.append(float(key_data[2]))
        val_data = points.get(key)
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



def do_forward_map(world_coords, points):
    P = get_P_matrix(points)

    pixels = np.dot(P, np.array([world_coords[0], world_coords[1], world_coords[2], 1]))
    pixels = (pixels / pixels[2])[:-1]
    pixels = tuple(pixels.astype(int))
    return pixels
    

draw_on_1()
