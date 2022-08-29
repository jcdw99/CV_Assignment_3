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

def do_map(p, coords):
    pixels = np.dot(p, np.array([coords[0], coords[1], coords[2], 1]))

    pixels = (pixels / pixels[2])[:-1]
    pixels = tuple(pixels.astype(int))
    return pixels

def get_plus_mat(mat):
    return np.linalg.pinv(mat)

def dehomogon(x):
    return (x/x[2])[:-1]
    
if __name__ == "__main__":
    P = get_P_matrix(mm1_to_pix)
    P_plus = get_plus_mat(P)
    P_prime = get_P_matrix(mm2_to_pix)
    P_prime_plus = get_plus_mat(P_prime)

    C = np.array([-687.14554115, -1034.97175416,   362.48207674])
    C_prime = np.array([-923.99874656, -514.65675181,  345.93903943])

    C = np.append(C, 1)
    C_prime = np.append(C_prime, 1)

    e_prime = np.dot(P_prime, C)
    e = np.dot(P, C_prime)

  
    F = np.dot(P_prime, P_plus)

    packed_e = np.array([
        [0, -e_prime[2], e_prime[1]],
        [e_prime[2], 0, -e_prime[0]],
        [-e_prime[1], e_prime[0], 0],
    ])


    F = np.dot(packed_e, F)
 
    def get_img2_line(point):

        l_prime = np.dot(F, np.array([point[0],point[1], 1]))

        a = l_prime[0]
        b = l_prime[1]
        c = l_prime[2]

        point1 = (int(-100), int(-100*(-a/b) - c/b))
        point2 = (int(10000), int(10000*(-a/b) - c/b))
        return point1, point2

    def get_img1_line(point):
        l = np.cross(np.array([point[0],point[1], 1]), e)
        a = l[0]
        b = l[1]
        c = l[2]
        point1 = (int(-100), int(-100*(-a/b) - c/b))
        point2 = (int(10000), int(10000*(-a/b) - c/b))
        return point1, point2

    pic2 = cv2.imread('resources/lego2.jpg')
    pic1 = cv2.imread('resources/lego1.jpg')
    
    steps = [(998,800), (996,875), (997,951), (995,1027), (995,1102), (994,1178), (995,1253), (992,1328), (992,1402), (992,1478), (992,1551), (992,1625), (992,1698), (991,1771)]
    colours = [tuple(np.random.randint(0, 255, size=(3, )).astype(int)) for i in range(len(steps))]
    for i in range(len(steps)):
        color = ( int (colours[i][ 0 ]), int (colours[i][ 1 ]), int (colours[i][ 2 ])) 
        feature = (steps[i][0], steps[i][1])
        point1, point2 = get_img2_line(feature)
        pic2 = cv2.line(pic2, point1, point2, color=color, thickness=9)

    for i in range(len(steps)):
        color = ( int (colours[i][ 0 ]), int (colours[i][ 1 ]), int (colours[i][ 2 ])) 
        feature = (steps[i][0], steps[i][1])
        point1, point2 = get_img1_line(feature)
        pic1 = cv2.line(pic1, point1, point2, color=color, thickness=9)
        
    pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2RGB)    
    pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)

    Image.fromarray(pic2).show()
    Image.fromarray(pic1).show()



