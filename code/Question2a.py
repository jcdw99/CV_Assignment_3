from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2


pic = Image.open('resources/lego2.jpg')

mm_to_pix = {
   '0,160,0':[407,1613], '0,128,19.2':[623,1503], '0,64,38.4':[1074,1431], '0,32,19.2':[1310,1635], '0,64,57.6':[1077,1271], '0,96,96':[850,912],
   '0,128,96':[626,879],'0,32,9.6':[1310,1715],'0,16,48':[1434,1413],'0,32,96':[1320,980],'0,0,105.6':[1564,936],'0,0,115.2':[1566,850],'0,64,115.2':[1082,782],
   '0,128,105.6':[625,799],'16,0,28.8':[1621,1566],'64,0,9.6':[1813,1625],'96,0,19.2':[1941,1486],'144,0,19.2':[2116,1398],'144,0,76.8':[2128,948],'16,0,105.6':[1633,913],
   '208,0,105.6':[2352,641],'112,0,115.2':[2023,687],'48,0,115.2':[1772,776],'16,0,115.2':[1637,829],'240,0,9.6':[2430,1307],'240,0,115.2':[2456,531],'128,0,96':[2074,821],'192,0,96':[2296,737]
}

def img_draw():
    image = cv2.imread('resources/lego2.jpg')
    counter = 0
    times = 0

    origin = (1550, 1840)
    x_ax = (1747, 1739)
    z_ax = (1555, 1435)
    y_ax = (1189, 1769)
    image = cv2.arrowedLine(image, origin, x_ax, (255,0,0), 9)
    image = cv2.putText(image, 'X', tuple(np.array(x_ax) - np.array([0, 20])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA, False)
    image = cv2.arrowedLine(image, origin, y_ax, (255,0,0), 9)
    image = cv2.putText(image, 'Y', tuple(np.array(y_ax) - np.array([0, 20])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA, False)
    image = cv2.arrowedLine(image, origin, z_ax, (255,0,0), 9)
    image = cv2.putText(image, 'Z', tuple(np.array(z_ax) - np.array([0, 20])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA, False)
    
    image = cv2.circle(image, origin, radius=18, color=(0, 0, 0), thickness=-1)
    for val in mm_to_pix:
        image = cv2.circle(image, tuple(mm_to_pix.get(val)), radius=13, color=(0, 0, 0), thickness=-1)
        image = cv2.circle(image, tuple(mm_to_pix.get(val)), radius=8, color=(0, 0, 255), thickness=-1)
        base = 'A' if times > 0 else ''
        image = cv2.putText(image, base + chr(ord('A') + counter), tuple(np.array(mm_to_pix.get(val)) + np.array([-7, -20])), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (0,0,0), 6, cv2.LINE_AA, False)
        image = cv2.putText(image, base + chr(ord('A') + counter), tuple(np.array(mm_to_pix.get(val)) + np.array([-7, -20])), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (255,255,255), 2, cv2.LINE_AA, False)
        counter = counter + 1
        if counter >= 26:
            counter = 0
            times = times + 1

    start = (185, 74)
    space = (0, 55)
    per = 7
    counter = 0
    letter_counter = 0
    times = 0
    offset = 55

    for val in mm_to_pix:
        if counter % per == 0 and counter > 0:
            start = tuple(np.array(start) + np.array([550,0]))
        base = 'A' if times > 0 else ''
        letter = base + chr(ord('A') + letter_counter)
        image = cv2.putText(image, letter + ': ', tuple(np.array(start) + (counter % per) * np.array(space)), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                 (0,255,0), 2, cv2.LINE_AA, False)
        image = cv2.putText(image, str(mm_to_pix.get(val)), tuple(np.array(start) + (counter % per) * np.array(space) + np.array([offset, 0])), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                 (0,0,0), 2, cv2.LINE_AA, False)
        image = cv2.putText(image, ', ' + val, tuple(np.array(start) + (counter % per) * np.array(space) + np.array([4.9*offset, 0]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (0,0,0), 2, cv2.LINE_AA, False)


        counter = counter + 1
        letter_counter = letter_counter + 1
        if letter_counter >= 26:
            letter_counter = 0
            times = times + 1

    start = (961,577)
    image = cv2.putText(image, 'Letter:', start, cv2.FONT_HERSHEY_SIMPLEX, 1, 
        (0,255,0), 2, cv2.LINE_AA, False)
    
    image = cv2.putText(image, '[xPix, yPix], xMM,yMM,zMM', tuple(np.array(start) + np.array([120, 0])), cv2.FONT_HERSHEY_SIMPLEX, 1, 
        (0,0,0), 2, cv2.LINE_AA, False)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(img).show()

def get_P_matrix():
    A = get_A_matrix()
    U, S, V = np.linalg.svd(A)
    P = (V[-1]).reshape((3,4))
    return P

def get_A_matrix(print_mode = False):
    x_s = []
    y_s = []
    z_s = []
    xprime = []
    yprime = []
    # mm_to_pix = corners
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


def decomposeP(P):
    '''
        The input P is assumed to be a 3-by-4 homogeneous camera matrix.
        The function returns a homogeneous 3-by-3 calibration matrix K,
        a 3-by-3 rotation matrix R and a 3-by-1 vector c such that
        K*R*[eye(3), -c] = P.

    '''
    W = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])
    # calculate K and R up to sign
    Qt, Rt = np.linalg.qr((W.dot(P[:,0:3])).T)
    K = W.dot(Rt.T.dot(W))
    R = W.dot(Qt.T)
    # correct for negative focal length(s) if necessary
    D = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    if K[0,0] < 0:
        D[0,0] = -1
    if K[1,1] < 0:
        D[1,1] = -1
    if K[2,2] < 0:
        D[2,2] = -1
    K = K.dot(D)
    R = D.dot(R)
    # calculate c
    c = -R.T.dot(np.linalg.inv(K).dot(P[:,3]))
    return K, R, c

def vector_dist(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


    
K, R, C = decomposeP(get_P_matrix())
K = K / K[2][2]
print(K)
print()
print(R)
print()
print(C)
# print(vector_dist(np.array([-923.99874656, -514.65675181, 345.93903943]), np.array([-265.49751349, -577.78797156, 188.97851488])))
