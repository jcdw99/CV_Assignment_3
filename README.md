# CV Assignment 3


## Question 1
The points obtained and used to estimate the projection matrix are shown at the top of the file in the form of a key value paired dictionary. These points are used by the functions get_A_matrix and get_P_matrix respectively. If you run the code, the projection matrix is estimated and subsequently decomposed using the provided decomposition function. The output is printed to the terminal.

## Question 2
This question is split into two files, a file for part A, and a file for part B.
### Part A
This section is essentially a repeat of question 1, the form of the file is identical, and the output is the same as well. Refer to question 1 if confused.
### Part B
If you run this file, the 3d plot is rendered and displayed on screen. You can zoom in on parts of the plot by holding the right-click button, and scrolling. The plot appears better if viewed in full screen mode.

## Question 3
There are two functions of interest here, draw_on_1() and draw_on_2(). By default the first function is called. If you run the code, it will determine the axes (from the projection matrix) of image 1. The other function does the same, but for the other image using the other projection matrix.

## Question 4
Question 4 is admittedly not coded in the most elegant way. It mostly contains a file that just executes the desired operations in a serial format, without the use of much functions. That being said, it is simple to run. Just press the run button, and it will display both images on screen via the Python Pillow Library.