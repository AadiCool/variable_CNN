import cv2
import glob

img_list = glob.glob("test/*")
img_list = sorted(img_list)

no_of_img_train = 600
no_of_label = 10
no_of_img_test = 100

f_in = open("data_input_train.txt","w")
f_out = open("data_output_train.txt","w")

for i in range( no_of_img_train ):
    img = cv2.imread(img_list[i],0)
    dat = ""
    for j in range(len(img)):
        for k in range(len(img[0])):
            dat += str(img[j][k])+" "
        dat += "\n"
    f_in.write( dat + "\n" )
    dat = ""
    label = int( img_list[i][-5] )
    for j in range( no_of_label ):
        if ( label == j ):
            dat = dat + str(1) + " "
        else:
            dat = dat + str(0) + " "
    f_out.write( dat+"\n" )

f_in.close()
f_out.close()

f_in = open("data_input_test.txt","w")
f_out = open("data_output_test.txt","w")

for i in range( no_of_img_test ):
    img = cv2.imread(img_list[i+no_of_img_train],0)
    dat = ""
    for j in range(len(img)):
        for k in range(len(img[0])):
            dat += str(img[j][k])+" "
        dat += "\n"
    f_in.write( dat + "\n" )
    dat = ""
    label = int( img_list[i+no_of_img_train][-5] )
    for j in range( no_of_label ):
        if ( label == j ):
            dat = dat + str(1) + " "
        else:
            dat = dat + str(0) + " "
    f_out.write( dat+"\n" )

f_in.close()
f_out.close()