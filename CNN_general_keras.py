import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

cnn_layer_types = ["CONV", "MAXPOOL"]
# ( layer type , x_length , y_length , zero_padding, no of mask ) zero_padding and no of mask not applicable for MAXPOOL
cnn_layer_info = []
ndelst = inpt_dim = []  # contains the node numbers in FC layer
cnn_layer_dim = []
mask_depth = []  # contains the mask depths of each layer
epoch_itr = optLyr = hydLyr = 0
lrn_rate = nrm_fac = 0.0
read_wt = 0

instructions_file = "instructions.txt"
train_data_input_file = "data_input_train.txt"
train_data_output_file = "data_output_train.txt"
test_data_input_file = "data_input_test.txt"
test_data_output_file = "data_output_test.txt"
weight_file = ""

f_ins = open(instructions_file, "r")
lns = f_ins.readlines()
# reading the instructions from the instruction files
try:
    lrn_rate = float(lns[0].strip(' \n'))  # first line should be learning rate
    epoch_itr = int(lns[1].strip(' \n'))  # second line should contain no of iterations
    inpt_dim = lns[2].strip(' \n').split(' ')  # third line should contain the input matrix dimensions
    inpt_dim = [int(inpt_dim[i]) for i in range(len(inpt_dim))]
    if (len(inpt_dim) == 3):
        mask_depth.append(inpt_dim[2])
    else:
        mask_depth.append(1)
    optLyr = int(lns[3].strip(' \n'))  # fourth line should contain no of nodes in output layer
    nrm_fac = float(lns[4].strip(' \n'))  # fifth line should contain normalization factor
    hydLyr = int(lns[5].strip(' \n'))  # sixth line should contain no of hidden layer
    ndelst.extend(
        [int(x) for x in lns[6].strip(' \n').split(' ')])  # seventh line should contain no of nodes in hidden layer
    ndelst.append(optLyr)
    read_wt_ln = lns[7].strip(' \n')
    if (int(read_wt_ln[0]) == 1):
        weight_file = (read_wt_ln.split(' '))[1]
        read_wt = 1
    for i in range(8, len(lns)):  # From eighth line the convolutions and pooling instructions are given
        intgs = lns[i].strip(' \n').split(' ')
        operate = cnn_layer_types.index(intgs[0])
        if (operate == 0):  # check for convolution or pooling
            cnn_layer_info.append((operate, int(intgs[1]), int(intgs[2]), int(intgs[3]), int(intgs[4])))
            mask_depth.append(int(intgs[4]))
        else:
            cnn_layer_info.append((operate, int(intgs[1]), int(intgs[2])))
            mask_depth.append(mask_depth[-1])
except:
    print("Wrong Instruction list ..   Exitting code")
    exit(1)
f_ins.close()


# checking whether convolution operations are correct or not
def check_input():
    row, col = inpt_dim[0], inpt_dim[1]
    cnn_layer_dim.append((mask_depth[0], row, col))
    for i in range(len(cnn_layer_info)):
        pad = 0  # the pad applied
        if (cnn_layer_info[i][0] == 0):
            pad = cnn_layer_info[i][3]
        row = row - cnn_layer_info[i][1] + 2 * pad + 1
        col = col - cnn_layer_info[i][2] + 2 * pad + 1
        cnn_layer_dim.append((mask_depth[i + 1], row, col))
    return row, col


row, col = check_input()
if (row <= 0 or col <= 0):  # row and column should be positive to be valid
    print("Invalid Convolution and pooling layers ..  Exitting code")
    exit(1)
inpLyr = row * col * mask_depth[-1]  # no of input nodes for the fully connected layer
ndelst.insert(0, inpLyr)
# printing the layer informations
print(" Learn Rate = " + str(lrn_rate))
print(" No of epoch iterations = " + str(epoch_itr))
print(" No of input layer node = " + str(inpLyr))
print(" No of output layer node = " + str(optLyr))
print(" No of normalization  = " + str(nrm_fac))
for i in range(len(cnn_layer_info)):
    pad = 0
    no_mask = None
    if (cnn_layer_info[i][0] == 0):
        pad = cnn_layer_info[i][3]
        no_mask = cnn_layer_info[i][4]
    print(" " + cnn_layer_types[cnn_layer_info[i][0]] + " " + str(cnn_layer_info[i][1]) + "X" + str(
        cnn_layer_info[i][2]) + " pad " + str(pad) + " no of masks " + str(no_mask))
print(" No of Hidden layers = " + str(hydLyr))
print(" No of nodes in the hidden layers = ", end="")
for i in range(1, len(ndelst) - 1):
    print(str(ndelst[i]), end=" ")
print("")

train_input = []
train_input_data = []
train_output = []

test_input = []
test_input_data = []
test_output = []

no_of_input_data_train = 0
no_of_input_data_test = 0

# accepting train input in the specified format and also the output
f_in = open(train_data_input_file, "r")
f_out = open(train_data_output_file, "r")
for lns in f_in:
    intgs = [(float(x)) for x in lns.strip(' \n').split()]
    if (len(intgs) == 0):
        train_input.append(np.array(train_input_data))
        train_input_data = []
        no_of_input_data_train += 1
        continue
    train_input_data.append(np.multiply(1.0 / nrm_fac, intgs))
f_in.close()
for lns in f_out:
    intgs = [float(x) for x in lns.split()]
    train_output.append(intgs)
f_out.close()

# accepting test input in the specified format and also the output
f_in = open(test_data_input_file, "r")
f_out = open(test_data_output_file, "r")
for lns in f_in:
    intgs = [(float(x)) for x in lns.strip(' \n').split()]
    if (len(intgs) == 0):
        test_input.append(np.array(test_input_data))
        test_input_data = []
        no_of_input_data_test += 1
        continue
    test_input_data.append(np.multiply(1.0 / nrm_fac, intgs))
f_in.close()
for lns in f_out:
    intgs = [float(x) for x in lns.split()]
    test_output.append(intgs)
f_out.close()

train_input = np.array(train_input).reshape(no_of_input_data_train, mask_depth[0], len(train_input[0]),
                                            len(train_input[0][0]))
train_output = np.array(train_output)
test_input = np.array(test_input).reshape(no_of_input_data_test, mask_depth[0], len(test_input[0]),
                                            len(test_input[0][0]))
test_output = np.array(test_output)

model = Sequential()
for i in range(len(cnn_layer_info)):
    input_shape = cnn_layer_dim[i]
    if (cnn_layer_info[i][0] == 0):
        p = cnn_layer_info[i][3]
        input_shape_pad = (cnn_layer_dim[i][0], cnn_layer_dim[i][1] + 2 * p, cnn_layer_dim[i][2] + 2 * p)
        model.add(ZeroPadding2D(padding=cnn_layer_info[i][3], data_format='channels_first', input_shape=input_shape))
        model.add(Conv2D(cnn_layer_info[i][-1], kernel_size=(cnn_layer_info[i][1], cnn_layer_info[i][2]),
                         data_format='channels_first', use_bias=False, input_shape=input_shape_pad))
    else:
        model.add(MaxPooling2D(pool_size=(cnn_layer_info[i][1], cnn_layer_info[i][2]), strides=(1, 1),
                               data_format='channels_first'))
model.add(Flatten(input_shape=cnn_layer_dim[-1]))
for i in range(hydLyr + 1):
    model.add(Dense(ndelst[i + 1], activation='sigmoid', use_bias=False, input_dim=ndelst[i]))
adam = optimizers.Adam(lr=lrn_rate)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
if (read_wt == 1):
    model.load_weights(weight_file)
filepath = 'weights_keras.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=100)
callbacks_list = [checkpoint]
model.fit(train_input, train_output, epochs=epoch_itr, batch_size=100, callbacks=callbacks_list)

model.save_weights('weights_keras.h5')

f_res = open("results.txt","w")
print("\n\n       .............Predicting results................    ")
res = model.predict(test_input, verbose=1)
for i in range(len(res)):
    f_res.write(' '.join([str(x) for x in res[i]]))
    f_res.write('\n')

cnf_mtx = np.zeros(shape=(optLyr, optLyr), dtype=int)
acc = 0.0
# rows are actual and columns are predicted

for i in range(len(test_output)):
    act = np.argmax(test_output[i])
    prdc = np.argmax(res[i])
    cnf_mtx[act][prdc] += 1
    if( act == prdc ):
        acc += 1.0
print( " Accuracy = "+str(acc/len(test_output)))

df_cm = pd.DataFrame(cnf_mtx, index = [i for i in range(optLyr) ], columns = [i for i in range(optLyr) ] )
plt.figure(figsize = (optLyr,optLyr))
sn.heatmap(df_cm, annot=True)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix")
plt.show(sn)

f_cnf = open("confusion_matrix.txt", "w")
f_cnf.write('\n'.join('\t'.join('{:3}'.format(item) for item in row) for row in cnf_mtx))
f_cnf.close()



