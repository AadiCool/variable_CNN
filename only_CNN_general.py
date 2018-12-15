import numpy as np

cnn_layer_types = ["CONV", "MAXPOOL"]
# ( layer type , x_length , y_length , zero_padding, no of mask ) zero_padding and no of mask not applicable for MAXPOOL
cnn_layer_info = []
ndelst = inpt_dim = []  # contains the node numbers in FC layer
mask_depth = []  # contains the mask depths of each layer
epoch_itr = optLyr = hydLyr = 0
lrn_rate = nrm_fac = 0.0
read_wt = 0

instructions_file = "instructions.txt"
data_input_file = "data_input_train.txt"
data_output_file = "data_output_train.txt"
weight_file = ""

f_ins = open(instructions_file, "r")
lns = f_ins.readlines()
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


def check_input():  # checking whether convolution operations are correct or not
    row, col = inpt_dim[0], inpt_dim[1]
    for i in range(len(cnn_layer_info)):
        pad = 0  # the pad applied
        if (cnn_layer_info[i][0] == 0):
            pad = cnn_layer_info[i][3]
        row = row - cnn_layer_info[i][1] + 2 * pad + 1
        col = col - cnn_layer_info[i][2] + 2 * pad + 1
    return row, col


row, col = check_input()
if (row <= 0 or col <= 0):  # row and column should be positive to be valid
    print("Invalid Convolution and pooling layers ..  Exitting code")
    exit(1)

inpLyr = row * col * mask_depth[-1]  # no of input nodes for the fully connected layer
ndelst.insert(0, inpLyr)
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
no_of_input_data = 0

# accepting input in the specified format and also the output
f_in = open(data_input_file, "r")
f_out = open(data_output_file, "r")
for lns in f_in:
    intgs = [(float(x)) for x in lns.strip(' \n').split()]
    if (len(intgs) == 0):
        train_input.append(np.array(train_input_data))
        train_input_data = []
        no_of_input_data += 1
        continue
    train_input_data.append(np.multiply(1.0 / nrm_fac, intgs))
f_in.close()
for lns in f_out:
    intgs = [float(x) for x in lns.split()]
    train_output.append(intgs)
f_out.close()


def make_conv_mask(dep, row, col):  # creating the mask for the convolution
    return np.random.rand(dep, row, col) - .5 * np.ones(shape=(dep, row, col), dtype=float)


def make_max_pool(dep, row, col):  # creating a dummy mask of same shape --  no use
    return np.zeros(shape=(dep, row, col), dtype=float)


# for max pool, the positions of the maximum wrt to the weight mask is stored
def create_masks():  # returning the masks for the convolution
    cnn_masks = []  # contains all the corelation masks for each layer
    func_dict = {0: make_conv_mask, 1: make_max_pool}  # the functions acc to masks
    for i in range(len(cnn_layer_info)):
        lyr_cnn_msk = []  # contains the mask for each layers
        if (cnn_layer_info[i][0] != 1):  # create masks for CONV Pool
            for k in range(mask_depth[i + 1]):  # creating specified no of masks in each layer
                lyr_cnn_msk.append(
                    func_dict[cnn_layer_info[i][0]](mask_depth[i], cnn_layer_info[i][1], cnn_layer_info[i][2]))
        else:
            lyr_cnn_msk.append(
                func_dict[cnn_layer_info[i][0]](mask_depth[i], cnn_layer_info[i][1], cnn_layer_info[i][2]))
        cnn_masks.append(lyr_cnn_msk)
    return cnn_masks


def add_padding(inpt, p):  # used for adding zero pad as necessary
    opt_arr = np.zeros((len(inpt), len(inpt[0]) + 2 * p, len(inpt[0][0]) + 2 * p), dtype=float)
    opt_arr[:, p:len(inpt[0]) + p, p:len(inpt[0][0]) + p] = inpt
    return opt_arr


def remove_pad(inpt, p): # used for removing the pad
    return inpt[:, p:len(inpt[0]) - p, p:len(inpt[0][0]) - p]


def convolute(mask, inpt, opt_dep):  # used for applying convolution for CONV layers
    row = len(inpt[0]) - len(mask[0][0]) + 1
    col = len(inpt[0][0]) - len(mask[0][0][0]) + 1
    result = np.zeros(shape=(opt_dep, row, col), dtype=float)
    for k in range(opt_dep):
        for i in range(row):
            for j in range(col):
                result[k][i][j] = np.sum(
                    np.multiply(mask[k], inpt[:, i:(i + len(mask[0][0])), j:j + len(mask[0][0][0])]))
    return result


def convolute_max_pool(mask, inpt, dep):  # used for applying MAX Pool layers
    row = len(inpt[0]) - len(mask[0]) + 1
    col = len(inpt[0][0]) - len(mask[0][0]) + 1
    max_pos = np.zeros(shape=(dep, row, col), dtype=float)
    result = np.zeros(shape=(dep, row, col), dtype=float)
    for k in range(dep):
        for i in range(row):
            for j in range(col):
                a = inpt[k, i:i + len(mask[0]), j:j + len(mask[0][0])]
                pos = np.unravel_index(np.argmax(a, axis=None), a.shape)
                max_pos[k][i][j] = 2 * pos[0] + pos[1]  # stores the 2D position where maximum occurs
                result[k][i][j] = np.amax(a)
    return max_pos, result


def forword_cnn(inpt, cnn_masks):
    inpt_list = []
    for i in range(len(cnn_layer_info)):
        if (cnn_layer_info[i][0] == 1):  # special treatment for MAXPOOL layers
            inpt_list.append(inpt)
            cnn_masks[i][0], inpt = convolute_max_pool(cnn_masks[i][0], inpt, mask_depth[i])
        else:
            if (cnn_layer_info[i][0] == 0):  # adding padding for CONV layers
                inpt = add_padding(inpt, cnn_layer_info[i][-2])
            inpt_list.append(inpt)
            inpt = convolute(cnn_masks[i], inpt, mask_depth[i + 1])
    inpt_list.append(inpt)
    return inpt_list, cnn_masks


def calc_mask_grad(mask, opt_lyr_grad, inpt_lyr):  # calculating mask gradient for CONV
    mask_grad = np.zeros(shape=(len(mask), len(mask[0]), len(mask[0][0])), dtype=float)
    for k in range(len(inpt_lyr)):  # calculating mask gradient layer-wise
        grad_2d = np.zeros(shape=(len(mask[0]), len(mask[0][0])), dtype=float)
        for i in range(len(mask[0])):
            for j in range(len(mask[0][0])):
                grad_2d[i][j] = np.sum(
                    np.multiply(opt_lyr_grad, inpt_lyr[k, i:i + len(opt_lyr_grad), j:j + len(opt_lyr_grad[0])]))
        mask_grad[k, :, :] = grad_2d
    return mask_grad


def jugar_grad(mask, opt_grad, i1, j1):  # calculating layer gradients at each position for CONV
    res = 0.0
    for i in range(i1, i1 - len(mask), -1):
        for j in range(j1, j1 - len(mask[0]), -1):
            try:  # for exitting index greater than highest length
                if (i < 0 or j < 0):  # for exitting negative indices
                    continue
                res += opt_grad[i][j] * mask[i1 - i][j1 - j]
            except:
                pass
    return res


def cnn_lyr_grad(mask_list, opt_lyr_grad, inpt_lyr):  # calculating layer gradients for CONV
    inpt_lyr_grad = np.zeros(shape=(len(inpt_lyr), len(inpt_lyr[0]), len(inpt_lyr[0][0])), dtype=float)
    for k in range(len(mask_list)):
        mask = mask_list[k]
        opt_grad = opt_lyr_grad[k]
        for k1 in range(len(inpt_lyr)):
            for i1 in range(len(inpt_lyr[0])):
                for j1 in range(len(inpt_lyr[0][0])):
                    inpt_lyr_grad[k1][i1][j1] += jugar_grad(mask[k1], opt_grad, i1, j1)
    return inpt_lyr_grad


def jugar_grad_max_pool(pos_mask, opt_grad, i1, j1, row_mask, col_mask):  # calculating layer gradients for MAX_POOL
    res = 0.0
    for i in range(i1, i1 - row_mask, -1):
        for j in range(j1, j1 - col_mask, -1):
            try:  # for exitting index greater than highest length
                if (i < 0 or j < 0):  # for exitting negative indices
                    continue
                mask = np.zeros(shape=(row_mask, col_mask), dtype=float)
                rw = int(pos_mask[i1 - i][j1 - j] / col_mask)
                cl = int(pos_mask[i1 - i][j1 - j]) - int(pos_mask[i1 - i][j1 - j] / col_mask)
                mask[rw][cl] = 1.0
                res += opt_grad[i][j] * mask[i1 - i][j1 - j]
            except:
                pass
    return res


def cnn_lyr_grad_max_pool(pos_mask_list, opt_lyr_grad, inpt_lyr):  # calculating layer gradients for MAX_POOL
    inpt_lyr_grad = np.zeros(shape=(len(inpt_lyr), len(inpt_lyr[0]), len(inpt_lyr[0][0])), dtype=float)
    row_mask = len(inpt_lyr[0]) - len(opt_lyr_grad[0]) + 1
    col_mask = len(inpt_lyr[0][0]) - len(opt_lyr_grad[0][0]) + 1
    for k1 in range(len(inpt_lyr)):
        pos_mask = pos_mask_list[k1]
        opt_grad = opt_lyr_grad[k1]
        for i1 in range(len(inpt_lyr[0])):
            for j1 in range(len(inpt_lyr[0][0])):
                inpt_lyr_grad[k1][i1][j1] = jugar_grad_max_pool(pos_mask, opt_grad, i1, j1, row_mask, col_mask)
    return inpt_lyr_grad


def backward_cnn(inpt_list, cnn_masks, last_lyr_grad):
    mask_grad_list = []
    layer_grad_list = []
    layer_grad_list.append(last_lyr_grad)
    for i in range(1,len(cnn_masks)+1):
        if (cnn_layer_info[-1 * i][0] == 0):
            mask_grad_lyr = []
            for j in range(len(cnn_masks[-1*i])):
                mask_grad_lyr.append(calc_mask_grad(cnn_masks[-1*i][j], layer_grad_list[-1][j], inpt_list[-1*i-1]) )
            mask_grad_list.append(mask_grad_lyr)
            lyr_grad = cnn_lyr_grad(cnn_masks[-1*i], layer_grad_list[-1], inpt_list[-1*i-1])
            layer_grad_list.append(remove_pad(lyr_grad,cnn_layer_info[-1*i][-2]))
            inpt_list[-1*i-1] = remove_pad(inpt_list[-1*i-1],cnn_layer_info[-1*i][-2])
        elif(cnn_layer_info[-1 * i][0] == 1):
            layer_grad_list.append(cnn_lyr_grad_max_pool(cnn_masks[-1*i][0], layer_grad_list[-1], inpt_list[-1*i-1]))
    mask_grad_list = mask_grad_list[::-1]
    layer_grad_list = layer_grad_list[::-1]
    return mask_grad_list , layer_grad_list


def cnn_update_masks(masks , masks_grad):
    new_masks = []
    for i in range(len(masks)):
        new_masks_lyr = []
        for j in range(len(masks[i])):
            new_masks_lyr.append( masks[i][j] + np.multiply(lrn_rate*(-1) , masks_grad[i][j]) )
        new_masks.append(new_masks_lyr)
    return new_masks

#b=np.array([ [ [1,2,3,4,5,6,7], [4,5,6,5,6,7,8], [4,5,6,2,3,1,2]] ])
inpt = np.array([[1, 2, 1, 2], [1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 2, 1]])
cnn_masks = create_masks()
inpt = np.reshape(inpt, (mask_depth[0], len(inpt), len(inpt[0])))
#arr1, arr2 = forword_cnn(inpt, cnn_masks)
#print(arr1[-1])
#print(arr1[1][0])
#print(arr2[1][0][0])
# print(arr2[0])
# print(np.sum(arr1[1][0]))
# print(arr1[1][1])
# print(jugar_grad(arr2[0][0][0],arr1[1][0],0,0))
#print(jugar_grad_max_pool(arr2[1][0][0], arr1[1][0], 0, 0, 2, 2))
#print("output")
#print(cnn_lyr_grad_max_pool(arr2[1][0], arr1[2], arr1[1]))
#print(calc_mask_grad(arr2[1][1], arr1[2][1] ,arr1[1] ))
# print(add_padding(np.array([[[1,2,3],[4,5,6]],[[1,3,2],[6,5,6]]]),2))
#ar1,ar2 = backward_cnn(arr1,arr2,arr1[-1])
#print(arr2[1][0])
#print("a")
#print(ar1[1][0])
#print("opt")
#a1 = cnn_update_masks(arr2,ar1)
#print(a1[1][0])
#print(add_padding(b,1))
#print(remove_pad(add_padding(b,1),1))


for i in range(20):
    inp, msk =forword_cnn(inpt, cnn_masks)
    err = np.sum(np.absolute(inp[-1]))
    inplst_grad = np.multiply(2,inp[-1])
    msk_grad, inp_grad = backward_cnn(inp,msk,inplst_grad)
    cnn_masks = cnn_update_masks(msk,msk_grad)
    print("error = "+str(err))
