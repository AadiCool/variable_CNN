import numpy as np
from datetime import datetime

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


#read weights and masks from a file
def read_masks_wts():
    f_wt = open(weight_file, "r")
    lns = f_wt.readlines()
    c = 0
    wtmtx = []  # the array of the corresponding weight matrices
    masks_list = [] # the convolution masks
    for i in range(len(cnn_layer_info)):
        if( cnn_layer_info[i][0] == 0 ):
            masks_list_lyr = []
            for j in range(cnn_layer_info[i][-1]):
                masks = np.zeros(shape=(mask_depth[i],cnn_layer_info[i][1],cnn_layer_info[i][2]), dtype=float)
                for row in range(len(masks[0])):
                    row_ln = [x for x in lns[c].strip(' \n').split('\t')]
                    c+=1
                    for dep in range(len(masks)):
                        mtx_row = [(float(x)) for x in row_ln[dep].strip(' \n').split(' ')]
                        for col in range(len(masks[0][0])):
                            masks[dep][row][col] = mtx_row[col]
                masks_list_lyr.append(masks)
                c+=1
            c+=1
        else:
            masks_list_lyr = []
            masks = np.zeros(shape=(mask_depth[i], cnn_layer_info[i][1], cnn_layer_info[i][2]), dtype=float)
            c = c + 3 + len(masks)
            masks_list_lyr.append(masks)
        masks_list.append(masks_list_lyr)
    c+=1
    for i in range(hydLyr + 1):
        wt = [] # the weights
        for j in range(0, ndelst[i + 1]):
            intgs = [(float(x)) for x in lns[c].split()]
            wt.append(np.array(intgs))
            c += 1
        wtmtx.append(np.array(wt))
        c += 2
    f_wt.close()
    return wtmtx, masks_list


# creates the initial weights for the  FC layer
def create_initial_wts():
    wtmtx = []  # initial weight matrix list
    for i in range(1, len(ndelst), 1):
        # creating zero-centered weights
        wtmtx.append(
            np.random.rand(ndelst[i], ndelst[i - 1]) - .5 * np.ones(shape=(ndelst[i], ndelst[i - 1]), dtype=float))
    return wtmtx


# used for adding zero pad as necessary
def add_padding(inpt, p):
    opt_arr = np.zeros((len(inpt), len(inpt[0]) + 2 * p, len(inpt[0][0]) + 2 * p), dtype=float)
    opt_arr[:, p:len(inpt[0]) + p, p:len(inpt[0][0]) + p] = inpt
    return opt_arr


# used for removing the pad
def remove_pad(inpt, p):
    return inpt[:, p:len(inpt[0]) - p, p:len(inpt[0][0]) - p]


def sigmoid(z):
    # sigmoid function
    return 1 / (1 + np.exp(-z))


def sigmoidPrime(z):
    # gradient of sigmoid function
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


# used for applying convolution for CONV layers
def convolute(mask, inpt, opt_dep):
    row = len(inpt[0]) - len(mask[0][0]) + 1
    col = len(inpt[0][0]) - len(mask[0][0][0]) + 1
    result = np.zeros(shape=(opt_dep, row, col), dtype=float)
    for k in range(opt_dep):
        for i in range(row):
            for j in range(col):
                result[k][i][j] = np.sum(
                    np.multiply(mask[k], inpt[:, i:(i + len(mask[0][0])), j:j + len(mask[0][0][0])]))
    return result


# used for applying MAX Pool layers
def convolute_max_pool(mask, inpt, dep):
    row = len(inpt[0]) - len(mask[0]) + 1
    col = len(inpt[0][0]) - len(mask[0][0]) + 1
    # print("row "+str(row))
    # print("col " + str(col))
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


# performs the forward pass of the CONV and MAXPOOL layers
def forword_cnn(inpt, cnn_masks):
    inpt_list = []
    for i in range(len(cnn_layer_info)):
        if (cnn_layer_info[i][0] == 1):  # special treatment for MAXPOOL layers
            # print(str(len(inpt[0])) + " in forward_cnn1")
            inpt_list.append(inpt)
            cnn_masks[i][0] = make_max_pool(mask_depth[i], cnn_layer_info[i][1], cnn_layer_info[i][2])
            cnn_masks[i][0], inpt = convolute_max_pool(cnn_masks[i][0], inpt, mask_depth[i])
            # print(str(len(inpt[0])) + " in forward_cnn2")
        else:
            if (cnn_layer_info[i][0] == 0):  # adding padding for CONV layers
                inpt = add_padding(inpt, cnn_layer_info[i][-2])
            inpt_list.append(inpt)
            inpt = convolute(cnn_masks[i], inpt, mask_depth[i + 1])
    inpt_list.append(inpt)
    return inpt_list, cnn_masks


# performs the forward pass of the FC layer
def forward_pass(wtmtx, lyrs):
    lyrs_list = []  # the layers contained in a list
    lyrs_list_no_sgm = []  # the layers before the sigmoid is applied
    lyrs_list.append(lyrs)
    lyrs_list_no_sgm.append(lyrs)
    for i in range(0, len(ndelst) - 1):
        lyrs_list_no_sgm.append(np.matmul(wtmtx[i], lyrs))
        lyrs = sigmoid(lyrs_list_no_sgm[-1])
        lyrs_list.append(lyrs)
    return lyrs_list, lyrs_list_no_sgm


# calculating mask gradient for CONV
def calc_mask_grad(mask, opt_lyr_grad, inpt_lyr):
    mask_grad = np.zeros(shape=(len(mask), len(mask[0]), len(mask[0][0])), dtype=float)
    for k in range(len(inpt_lyr)):  # calculating mask gradient layer-wise
        grad_2d = np.zeros(shape=(len(mask[0]), len(mask[0][0])), dtype=float)
        for i in range(len(mask[0])):
            for j in range(len(mask[0][0])):
                grad_2d[i][j] = np.sum(
                    np.multiply(opt_lyr_grad, inpt_lyr[k, i:i + len(opt_lyr_grad), j:j + len(opt_lyr_grad[0])]))
        mask_grad[k, :, :] = grad_2d
    return mask_grad


# calculating layer gradients at each position for CONV
def jugar_grad(mask, opt_grad, i1, j1):
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


# calculating layer gradients for CONV
def cnn_lyr_grad(mask_list, opt_lyr_grad, inpt_lyr):
    inpt_lyr_grad = np.zeros(shape=(len(inpt_lyr), len(inpt_lyr[0]), len(inpt_lyr[0][0])), dtype=float)
    for k in range(len(mask_list)):
        mask = mask_list[k]
        opt_grad = opt_lyr_grad[k]
        for k1 in range(len(inpt_lyr)):
            for i1 in range(len(inpt_lyr[0])):
                for j1 in range(len(inpt_lyr[0][0])):
                    inpt_lyr_grad[k1][i1][j1] += jugar_grad(mask[k1], opt_grad, i1, j1)
    return inpt_lyr_grad


# calculating layer gradients for MAX_POOL
def jugar_grad_max_pool(pos_mask, opt_grad, i1, j1, row_mask, col_mask):
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


# calculating layer gradients for MAX_POOL
def cnn_lyr_grad_max_pool(pos_mask_list, opt_lyr_grad, inpt_lyr):
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


# calculates the backward pass of the CONV and MAXPOOL layers
def backward_cnn(inpt_list, cnn_masks, last_lyr_grad):
    mask_grad_list = []
    layer_grad_list = []
    layer_grad_list.append(last_lyr_grad)
    for i in range(1, len(cnn_masks) + 1):
        if (cnn_layer_info[-1 * i][0] == 0):
            mask_grad_lyr = []
            for j in range(len(cnn_masks[-1 * i])):
                mask_grad_lyr.append(
                    calc_mask_grad(cnn_masks[-1 * i][j], layer_grad_list[-1][j], inpt_list[-1 * i - 1]))
            mask_grad_list.append(mask_grad_lyr)
            lyr_grad = cnn_lyr_grad(cnn_masks[-1 * i], layer_grad_list[-1], inpt_list[-1 * i - 1])
            layer_grad_list.append(remove_pad(lyr_grad, cnn_layer_info[-1 * i][-2]))
            inpt_list[-1 * i - 1] = remove_pad(inpt_list[-1 * i - 1], cnn_layer_info[-1 * i][-2])
        elif (cnn_layer_info[-1 * i][0] == 1):
            layer_grad_list.append(
                cnn_lyr_grad_max_pool(cnn_masks[-1 * i][0], layer_grad_list[-1], inpt_list[-1 * i - 1]))
            mask_grad_list.append(cnn_masks[-1 * i])  # adding dummy gradients to maintain indices
    mask_grad_list = mask_grad_list[::-1]
    layer_grad_list = layer_grad_list[::-1]
    return mask_grad_list, layer_grad_list


# performs the cost function of the entire network
def cost_func(final_lyr, label):
    for i in range(len(final_lyr)):
        final_lyr[i] = final_lyr[i] - label[i]  # difference between the required labels
    err = np.linalg.norm(final_lyr) ** 2  # taking the squares
    return final_lyr, err


# performs the backpropagation of the FC layer
def backprop(wtmtx, lyrs, lyrs_list_no_sgm):
    lyr_grad = []  # gradient for the corresponding layers
    wt_grad = []  # gradient for the weight matrices
    opt_lyr = np.multiply(2, lyrs[-1])  # gradient from the error function
    x = sigmoidPrime(np.array(lyrs_list_no_sgm[-1]))  # gradient while passing the sigmoid layer
    opt_lyr = np.multiply(opt_lyr, x)  # final output layer gradient with weights multiplied
    lyr_grad.append(opt_lyr)
    for i in range(2, len(lyrs) + 1):
        x = np.matmul(lyr_grad[-1], np.transpose(lyrs[-1 * i]))
        wt_grad.append(x)
        opt_lyr = np.matmul(np.transpose(wtmtx[1 - i]), lyr_grad[-1])
        opt_lyr = np.multiply(opt_lyr, sigmoidPrime(np.array(lyrs_list_no_sgm[-1 * i])))
        lyr_grad.append(opt_lyr)
    wt_grad = wt_grad[::-1]  # reversing the array
    lyr_grad = lyr_grad[::-1]  # reversing the array
    return wt_grad, lyr_grad


# update the CONV and the MAXPOOL layers masks
def cnn_update_masks(masks, masks_grad):
    global lrn_rate
    new_masks = []
    for i in range(len(masks)):
        if (cnn_layer_info[i][0] == 1):
            new_masks.append(masks[i])
        else:
            new_masks_lyr = []
            for j in range(len(masks[i])):
                new_masks_lyr.append(masks[i][j] + np.multiply(lrn_rate * (-1), masks_grad[i][j]))
            new_masks.append(new_masks_lyr)
    return new_masks


# updating the new weight matrix as per gradient of the FC layer
def wt_update(wtx_grad_dt_pts, wtx):
    global lrn_rate
    return np.add(wtx, np.multiply(lrn_rate * (-1), wtx_grad_dt_pts[0]))

#used for calculating gradients over all the data points
def run(cnn_masks, wtmx, k):
    mask_grad_dt_pts = []
    wt_grad_dt_pts = []
    err_total = 0.0
    for i in range(no_of_input_data):
        inptt = np.array(train_input[i]).reshape(mask_depth[0], len(train_input[i]), len(train_input[i][0]))
        inp, msk = forword_cnn(inptt, cnn_masks)
        inp_last = np.array(inp[-1])
        sgm, no_sgm = forward_pass(wtmx, inp_last.reshape(inpLyr, 1))
        sgm[-1], err = cost_func(sgm[-1], train_output[i])
        err_total += err  # taking up for the total error
        wt_grad, lyrs_grad = backprop(wtmx, sgm, no_sgm)
        fst_lyr_grad = np.array(lyrs_grad[0]).reshape(inp_last.shape)
        msk_grad, inp_grad = backward_cnn(inp, msk, fst_lyr_grad)
        wt_grad_dt_pts.append(wt_grad)
        mask_grad_dt_pts.append(msk_grad)
        if (i != 0):
            wt_grad_dt_pts[0] = np.add(wt_grad_dt_pts[0], wt_grad_dt_pts[1])  # the zeroth element is the sum
            wt_grad_dt_pts = wt_grad_dt_pts[:1]  # discarding the next element, the grad weight for that data point
            for i in range(len(mask_grad_dt_pts[0])):
                for j in range(len(mask_grad_dt_pts[0][i])):
                    mask_grad_dt_pts[0][i][j] = np.add(mask_grad_dt_pts[0][i][j], mask_grad_dt_pts[1][i][j])
            mask_grad_dt_pts = mask_grad_dt_pts[:1]  # discarding the next element, the grad mask for that data point

    wtmx = wt_update(wt_grad_dt_pts, wtmx)
    cnn_masks = cnn_update_masks(cnn_masks, mask_grad_dt_pts[0])
    print("The error for the epoch " + str(k) + " " + str(err_total), end="")

    return wtmx, cnn_masks, err_total


# used for copying CNN masks
def copy_cnn_mask(cnn_masks):
    mask_new = []
    for i in range(len(cnn_masks)):
        mask_lyr_new = []
        for j in range(len(cnn_masks[i])):
            mask_lyr_new.append(np.copy(cnn_masks[i][j]))
        mask_new.append(mask_lyr_new)
    return mask_new


# used for executing the code and calculating the final masks and weights over all epochs
def execute():
    print(" ")
    global read_wt
    if( read_wt == 0):
        wtmx = create_initial_wts()
        cnn_masks = create_masks()
    else:
        wtmx, cnn_masks = read_masks_wts()
    tmstart = datetime.now()
    wtmx, cnn_masks, err_prev = run(cnn_masks, wtmx, 1)  # performing first iteration
    tmend = datetime.now()
    print(" Tiem required = " + str((tmend - tmstart).total_seconds()))
    wtmx_min_err = np.copy(wtmx)
    cnn_masks_min_err = copy_cnn_mask(cnn_masks)
    for i in range(1, epoch_itr):
        tmstart = datetime.now()
        wtmx, cnn_masks, err_total = run(cnn_masks, wtmx, i + 1)
        tmend = datetime.now()
        print(" Tiem required = "+str((tmend-tmstart).total_seconds()))
        if (err_total < err_prev):  # taking the weight matrix for minimum error
            wtmx_min_err = np.copy(wtmx)
            cnn_masks_min_err = copy_cnn_mask(cnn_masks)
            err_prev = err_total

    print("\n The minimum error is " + str(err_prev))
    return wtmx_min_err, cnn_masks_min_err


def write_Matrix(wtmtx,cnn_masks):  #writing the weight matrices to a file
    f=open("weightMatrix.txt","w")
    #write the CONV and MAXPOOL masks
    for i1 in range(len(cnn_masks)):
        for j1 in range(len(cnn_masks[i1])):
            if(cnn_layer_info[i1][0] == 0):
                mask = cnn_masks[i1][j1]
            else:
                mask = np.zeros(shape=(len(cnn_masks[i1][j1]), cnn_layer_info[i1][1], cnn_layer_info[i1][2]), dtype=float)
            for row in range(len(mask[0])):
                for dep in range(len(mask)):
                    for col in range(len(mask[0][0])):
                        f.write(str(mask[dep][row][col])+" ")
                    f.write("\t")
                f.write("\n")
            f.write("\n")
        f.write("\n")
    f.write("\n")
    # write the FC weights
    for i in range(len(wtmtx)):
        for j in range(len(wtmtx[i])):
            for k in range(len(wtmtx[i][j])):
                f.write( str(wtmtx[i][j][k]) +" " )
            f.write("\n")
        f.write("\n\n")
    f.close()

wtmtx, cnn_msks = execute()
write_Matrix(wtmtx, cnn_msks)
