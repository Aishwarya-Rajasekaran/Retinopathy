import cv2
import time
import h5py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda')
#print(torch.cuda.current_device())

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
        self.fc_1 = nn.Linear(2048, 512, bias=True)
        self.fc_2 = nn.Linear(512, 2, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 2048)
        x = F.relu(self.fc_1(x))
        x = F.softmax(self.fc_2(x))
        return x

def occlusion(window,prob,path):
    size=256
    index =0
    img=cv2.resize(cv2.imread(path),(256,256))
    for i in range(0,size,window):
        for j in range(0,size,window):
            cv2.putText(img,str(prob[index]),(i,j),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,0,0),1,cv2.LINE_AA)
            index+=1   
   #cv2.imshow("Probablity matrix",img)
    cv2.imwrite("/storage/home/aishwaryar/kube-test/job/occlusion.jpg",img)

def loadData(path):
    dataset = h5py.File(path, 'r')
    X, Y = dataset['X'][:].reshape((-1, 1, 256, 256)), np.int32(dataset['Y'][:])
    print("size of the data:", X.shape, Y.shape)
    return torch.from_numpy(X), torch.from_numpy(Y).long()

def SlidingWindow(window,path):
    size=256
    images =[]
    img = cv2.resize(cv2.imread(path,0),(256,256))

    #Sliding window
    for i in range(0,size,window):
        for j in range(0,size,window):
            x= np.copy(img)
            x[i:i+window,j:j+window] = 0  # moving horizontally
            images.append(x/255)

    #Save the window
    images= np.float32(images)
    np.save('/scratch/scratch1/retinopathy/data/gaussian_filter/model/window.npy',images)
    print("saved the output of sliding window")

def loadCheck(path):
    dataset = np.load(path)
    X = dataset.reshape((-1,1, 256, 256))
    Y = torch.ones(X.shape[0],dtype=torch.long) 
    print("size of the data:", X.shape)
    return torch.from_numpy(X),Y

def trainTest(data, batch_size, net, criterion, optimizer, is_train = True):
    data_X, data_Y = data
    pred_Y = np.zeros((data_Y.shape[0], 2))
    running_loss = 0.0
    num_batches = int(data_X.shape[0] / batch_size)
    for batch in range(num_batches):
        # get the inputs
        start, end = batch * batch_size, (batch + 1) * batch_size
        inputs, labels = data_X[start : end].to(device), data_Y[start : end].to(device)
        # zero the parameter gradients
        if is_train == True:
            optimizer.zero_grad()

        outputs = net(inputs)
        #print('output size : ', outputs.size())
        pred_Y[start : end] = outputs.cpu().detach().numpy().copy()
        loss = criterion(outputs, labels)
        if is_train == True:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
    return running_loss / data_X.shape[0], pred_Y

print('Initializing network')
net = Net().to(device)
print(net)

criterion = nn.CrossEntropyLoss(size_average = False)
lr = 0.001
optimizer = optim.Adam(net.parameters(), lr = lr)

epochs = 200
batch_size = 50

MODE = 'TEST'
finetune = False


if 'TRAIN' in MODE:
    train_X, train_Y = loadData('/scratch/scratch1/retinopathy/data/gaussian_filter/model/train.h5')
    valid_X, valid_Y = loadData('/scratch/scratch1/retinopathy/data/gaussian_filter/model/valid.h5')
    print(valid_Y)
    num_batches = int(train_X.shape[0] / batch_size)
    if finetune == True:
        net.load_state_dict(torch.load('/scratch/scratch1/retinopathy/data/gaussian_filter/model/occ_classifier.h5'))
    # Training
    loss_history = [np.inf]
    patience, impatience, limit = 50, 0, 6
    best_epoch, best_valid_loss = 0, np.inf
    for epoch in range(epochs):

        train_loss, _ = trainTest((train_X, train_Y), batch_size, net, criterion, optimizer, is_train = False)
        valid_loss, _ = trainTest((valid_X, valid_Y), batch_size, net, criterion, optimizer, is_train = False)
        start = time.time()
        trainTest((train_X, train_Y), batch_size, net, criterion, optimizer, is_train = True)
        end = time.time()

        print('Epoch {}, Training-Loss = {}, Valid-Loss = {}, Time = {}'.format(epoch, train_loss, valid_loss, end - start))

        impatience += 1
        if valid_loss < min(loss_history):
            print('A better model has been obtained. Saving this model to /scratch/scratch1/retinopathy/data/gaussian_filter/model/occ_classifier.h5 ')
            torch.save(net.state_dict(), '/scratch/scratch1/retinopathy/data/gaussian_filter/model/occ_classifier.h5')
            best_loss, best_epoch = valid_loss, epoch + 1
            impatience = 0
        loss_history.append(valid_loss)
        if impatience == patience:
            if limit == 0:
                break
            else:
                limit -= 1
                impatience = 0
                lr /= 2
                optimizer = optim.Adam(net.parameters(), lr = lr)
                print('Current limit = ', limit)
    print('Finished Training: best model at {} epochs with valid loss = {}'.format(best_epoch, best_loss))

if 'TEST' in MODE:
    img_path ='/scratch/scratch1/retinopathy/data/gaussian_filter/test/abnormal/ab_1.jpg'
    SlidingWindow(30,img_path)
    test_X, test_Y = loadCheck('/scratch/scratch1/retinopathy/data/gaussian_filter/model/window.npy')
    print (test_Y)
    print ("Test_X",test_X)
    test_loss, _ = trainTest((test_X, test_Y), batch_size, net, criterion, optimizer, is_train = False)
    print('Test loss for random basline = {}'.format(test_loss))
    # Testing
    net.load_state_dict(torch.load('/scratch/scratch1/retinopathy/data/gaussian_filter/model/occ_classifier.h5'))
    start = time.time()
    test_loss, pred_Y = trainTest((test_X, test_Y), batch_size, net, criterion, optimizer, is_train = False)
    end = time.time()
    print(pred_Y)
    occlusion(30,pred_Y[:,1],img_path)
    print("should have displayed")

    total=test_Y.size(0)
    pq=torch.tensor(pred_Y)
    p=torch.max(pq,1)
    correct=(p[1]==test_Y).sum().item()
    test_accuracy=correct/total
    print('Test loss = {}'.format(test_loss))
    print('Test Accuracy ={}'.format(test_accuracy))
