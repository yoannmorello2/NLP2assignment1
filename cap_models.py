'''This module contains the main model class, the batcher class and the training function
'''
import numpy as np
import torch
import math
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import os
device = torch.device("cuda:2")

class Caption_checker(nn.Module):
'''class used to define the main model.
----------
params:
hidden_size. Set to 3000 by default
LSTM_out_size. 500 by default
height of the images to classify : 100 by default
width of the images to classify : 10 by default
device : torch.device("cuda:2") by default 

'''
    def __init__(self, hidden_size=3000, LSTM_out_size=500, height=100, width=100, device=device):
        super(Caption_checker, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.height = height
        self.width = width
        self.LSTM_out_size = LSTM_out_size
        
        
        self.layers1 = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1), # three input channels, three output channels, 3x3 window
            nn.BatchNorm2d(3),
            nn.MaxPool2d(2, 2), # 2x2 window with stride of 2
            nn.Tanh()
        )
        
        self.layers2 = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1), # three input channels, three output channels, 3x3 window
            nn.BatchNorm2d(3),
            nn.MaxPool2d(2, 2), # 2x2 window with stride of 2
            nn.Tanh() 
        )
        
        self.layers3 = nn.Sequential(
            nn.Conv2d(3, 3, 4, padding=1), # three input channels, three output channels, 4x4 window
            nn.BatchNorm2d(3),
            nn.Tanh() 
        )
        
                
        self.lstm = nn.LSTM(100, LSTM_out_size, batch_first=True)
        
        self.layers4 = nn.Sequential(                               #3 dense layers to classify the output
            nn.Linear(((int(self.height * self.width * 3/16))+ self.LSTM_out_size), hidden_size),  
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.Tanh(),
            nn.Linear(int(hidden_size/2), 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch):            #batch is a tupple 
        batch_size = batch[0].size()[0]
        
        # permute the channels to the form pytorch expects
        current_matrix = batch[0].float().permute(0, 3, 1, 2)
        current_matrix = self.layers1(current_matrix)
        Im_matrix = self.layers2(current_matrix)
        Im_matrix = Im_matrix.reshape(-1, int(self.height * self.width * 3/16))
        
        h_0 = Variable(torch.zeros(1, batch_size, self.LSTM_out_size).to(device))
        c_0 = Variable(torch.zeros(1, batch_size, self.LSTM_out_size).to(device))

        output, (final_hidden_state, final_cell_state) = self.lstm(batch[1], (h_0, c_0))
        out =final_hidden_state[-1]        
        out_vec = torch.cat((Im_matrix,out),dim=1)
        return self.layers4(out_vec)
    

class Batcher: 
'''when next is called, return a randomized zip iterator containing the list of all batches.
Needed fo the train function.
'''
    def __init__(self, im, cap, y, batch_size=64, max_iter=None):
        self.im = im
        self.cap = cap
        self.y = y
        self.batch_size=batch_size
        self.max_iter = max_iter
        self.curr_iter = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_iter == self.max_iter:
            raise StopIteration
               
        idx_perm = np.random.permutation(self.im.shape[0])
        perm_im = self.im[idx_perm]
        perm_cap = self.cap[idx_perm]
        perm_y = self.y[idx_perm]
        
        list_split_im=[]
        list_split_cap=[]
        list_split_y=[]
        for i in range(0,self.im.shape[0],self.batch_size):
            try:
                split_im = perm_im[i:i+self.batch_size,:,:,:]
                split_cap = perm_cap[i:i+self.batch_size,:,:]
                split_y = perm_y[i:i+self.batch_size]
            except:
                split_im = perm_im[i:,:,:,:]
                split_cap = perm_cap[i:,:,:]
                split_y = perm_y[i:]
            
            list_split_im.append(split_im)
            list_split_cap.append(split_cap)
            list_split_y.append(split_y)
        
        self.curr_iter += 1
        return zip(list_split_im, list_split_cap, list_split_y)
         
    
def train(im_train,cap_train,y_train, im_val, cap_val, y_val, batch_size, epochs, device, model=None):
'''The function used to train the model
-------------------------
param:
im_train : np.array of shape (size_train*100*100*3) representing scaled images of the training set
cap_train : np.array of shape (size_train*25*100) representing the corresponding captions where words are converted to vectors of dim 100 
and the sequence is padded with 0 up to seq_len=25
y_train : int ----------------------- 1 if the caption matches the image
im_val,cap_val,y_val : same for the validation set
batch_size: int --------------------- the desired batch size
epochs : int ------------------------ number of epochs to train
device : torch.device --------------- device to use
model : class Caption_checker ------- the model to train

output:
model : the model trained
train_loss_history : list of float -- the list of trainig loss at each epoch
train_val_history : list of float --- the list of validation loss every 5 epochs

'''
    b = Batcher(im_train,cap_train,y_train, batch_size, max_iter=epochs)
    if not model:
        m =Im_conv_layer(3000, im_train.shape[1], im_train.shape[2]).to(device)
    else:
        m = model
    m.train()
    loss = nn.BCELoss().to(device)
    optimizer = optim.Adam(m.parameters(), lr=0.005)
    epoch = 0
    train_loss_his=[]
    val_loss_his=[]
    for split in b:
        train_acc = []           #keeps trace of the number of matching captions properly predicted in each bach 
        tot_loss = 0             #and the size of the batch
        for batch in split:      
            
                      
            optimizer.zero_grad()
            o = m((torch.tensor(batch[0],dtype=torch.float32, device=device),torch.tensor(batch[1],dtype=torch.float32, device=device)))
            l = loss(o.reshape(batch[0].shape[0]), torch.tensor(batch[2],dtype=torch.float32, device=device))
            tot_loss += l
            l.backward()
            optimizer.step()
            batch_train_accuracy = torch.sum((o.squeeze(-1)>.5)==torch.LongTensor(batch[2]).to(device))
            train_acc.append((batch_train_accuracy,batch[0].shape[0]))
        print("Total loss in epoch {} is {}.".format(epoch, tot_loss))
        train_loss_his.append(tot_loss.cpu().detach().numpy())
        epoch += 1
        
        if epoch %5 == 0:      #every 5 epochs, print the training accuracy, the validation accuracy and loss, 
            np=0                #and save the model param
            tp=0
            for n,denom in train_acc:
                tp+=n
                np+=denom
            train_accuracy = tp/np
            m.eval()
            
            with torch.no_grad():
                b_val = Batcher(im_val,cap_val,y_val,  batch_size, max_iter=1)

                correct = 0
                split = next(b_val) 
                val_acc = []
                l_val=0
                for batch in split:  # Iterate in batches over the validation dataset.


                    o =  m((torch.tensor(batch[0],dtype=torch.float32, device=device),torch.tensor(batch[1],dtype=torch.float32, device=device))) 
                    l_val+=loss(o.reshape(batch[0].shape[0]), torch.tensor(batch[2],dtype=torch.float32, device=device))
                    batch_val_accuracy = torch.sum((o.squeeze(-1)>.5)==torch.LongTensor(batch[2]).to(device))
                    val_acc.append((batch_val_accuracy,batch[0].shape[0]))
                val_loss_his.append(l_val.cpu().detach().numpy())
                tpv=0
                npv=0
                for n,denom in val_acc:
                    tpv+=n
                    npv+=denom
                val_accuracy = tpv/npv
                print('val_loss={}'.format(l_val))
                print('train_acc={}'.format(train_accuracy))
                print('val_acc={}'.format(val_accuracy))
                
                if val_loss_his!=[] and l_val.cpu().detach().numpy()<min(val_loss_his):     #save the model only if the validation loss is smaller than all the precedent
                    for i in range(epoch):                                                  #cleaning before saving
                        if 'caption4'+str(i) in os.listdir(os.getcwd()):
                            os.remove('caption4'+str(i))
                    file_name = 'caption4'+str(epoch)
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': m.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, file_name)
                
            m.train()

    return m, train_loss_his, val_loss_his