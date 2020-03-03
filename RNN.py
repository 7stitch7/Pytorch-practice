import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data  as Data
import os

EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = False   # set to True if haven't download the data

# load data
if not(os.path.exists('./minist/')) or not os.listdir('./minist/'):
    DOWNLOAD_MNIST = True

train_data = dsets.MNIST(
    root = './minist/',
    train = True,
    transform = transforms.ToTensor(),   # convert a PIL.image into numpy array
    download = DOWNLOAD_MNIST
)

# plot an example
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# mini-batch
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

# cross-validation
test_data = dsets.MNIST(root='./minist/',train = False)
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy()[:2000]

# build a RNN net
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn = nn.LSTM(     # input shape (1,28,28)
            input_size= INPUT_SIZE,
            hidden_size= 64,
            num_layers= 1,
            batch_first= True       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )

        self.out = nn.Linear(64,10)
    def forward(self,x):
        r_out,(h_n,h_c) = self.rnn(x,None)      # None represents zero initial hidden state, h_n is 分线层的hidden state,h_c is 主线层的hidden state
        out = self.out(r_out[:, -1, :])     # r_out中储存了所有时刻的output,这里选取最后一个时刻的output,(batch, time_step, input_size)
        return out

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(),lr= LR)
loss_fun = nn.CrossEntropyLoss()




# training and testing
for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x = b_x.view(-1,28,28)
        output = rnn(b_x)
        loss = loss_fun(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    # apply gradients

        if step % 50 ==0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.numpy()
            accuracy = float((pred_y==test_y).astype(int).sum())/float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')