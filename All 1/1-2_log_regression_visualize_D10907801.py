import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
num_epochs = 1000
learning_rate = 0.1

# Toy dataset
x_train = torch.FloatTensor([[3.3, 1.7], [4.4, 2.76], [5.5, 2.09], [6.71, 3.19], [6.93, 1.694], [4.168, 1.573], 
                    [9.779, 3.366], [6.182, 2.596], [7.59, 2.53], [2.167, 1.221], [7.042, 2.827], 
                    [10.791, 3.465], [5.313, 1.65], [7.997, 2.904], [3.1, 1.3]])

y_train = torch.LongTensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0])

# Linear regression model
class Feedforward(torch.nn.Module):
        def __init__(self):
            super(Feedforward, self).__init__()
            self.input_size = 2
            self.hidden1_size = 4
            self.hidden2_size = 4
            self.output_size = 2
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden1_size)			#weight input-hidden1 layer 
            self.relu1 = torch.nn.ReLU()											#activation fn for #hidden layer 1
            self.fc2 = torch.nn.Linear(self.hidden1_size, self.hidden2_size)		#weight hidden1-hidden2 layer
            self.relu2 = torch.nn.ReLU()											#activation fn for #hidden layer 2
            self.fc3 = torch.nn.Linear(self.hidden2_size, self.output_size)			#weight hidden2-output layer
            

        def forward(self, x):
            hidden1 = self.fc1(x)					#hidden layer 1
            relu1 = self.relu1(hidden1)				#FF activation fn for #hidden layer 1
            hidden2 = self.fc2(relu1)				#hidden layer 2
            relu2 = self.relu2(hidden2)				#FF activation fn for #hidden layer 2
            output = self.fc3(relu2)				#output layer
            return output

def test(model, data, target):
    model.eval()
    correct = 0
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('Accuracy: {}/{} - {}%'.format(correct, len(pred), (correct/len(pred))*100))

# Loss and optimizer
model=Feedforward()
print(model)
for p in model.parameters():
    p.data.fill_(0.02)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = x_train
    targets = y_train

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    test(model, inputs, targets)

# Plot the graph
predicted = model(x_train).argmax(dim=1, keepdim=True).detach().numpy().flatten()
plt.subplot(1,2,1)
plt.title('Ground Truth')
plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.subplot(1,2,2)
plt.title('Predict')
plt.scatter(x_train[:,0], x_train[:,1], c=predicted)
plt.show()