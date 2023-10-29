import torch
from tqdm.auto import tqdm
import torch.nn as nn

def train(model: nn.Module,
          data: torch.utils.data.DataLoader,
          loss: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device):

  ''' training loop for a single epoch

  turns on train mode of model , proceeds to follow the required
  train steps(forward pass , loss calculation , optimizer)

  Args:
  model: network
  data: training data (dataloader)
  loss: loss function
  optimizer: pytorch optimizer'''

  model.train()

  train_loss, train_accuracy = 0,0

  for batch , (x,y) in enumerate(data):
    
    x = x.to(device)
    y = y.to(device)

    prds = model(x)
    lss = loss(prds,y)
    optimizer.zero_grad()
    lss.backward()
    optimizer.step()

    train_loss += lss
    train_accuracy += (y== torch.argmax(prds)).sum().item() / len(y)

  train_loss = train_loss / len(data)
  train_accuracy = train_accuracy / len(data)

  return train_loss , train_accuracy

def test(model: nn.Module,
         data: torch.utils.data.DataLoader,
         loss: nn.Module,
         device = torch.device):

  ''' testing loop for a single epoch

  puts model into inference mode , proceeds to pass the test data
  and calculate its loss

  Args:
  model: pytorch model to be tested
  data: testing data to be tested on
  loss: pytorch loss function'''

  model.eval()

  test_loss, test_accuracy = 0,0

  for batch, (x,y) in enumerate(data):
    
    x,y = x.to(device) , y.to(device)

    prds= model(x)
    lss = loss(prds,y)

    test_loss += lss
    test_accuracy += (torch.argmax(prds) == y).sum().item() / len(y)

  test_loss = test_loss / len(data)
  test_accuracy = test_accuracy / len(data)

  return test_loss , test_accuracy

from tqdm.auto import tqdm

def train_loop(model: nn.Module,
               train_data: torch.utils.data.DataLoader,
               test_data: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer : torch.optim.Optimizer,
               device: torch.device,
               epochs: int):

  '''combines train and test function into final training loop

  passes input data and model into train and test function under a for loop
  for the given number of epochs and stores the loss , accuracy in a results dict

  Args:
  model: model to be trained and tested on
  train_data: data to be trained on
  test_data: data to be tested on
  loss_fn: pytorch loss function
  optimizer: pytorch optimizer to reduce loss function'''

  results = {'train loss':[],'train accu':[],'test loss':[],'test accu':[]}

  for epoch in tqdm(range(epochs)):

    train_loss , train_accu = train(model= model, data = train_data,
                                    loss= loss_fn, optimizer= optimizer, device= device)

    test_loss , test_accu = test(model= model, data= test_data,
                                 loss = loss_fn,device = device)

    print(f"epoch: {epoch+1} | train loss: {train_loss} | test loss: {test_loss}")

    results['train loss'].append(train_loss.detach().numpy())
    results['train accu'].append(train_accu)
    results['test loss'].append(test_loss.detach().numpy())
    results['test accu'].append(test_accu)

  return results
