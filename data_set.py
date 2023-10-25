import torchvision
from torchvision import datasets , transforms
from torch.utils.data import DataLoader

'''
function to take in train and test data path , along with
transformation and batchsize

and returns train , test dataloaders along with class names

'''

def save_data(train: str,
          test: str,
          transform: transforms.Compose,
          batch_size: int):


  if transform == None:
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        
       transforms.Normalize(mean=(0.5),std=(0.5))
    ])

  train_set = datasets.ImageFolder(train,transform)
  test_set = datasets.ImageFolder(test,transform)

  train_data = DataLoader(dataset= train_set,
                          batch_size=batch_size,
                          shuffle=True)

  test_data = DataLoader(dataset= test_set,
                         batch_size= batch_size,
                         shuffle= False)

  classes = train_set.classes

  return train_data , test_data , classes
