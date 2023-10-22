
import matplotlib.pyplot as plt

def plot(results):

  ''' plots epochs vs loss and epochs vs accuracy count
  takes results dictionay as input'''

  plt.figure(figsize=(7,5))

  epochs = range(1,11)
  plt.subplot(1,2,1)
  plt.plot(epochs , results['train loss'], label='train loss')
  plt.plot(epochs, results['test loss'],label='test loss')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()

  plt.subplot(1,2,2)
  plt.plot(epochs , results['train accu'],label='train accuracy')
  plt.plot(epochs,results['test accu'],label='test accuracy')
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.legend()
