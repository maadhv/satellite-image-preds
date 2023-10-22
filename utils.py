
import torch
from modular import save_file , data_set, engine , plot_func , network

train_dir = 'data/satellite/train'
test_dir = 'data/satellite/test'

batch_size = 32
epochs = 10

train_data , test_data , classes = data_set.save_data(train=train_dir,
                                                      test=test_dir,
                                                      batch_size= batch_size)

model = network.network()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD()

results = engine.train_loop(model=model,
                            train_data = train_data,
                            test_data= test_data,
                            loss_fn=loss_fn,
                            optimizer= optimizer,
                            epochs=epochs)

plot_func(results)

save_file.save_model(file_name='modelv1.pth',
                     model= model,
                     model_dir='models')
