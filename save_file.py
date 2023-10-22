
import torch
from pathlib import Path

def save_model(file_name: str,
               model: nn.Module,
               model_dir: str):

  ''' a function to save the given pytorch model
  in the particular file name and directory

  Args:
  file_name: name the file should be named after
  model: the actual pytroch model,
  model_dir: directory location where file should be stored'''

  model_final_path = model_dir
  model_final_path = Path(model_final_path)
  if model_final_path.is_dir() != True:
    model_final_path.mkdir(parents=True,exist_ok=True)

  torch.save(obj= model.state_dict(), f=model_final_path/file_name)
