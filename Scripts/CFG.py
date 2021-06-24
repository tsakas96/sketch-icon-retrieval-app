import torch
class CFG:
  debug = False
  sketch_path = "Code-Human-Dataset/sketch"
  icon_path = "Code-Human-Dataset/icon"
  folder_path = "Code-Human-Dataset"
  batch_size = 32
  num_workers = 2
  head_lr = 1e-3
  image_encoder_lr = 1e-4
  weight_decay = 1e-3
  patience = 1
  factor = 0.8
  epochs = 20
  device = torch.device("cpu") # ("cuda" if torch.cuda.is_available() else "cpu")

  model_name = 'resnet50'
  image_embedding = 2048

  pretrained = True
  trainable = True
  temperature = 1.0

  # image size
  size = 224

  # for projection head; used for both image encoder
  num_projection_layers = 1
  projection_dim = 256 
  dropout = 0.1