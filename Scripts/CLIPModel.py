import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import CFG
from CFG import *
class ImageEncoder(nn.Module):
  """
  Encode images to a fixed size vector
  """

  def __init__(
      self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
  ):
      super().__init__()
      self.model = timm.create_model(
          model_name, pretrained, num_classes=0, global_pool="avg"
      )
      for p in self.model.parameters():
          p.requires_grad = trainable

  def forward(self, x):
      return self.model(x)

class ProjectionHead(nn.Module):
  def __init__(
      self,
      embedding_dim,
      projection_dim=CFG.projection_dim,
      dropout=CFG.dropout
  ):
      super().__init__()
      self.projection = nn.Linear(embedding_dim, projection_dim)
      self.gelu = nn.GELU()
      self.fc = nn.Linear(projection_dim, projection_dim)
      self.dropout = nn.Dropout(dropout)
      self.layer_norm = nn.LayerNorm(projection_dim)
  
  def forward(self, x):
      projected = self.projection(x)
      x = self.gelu(projected)
      x = self.fc(x)
      x = self.dropout(x)
      x = x + projected
      x = self.layer_norm(x)
      return x

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.temperature = temperature

    def forward(self, batch):
      # Getting Image and Text Features
      sketch_features = self.image_encoder(batch["sketch"])
      icon_features = self.image_encoder(batch["icon"])

      # Getting Image and Icon Embeddings (with same dimension)
      sketch_embeddings = self.image_projection(sketch_features)
      icon_embeddings = self.image_projection(icon_features)

      # Calculating the Loss
      logits = (icon_embeddings @ sketch_embeddings.T) / self.temperature
      sketch_similarity = sketch_embeddings @ sketch_embeddings.T
      texts_similarity = icon_embeddings @ icon_embeddings.T
      targets = F.softmax(
          (sketch_similarity + texts_similarity) / 2 * self.temperature, dim=-1
      )
      icon_loss = cross_entropy(logits, targets, reduction='none')
      sketch_loss = cross_entropy(logits.T, targets.T, reduction='none')
      loss =  (sketch_loss + icon_loss) / 2.0 # shape: (batch_size)
      return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
  log_softmax = nn.LogSoftmax(dim=-1)
  loss = (-targets * log_softmax(preds)).sum(1)
  if reduction == "none":
      return loss
  elif reduction == "mean":
      return loss.mean()