#!/usr/bin/env python
# coding: utf-8

# In[42]:


import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

import torchvision.transforms as T


# In[51]:


import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("./config.ini")


# In[3]:




# In[52]:


DATA_DIR = config["values"]["data_dir"]
OUTPUT_DIR = config["values"]["output_dir"]
image_size = (int(config["values"]["image_size"]),int(config["values"]["image_size"]))
max_samples =  None if config["values"]["max_samples"]=='' else int(config["values"]["max_samples"])
max_objects = int(config["values"]["max_objects_per_image"])
use_orphaned_objects = config.getboolean("values", "use_orphaned_objects", )
include_relationships = config.getboolean("values", "include_relationships", )


batch_size = int(config.get("hparams", "batch_size"))

num_workers = int(config.get("hparams", "num_workers"))


# In[5]:


from graphmuse import data_loader


# In[6]:


float_dtype = torch.cuda.FloatTensor
long_dtype = torch.cuda.LongTensor


# In[7]:


config["values"]["image_size"], image_size


# In[8]:


config.read("./config.ini",)


# In[9]:


## building  dataset
with open(os.path.join(DATA_DIR, 'vocab.json'), 'r') as f:
    vocab = json.load(f)





train_dset = data_loader.VgSceneGraphDataset(**{
        'vocab': vocab,
        'h5_path': os.path.join(DATA_DIR, 'train.h5'),
        'image_dir': os.path.join(DATA_DIR),
        'image_size': image_size,
        'max_samples': max_samples,
        'max_objects': max_objects,
        'use_orphaned_objects': use_orphaned_objects,
        'include_relationships': include_relationships,
        
    })




iter_per_epoch = len(train_dset) // batch_size
print('There are %d iterations per epoch' % iter_per_epoch)

val_dset = data_loader.VgSceneGraphDataset(**{
        'vocab': vocab,
        'h5_path': os.path.join(DATA_DIR, 'val.h5'),
        'image_dir': os.path.join(DATA_DIR, 'VG_100K'),
        'image_size': image_size,
        'max_objects': max_objects,
        'use_orphaned_objects': use_orphaned_objects,
        'include_relationships': include_relationships,
        
    })

iter_per_epoch = len(val_dset) // batch_size
print('There are %d iterations per epoch in Val Set' % iter_per_epoch)




# In[14]:


train_loader = DataLoader(train_dset, batch_size=batch_size, num_workers = num_workers, shuffle = True, collate_fn = data_loader.vg_collate_fn)


# In[15]:


imgs, objs, boxes, triples, obj_to_img, triple_to_img = next(iter(train_loader))


# ## Model

# In[16]:


def boxes_to_layout(vecs, boxes, obj_to_img, H, W=None, pooling='sum'):

    
    O, D = vecs.size()
    if W is None:
        W = H

    
    

    boxes = boxes.view(O, 4, 1, 1)

    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]
    ww = x1 - x0
    hh = y1 - y0

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww   
    Y = (Y - y0) / hh   


    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)


    grid = grid.mul(2).sub(1)


    img_in = vecs.view(O, D, 1, 1).expand(O, D, 8, 8)
    sampled = F.grid_sample(img_in, grid, align_corners=True)   # (O, D, H, W)


    dtype, device = sampled.dtype, sampled.device
    O, D, H, W = sampled.size()
    N = obj_to_img.data.max().item() + 1


    out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
    idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
    out = out.scatter_add(0, idx, sampled)
    
    return out
    
def masks_to_layout(vecs, boxes, masks, obj_to_img, H, W=None, pooling='sum'):

  O, D = vecs.size()
  M = masks.size(1)

  if W is None:
    W = H

  grid = _boxes_to_grid(boxes, H, W)

  img_in = vecs.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)
  sampled = F.grid_sample(img_in, grid)

  out = _pool_samples(sampled, obj_to_img, pooling=pooling)
  return out


def _boxes_to_grid(boxes, H, W):

  O = boxes.size(0)

  boxes = boxes.view(O, 4, 1, 1)


  x0, y0 = boxes[:, 0], boxes[:, 1]
  x1, y1 = boxes[:, 2], boxes[:, 3]
  ww = x1 - x0
  hh = y1 - y0

  X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
  Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)
  
  X = (X - x0) / ww   # (O, 1, W)
  Y = (Y - y0) / hh   # (O, H, 1)
  
 
  X = X.expand(O, H, W)
  Y = Y.expand(O, H, W)
  grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)


  grid = grid.mul(2).sub(1)

  return grid


def _pool_samples(samples, obj_to_img, pooling='sum'):

  dtype, device = samples.dtype, samples.device
  O, D, H, W = samples.size()
  N = obj_to_img.data.max().item() + 1
  
  # Use scatter_add to sum the sampled outputs for each image
  out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
  idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
  out = out.scatter_add(0, idx, samples)

  if pooling == 'avg':
    # Divide each output mask by the number of objects; use scatter_add again
    # to count the number of objects per image.
    ones = torch.ones(O, dtype=dtype, device=device)
    obj_counts = torch.zeros(N, dtype=dtype, device=device)
    obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
    print(obj_counts)
    obj_counts = obj_counts.clamp(min=1)
    out = out / obj_counts.view(N, 1, 1, 1)
  elif pooling != 'sum':
    raise ValueError('Invalid pooling "%s"' % pooling)

  return out
    



class cnn(nn.Module):
    def __init__(self, dim_layout, dim_input, dim_output):
        super(cnn, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(dim_layout+dim_input, dim_output, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(dim_output))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(dim_output, dim_output, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(dim_output))
        layers.append(nn.LeakyReLU())
        for l in layers:
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_normal_(l.weight)
        self.model = nn.Sequential(*layers)
        
    def forward(self, layout, feats):

        _, _, HH, WW = layout.size()
        _, _, H, W = feats.size()
        
        if HH > H:
            f = round(HH // H)

            layout = F.avg_pool2d(layout, kernel_size=f, stride=f)
            
            ##
            
        net_input = torch.cat([layout, feats], dim=1)
        out = self.model(net_input)
        return out


# In[18]:


class CnnNetwork(nn.Module):
    def __init__(self, dims,):
        super(CnnNetwork, self).__init__()
        
        dim_layout = dims[0]
        
        self.cnn_modules = nn.ModuleList()
        
        
        for i in range(1, len(dims)):
            dim_input = 1 if i == 1 else dims[i - 1]
            dim_output = dims[i]
            
            mod = cnn(dim_layout, dim_input, dim_output)
            self.cnn_modules.append(mod)
            
            
        output_conv_layers = [
                    nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0)
        ]
        nn.init.kaiming_normal_(output_conv_layers[0].weight)
        nn.init.kaiming_normal_(output_conv_layers[2].weight)
        self.output_conv = nn.Sequential(*output_conv_layers)
        
    def forward(self, layout):

        N, _, H, W = layout.size()
        self.layout = layout

        # Figure out size of input
        input_H, input_W = H, W
        for _ in range(len(self.cnn_modules)):
            input_H //= 2
            input_W //= 2


        feats = torch.zeros(N, 1, input_H, input_W).to(layout)

        for mod in self.cnn_modules:
            feats = F.upsample(feats, scale_factor=2, mode='nearest')
            feats = mod(layout, feats)

        out = self.output_conv(feats)
        return out


# In[19]:


class ResidualBlock(nn.Module):
  def __init__(self, channels, activation='relu',
               padding='same', kernel_size=3, init='default'):
    super(ResidualBlock, self).__init__()

    K = kernel_size
    P = _get_padding(K, padding)
    C = channels
    self.padding = P
    layers = [
      nn.BatchNorm2d(C),
      nn.LeakyReLU(),
      nn.Conv2d(C, C, kernel_size=K, padding=P),
      nn.BatchNorm2d(C),
      nn.LeakyReLU(),
      nn.Conv2d(C, C, kernel_size=K, padding=P),
    ]
    layers = [layer for layer in layers if layer is not None]
    for layer in layers:
         if isinstance(layer, nn.Conv2d):

            nn.init.kaiming_normal_(layer.weight)

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    P = self.padding
    shortcut = x
    if P == 0:
      shortcut = x[:, :, P:-P, P:-P]
    y = self.net(x)
    return shortcut + self.net(x)


def _get_padding(K, mode):

  # Helper function
  if mode == 'valid':
    return 0
  elif mode == 'same':

    return (K - 1) // 2


def build_cnn(arch, normalization='batch', padding='valid',
              pooling='max', init='default'):

  if isinstance(arch, str):
    arch = arch.split(',')
  cur_C = 3
  if len(arch) > 0 and arch[0][0] == 'I':
    cur_C = int(arch[0][1:])
    arch = arch[1:]

  first_conv = True
  flat = False
  layers = []
  for i, s in enumerate(arch):
    if s[0] == 'C':
      if not first_conv:
        layers.append(nn.BatchNorm2d(cur_C))
        layers.append(nn.LeakyReLU())
      first_conv = False
      vals = [int(i) for i in s[1:].split('-')]
      if len(vals) == 2:
        K, next_C = vals
        stride = 1
      elif len(vals) == 3:
        K, next_C, stride = vals
      # K, next_C = (int(i) for i in s[1:].split('-'))
      P = _get_padding(K, padding)
      conv = nn.Conv2d(cur_C, next_C, kernel_size=K, padding=P, stride=stride)
      layers.append(conv)
      nn.init.kaiming_normal_(layers[-1].weight)
        
      
      cur_C = next_C
    elif s[0] == 'R':
      norm = 'none' if first_conv else normalization
      res = ResidualBlock(cur_C, normalization=norm,
                          padding=padding, init=init)
      layers.append(res)
      first_conv = False
    elif s[0] == 'U':
      factor = int(s[1:])
      layers.append(nn.Upsample(scale_factor=factor, mode='nearest'))
    elif s[0] == 'P':
      factor = int(s[1:])
      if pooling == 'max':
        pool = nn.MaxPool2d(kernel_size=factor, stride=factor)
      elif pooling == 'avg':
        pool = nn.AvgPool2d(kernel_size=factor, stride=factor)
      layers.append(pool)
    elif s[:2] == 'FC':
      _, Din, Dout = s.split('-')
      Din, Dout = int(Din), int(Dout)
      if not flat:
        layers.append(Flatten())
      flat = True
      layers.append(nn.Linear(Din, Dout))
      if i + 1 < len(arch):
        layers.append(nn.LeakyReLU())
      cur_C = Dout
    else:
      raise ValueError('Invalid layer "%s"' % s)
  layers = [layer for layer in layers if layer is not None]

  return nn.Sequential(*layers), cur_C


# In[20]:


class GlobalAvgPool(nn.Module):
  def forward(self, x):
    N, C = x.size(0), x.size(1)
    return x.view(N, C, -1).mean(dim=2)

class ObjDiscriminator(nn.Module):
  def __init__(self, vocab, arch,
               padding='valid', pooling='avg'):
    super(ObjDiscriminator, self).__init__()
    self.vocab = vocab

    
    cnn, D = build_cnn(arch,pooling=pooling)
    self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
    num_objects = len(vocab['object_idx_to_name'])

    self.real_classifier = nn.Linear(1024, 1)
    self.obj_classifier = nn.Linear(1024, num_objects)

  def forward(self, x, y):
    if x.dim() == 3:
      x = x[:, None]
    vecs = self.cnn(x)
    real_scores = self.real_classifier(vecs)
    obj_scores = self.obj_classifier(vecs)
    ac_loss = F.cross_entropy(obj_scores, y)
    return real_scores, ac_loss


# In[21]:


class ObjCropDiscriminator(nn.Module):
  def __init__(self, vocab, arch,
               object_size=64, padding='valid', pooling='avg'):
    super(ObjCropDiscriminator, self).__init__()
    self.vocab = vocab
    self.discriminator = ObjDiscriminator(vocab, arch, pooling = pooling)
    self.object_size = object_size

  def forward(self, imgs, objs, boxes, obj_to_img):
    crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
    real_scores, ac_loss = self.discriminator(crops, objs)
    return real_scores, ac_loss


class PatchDiscriminator(nn.Module):
  def __init__(self, arch,
               padding='same', pooling='avg', input_size=(128,128),
               layout_dim=0):
    super(PatchDiscriminator, self).__init__()
    input_dim = 3 + layout_dim
    arch = 'I%d,%s' % (input_dim, arch)

    self.cnn, output_dim = build_cnn(arch,pooling = pooling)
    self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

  def forward(self, x, layout=None):
    if layout is not None:
      x = torch.cat([x, layout], dim=1)
    return self.cnn(x)


# In[22]:


import torch.nn as nn
import torch.nn.functional as F
import math
from graphmuse import gnn

class FinalModel(nn.Module):
    def __init__(self, vocab, image_size=(64,64), embedding_dimension=128, gconv_num_layers = 5, gconv_dim=128,
                gconv_hidden_dim=512, gconv_pooling="avg",mlp_normalization=None,mask_size=16,
                cnn_dims=(1024, 512, 256, 128, 64)):
        super(FinalModel, self).__init__()
        
        self.vocab = vocab
        self.image_size = image_size
        
        n_objs = len(vocab['object_idx_to_name'])
        n_preds = len(vocab['pred_idx_to_name'])
        self.obj_embeddings = nn.Embedding(n_objs + 1, embedding_dimension)
        self.pred_embeddings = nn.Embedding(n_preds, embedding_dimension)
        
        self.gconv = gnn.GraphTripleConv(input_dim=embedding_dimension,
                                        output_dim = gconv_dim,
                                        hidden_dim = gconv_hidden_dim,
                                        pooling = gconv_pooling,
                                        mlp_normalization=mlp_normalization)
        
        self.gconv_net = gnn.GraphTripleConvNet(input_dim=gconv_dim,
                                        hidden_dim = gconv_hidden_dim,
                                        pooling = gconv_pooling,
                                        num_layers = gconv_num_layers-1,
                                        mlp_normalization=mlp_normalization)
        
        
        
        box_net_layers = [gconv_dim, gconv_hidden_dim, 4]
        self.box_net = gnn.build_mlp(box_net_layers, batch_norm=mlp_normalization)
        self.layout_noise_dim=32
        
        self.mask_net = None
        if mask_size is not None and mask_size > 0:
            self.mask_net = self._build_mask_net(n_objs, gconv_dim, mask_size)
        
        
        rel_aux_layers = [2 * embedding_dimension + 8, gconv_hidden_dim, n_preds]
        self.rel_aux_net = gnn.build_mlp(rel_aux_layers, batch_norm=mlp_normalization) # to classify relations
        
        
        self.cnn_net = CnnNetwork(dims = (gconv_dim+self.layout_noise_dim,)+cnn_dims)
        
    def _build_mask_net(self, num_objs, dim, mask_size):
        output_dim = 1
        layers, cur_size = [], 1
        while cur_size < mask_size:
          layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
          layers.append(nn.BatchNorm2d(dim))
          layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
          layers.append(nn.ReLU())
          cur_size *= 2
        if cur_size != mask_size:
          raise ValueError('Mask size must be a power of 2')
        layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
        return nn.Sequential(*layers)
        

    def forward(self, objs, triples, obj_to_img=None,
              boxes_gt=None, masks_gt=None):
        
        
        O, T = objs.size(0), triples.size(0)
        s, p, o = [x.squeeze(1) for x in triples.chunk(3, dim=1) ] # (T,1) --> (T,)
        edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        obj_vecs_orig = obj_vecs
        pred_vecs = self.pred_embeddings(p)
        
        obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges) #layer 1
        obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges) #rest of the layers
        
        ## try pytorch geometric heterogeneous GNN
        
        boxes_pred = self.box_net(obj_vecs)
        
        masks_pred = None
        if self.mask_net is not None:
            mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
            masks_pred = mask_scores.squeeze(1).sigmoid()
        
        s_boxes, o_boxes = boxes_pred[s], boxes_pred[o]
        s_vecs, o_vecs = obj_vecs_orig[s], obj_vecs_orig[o]
        rel_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, o_vecs], dim=1)
        rel_scores = self.rel_aux_net(rel_aux_input)
        
        H, W = self.image_size
        if boxes_gt is not None:
            bbs = boxes_gt
        else:
            bbs = boxes_pred
            
        
        
        if masks_pred is None:
            layout = boxes_to_layout(obj_vecs, bbs, obj_to_img, H, W)
        else:
            layout_masks = masks_pred if masks_gt is None else masks_gt
            layout = masks_to_layout(obj_vecs, bbs, layout_masks,
                               obj_to_img, H, W)
        
        
        if self.layout_noise_dim > 0:
            N, C, H, W = layout.size()
            noise_shape = (N, self.layout_noise_dim, H, W)
            layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                     device=layout.device)
            layout = torch.cat([layout, layout_noise], dim=1)\
            
        img = self.cnn_net(layout)
        
        return img, boxes_pred,masks_pred, rel_scores
        
        
 

# In[23]:


def bce_loss(input, target):

    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()



def gan_g_loss(scores_fake):

  if scores_fake.dim() > 1:
    scores_fake = scores_fake.view(-1)
  y_fake = torch.full_like(scores_fake, 1)
  return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):


  if scores_real.dim() > 1:
    scores_real = scores_real.view(-1)
    scores_fake = scores_fake.view(-1)
  y_real = torch.full_like(scores_real, 1)
  y_fake = torch.full_like(scores_fake, 0)
  loss_real = bce_loss(scores_real, y_real)
  loss_fake = bce_loss(scores_fake, y_fake)
  return loss_real + loss_fake


# In[29]:


model = FinalModel(vocab)
model.type(float_dtype)
learning_rate = 1e-4


object_discriminator = ObjCropDiscriminator(vocab,'C4-64-2,C4-128-2,C4-256-2', object_size=32)
img_discriminator = PatchDiscriminator('C4-64-2,C4-128-2,C4-256-2')





optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

object_discriminator.type(float_dtype)
object_discriminator.train()


img_discriminator.type(float_dtype)
img_discriminator.train()



object_optimizer = torch.optim.Adam(object_discriminator.parameters(),
                                   lr=learning_rate)

img_optimizer = torch.optim.Adam(img_discriminator.parameters(),
                                   lr=learning_rate)


# In[30]:


def calculate_model_losses(model, img, img_pred, bbox, bbox_pred,predicates, predicate_scores,
                           masks=None, masks_pred=None,bbox_pred_loss_weight=10, l1_pixel_loss_weight=1.0, 
                           predicate_pred_loss_weight=0,mask_loss_weight=0,
                           skip_pixel_loss=False):
  total_loss = torch.zeros(1).to(img)
  losses = {}

  l1_pixel_weight = l1_pixel_loss_weight
  if skip_pixel_loss:
    l1_pixel_weight = 0
  l1_pixel_loss = F.l1_loss(img_pred, img)
  total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                        l1_pixel_weight)
  loss_bbox = F.mse_loss(bbox_pred, bbox)
  total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                        bbox_pred_loss_weight)

  if predicate_pred_loss_weight > 0:
    loss_predicate = F.cross_entropy(predicate_scores, predicates)
    total_loss = add_loss(total_loss, loss_predicate, losses, 'predicate_pred',
                          predicate_pred_loss_weight)

  if mask_loss_weight > 0 and masks is not None and masks_pred is not None:
    mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
    total_loss = add_loss(total_loss, mask_loss, losses, 'mask_loss',
                          mask_loss_weight)
  return total_loss, losses

def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
  curr_loss = curr_loss * weight
  loss_dict[loss_name] = curr_loss.item()
  if total_loss is not None:
    total_loss += curr_loss
  else:
    total_loss = curr_loss
  return total_loss

class LossManager(object):
  def __init__(self):
    self.total_loss = None
    self.all_losses = {}

  def add_loss(self, loss, name, weight=1.0):
    cur_loss = loss * weight
    if self.total_loss is not None:
      self.total_loss += cur_loss
    else:
      self.total_loss = cur_loss

    self.all_losses[name] = cur_loss.data.cpu().item()

  def items(self):
    return self.all_losses.items()




def crop_bbox_batch(feats, bbox, bbox_to_feats, HH, WW=None, backend='cudnn'):
  
  if backend == 'cudnn':
    return crop_bbox_batch_cudnn(feats, bbox, bbox_to_feats, HH, WW)
  N, C, H, W = feats.size()
  B = bbox.size(0)
  if WW is None: WW = HH
  dtype, device = feats.dtype, feats.device
  crops = torch.zeros(B, C, HH, WW, dtype=dtype, device=device)
  for i in range(N):
    idx = (bbox_to_feats.data == i).nonzero()
    if idx.dim() == 0:
      continue
    idx = idx.view(-1)
    n = idx.size(0)
    cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
    cur_bbox = bbox[idx]
    cur_crops = crop_bbox(cur_feats, cur_bbox, HH, WW)
    crops[idx] = cur_crops
  return crops


def crop_bbox_batch_cudnn(feats, bbox, bbox_to_feats, HH, WW=None):
  N, C, H, W = feats.size()
  B = bbox.size(0)
  if WW is None: WW = HH
  dtype = feats.data.type()

  feats_flat, bbox_flat, all_idx = [], [], []
  for i in range(N):
    idx = (bbox_to_feats.data == i).nonzero()
    if idx.dim() == 0:
      continue
    idx = idx.view(-1)
    n = idx.size(0)
    cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
    cur_bbox = bbox[idx]

    feats_flat.append(cur_feats)
    bbox_flat.append(cur_bbox)
    all_idx.append(idx)

  feats_flat = torch.cat(feats_flat, dim=0)
  bbox_flat = torch.cat(bbox_flat, dim=0)
  crops = crop_bbox(feats_flat, bbox_flat, HH, WW, backend='cudnn')


  all_idx = torch.cat(all_idx, dim=0)
  eye = torch.arange(0, B).type_as(all_idx)
  if (all_idx == eye).all():
    return crops
  return crops[_invperm(all_idx)]


def crop_bbox(feats, bbox, HH, WW=None, backend='cudnn'):

  N = feats.size(0)
  if WW is None: WW = HH
  if backend == 'cudnn':
    # Change box from [0, 1] to [-1, 1] coordinate system
    bbox = 2 * bbox - 1
  x0, y0 = bbox[:, 0], bbox[:, 1]
  x1, y1 = bbox[:, 2], bbox[:, 3]
  X = tensor_linspace(x0, x1, steps=WW).view(N, 1, WW).expand(N, HH, WW)
  Y = tensor_linspace(y0, y1, steps=HH).view(N, HH, 1).expand(N, HH, WW)
  if backend == 'jj':
    return bilinear_sample(feats, X, Y)
  elif backend == 'cudnn':
    grid = torch.stack([X, Y], dim=3)
    return F.grid_sample(feats, grid)

def tensor_linspace(start, end, steps=10):

  view_size = start.size() + (1,)
  w_size = (1,) * start.dim() + (steps,)
  out_size = start.size() + (steps,)

  start_w = torch.linspace(1, 0, steps=steps).to(start)
  start_w = start_w.view(w_size).expand(out_size)
  end_w = torch.linspace(0, 1, steps=steps).to(start)
  end_w = end_w.view(w_size).expand(out_size)

  start = start.contiguous().view(view_size).expand(out_size)
  end = end.contiguous().view(view_size).expand(out_size)

  out = start_w * start + end_w * end
  return out

def train():
    t = 0
    epoch = 0
    while True:

        for batch in (train_loader):
            batch = [tensor.cuda() for tensor in batch]
            imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch

            relations =  triples[:, 1]

            imgs_pred, boxes_pred, masks_pred,rel_scores = model(objs,triples, obj_to_img, boxes_gt=boxes)

            total_loss, losses =  calculate_model_losses(
                                        model, imgs, imgs_pred,
                                        boxes, boxes_pred,
                                        relations, rel_scores)


            scores_fake, obj_loss = object_discriminator(imgs_pred, objs, boxes, obj_to_img)
            total_loss = add_loss(total_loss, obj_loss, losses, 'obj_loss',0.1)

            total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                                      'g_gan_obj_loss', 0.1)


            scores_fake = img_discriminator(imgs_pred)

            total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                                  'g_gan_img_loss', 0.1)

            losses['total_loss'] = total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()






            total_loss_d = None
            ac_loss_real = None
            ac_loss_fake = None
            d_losses = {}

            d_obj_losses = LossManager()
            imgs_fake = imgs_pred.detach()
            scores_fake, ac_loss_fake = object_discriminator(imgs_fake, objs, boxes, obj_to_img)
            scores_real, ac_loss_real = object_discriminator(imgs, objs, boxes, obj_to_img)

            d_obj_gan_loss = gan_d_loss(scores_real, scores_fake)
            d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
            d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
            d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

            object_optimizer.zero_grad()
            d_obj_losses.total_loss.backward()
            object_optimizer.step()


            d_img_losses = LossManager()
            imgs_fake = imgs_pred.detach()
            scores_fake = img_discriminator(imgs_fake)
            scores_real = img_discriminator(imgs)

            d_img_gan_loss = gan_d_loss(scores_real, scores_fake)
            d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')

            img_optimizer.zero_grad()
            d_img_losses.total_loss.backward()
            img_optimizer.step()

            t+=1
        #     if t%10==3:


            for name, val in losses.items():
                  print(' G [%s]: %.4f' % (name, val))
            for name, val in d_obj_losses.items():
                    print(' D_obj [%s]: %.4f' % (name, val))
            for name, val in d_img_losses.items():
                    print(' D_img [%s]: %.4f' % (name, val))

        #         break
            if t%200== 0:
                checkpoint = {}
                checkpoint['model_state'] = model.state_dict()

                checkpoint['d_obj_state'] = object_discriminator.state_dict()
                checkpoint['d_obj_optim_state'] = object_optimizer.state_dict()


                checkpoint['d_img_state'] = img_discriminator.state_dict()
                checkpoint['d_img_optim_state'] = img_optimizer.state_dict()

                checkpoint['optim_state'] = optimizer.state_dict()

                checkpoint_path = os.path.join(OUTPUT_DIR,
                                      'checking_with_model.pt' )
                print('Saving checkpoint to ', checkpoint_path)
                torch.save(checkpoint, checkpoint_path)


if __name__=="__main__":
    train()

##**Most of the utility codes and helper codes were used from https://github.com/google/sg2im**

