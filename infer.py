import argparse

from imageio import imwrite
import torch
from graphmuse.train_py import *


import configparser
config = configparser.ConfigParser(allow_no_value=True)



parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='/vg128.pt')
parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
parser.add_argument('--config_path', default="./config.ini")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

def rescale(x):
  lo, hi = x.min(), x.max()
  return x.sub(lo).div(hi - lo)


def imagenet_deprocess(rescale_image=True):
  transforms = [
    T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
    T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
  ]
  if rescale_image:
    transforms.append(rescale)
  return T.Compose(transforms)


def imagenet_deprocess_batch(imgs, rescale=True):

  if isinstance(imgs, torch.autograd.Variable):
    imgs = imgs.data
  imgs = imgs.cpu().clone()
  deprocess_fn = imagenet_deprocess(rescale_image=rescale)
  imgs_de = []
  for i in range(imgs.size(0)):
    img_de = deprocess_fn(imgs[i])[None]
    img_de = img_de.mul(255).clamp(0, 255).byte()
    imgs_de.append(img_de)
  imgs_de = torch.cat(imgs_de, dim=0)
  return imgs_de



def encode_scene_graphs( scene_graphs,vocab):

    if isinstance(scene_graphs, dict):
      scene_graphs = [scene_graphs]

    objs, triples, obj_to_img = [], [], []
    obj_offset = 0
    for i, sg in enumerate(scene_graphs):
      sg['objects'].append('__image__')
      image_idx = len(sg['objects']) - 1
      for j in range(image_idx):
        sg['relationships'].append([j, '__in_image__', image_idx])

      for obj in sg['objects']:
        obj_idx = vocab['object_name_to_idx'].get(obj, None)
        if obj_idx is None:
          raise ValueError('Object "%s" not in vocab' % obj)
        objs.append(obj_idx)
        obj_to_img.append(i)
      for s, p, o in sg['relationships']:
        pred_idx = vocab['pred_name_to_idx'].get(p, None)
        if pred_idx is None:
          raise ValueError('Relationship "%s" not in vocab' % p)
        triples.append([s + obj_offset, pred_idx, o + obj_offset])
      obj_offset += len(sg['objects'])
    device = "cpu"
    objs = torch.tensor(objs, dtype=torch.int64, device=device)
    triples = torch.tensor(triples, dtype=torch.int64, device=device)
    obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
    return objs, triples, obj_to_img


if __name__=="__main__":
    
    args = parser.parse_args()
    
    config.read(args.config_path)

    with open(os.path.join(config["values"]["data_dir"], 'vocab.json'), 'r') as f:
        vocab = json.load(f)
    
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    

    model = FinalModel(vocab)

    model.load_state_dict(checkpoint['model_state'])
    
    model.eval()
    model.to('cpu')
    
    with open(args.scene_graphs_json, 'r') as f:
        scene_graphs = json.load(f)
        
    with torch.no_grad():
        objs, triples, obj_to_img = encode_scene_graphs(scene_graphs, vocab)
        imgs, boxes_pred,_, _ = model.forward(objs, triples, obj_to_img)
        
    imgs = imagenet_deprocess_batch(imgs)
    
    for i in range(imgs.shape[0]):
        img_np = imgs[i].numpy().transpose(1, 2, 0)
        img_path = os.path.join(args.output_dir, 'img%06d.png' % i)
        imwrite(img_path, img_np)
        
 ##**Most of the utility codes and helper codes were used from https://github.com/google/sg2im**
