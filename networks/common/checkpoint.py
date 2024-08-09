from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch
import os
from glob import glob
from collections import OrderedDict
from networks.common.io_tools import _remove_recursively, _create_directory


def load(model, optimizer, scheduler, resume, path, logger):
  '''
  Load checkpoint file
  '''

  # If not resume, initialize model and return everything as it is
  if not resume:
    logger.info('=> No checkpoint. Initializing model from scratch')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model.module.weights_init()
    else:
        model.weights_init()
    print('1')
    epoch = 1
    return model, optimizer, scheduler, epoch

  # If resume, check that path exists and load everything to return
  else:
    file_path = glob(os.path.join(path, '*.pth'))[0]
    assert os.path.isfile(file_path), '=> No checkpoint found at {}'.format(path)
    checkpoint = torch.load(file_path, map_location='cpu')
    epoch = checkpoint.pop('startEpoch')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
      # model.module.load_state_dict(checkpoint.pop('model'))
      model.load_state_dict(checkpoint.pop('model'))
    else:
      model.load_state_dict(checkpoint.pop('model'))
      # model.modules.load_state_dict(checkpoint.pop('model'))
    optimizer.load_state_dict(checkpoint.pop('optimizer'))
    scheduler.load_state_dict(checkpoint.pop('scheduler'))
    logger.info('=> Continuing training routine. Checkpoint loaded at {}'.format(file_path))
    return model, optimizer, scheduler, epoch


def load_poss(model, optimizer, scheduler, resume, path, logger):
  '''
  Load checkpoint file
  '''

  # If not resume, initialize model and return everything as it is
  if not resume:
    logger.info('=> No checkpoint. Initializing model from scratch')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model.module.weights_init()
    else:
        model.weights_init()
    print('1')
    epoch = 1
    return model, optimizer, scheduler, epoch

  # If resume, check that path exists and load everything to return
  else:
    file_path = glob(os.path.join(path, '*.pth'))[0]
    assert os.path.isfile(file_path), '=> No checkpoint found at {}'.format(path)
    checkpoint = torch.load(file_path, map_location='cpu')
    epoch = checkpoint.pop('startEpoch')
    state_dict = checkpoint.pop('model')
    del state_dict['module.SegNet.logits.weight']
    del state_dict['module.SegNet.logits.bias']
    del state_dict['module.logits.weight']
    del state_dict['module.logits.bias']
    if isinstance(model, (DataParallel, DistributedDataParallel)):
      # model.module.load_state_dict(checkpoint.pop('model'))
      # model.load_state_dict(checkpoint.pop('model'))
       model.load_state_dict(state_dict, strict=False)
    else:
      # model.load_state_dict(checkpoint.pop('model'))
      # model.modules.load_state_dict(checkpoint.pop('model'))
       model.load_state_dict(state_dict, strict=False)
    optimizer.load_state_dict(checkpoint.pop('optimizer'))
    scheduler.load_state_dict(checkpoint.pop('scheduler'))
    logger.info('=> Continuing training routine. Checkpoint loaded at {}'.format(file_path))
    return model, optimizer, scheduler, epoch


def load_model(model, filepath, logger):
  '''
  Load checkpoint file
  '''

  # check that path exists and load everything to return
  assert os.path.isfile(filepath), '=> No file found at {}'
  checkpoint = torch.load(filepath)

  if isinstance(model, (DataParallel, DistributedDataParallel)):
    model.module.load_state_dict(checkpoint.pop('model'))
    # model.load_state_dict(checkpoint.pop('model'))
  else:
    model.load_state_dict(checkpoint.pop('model'))
  logger.info('=> Model loaded at {}'.format(filepath))
  return model

def load_model_v2(model, filepath, logger):
  '''
  Load checkpoint file
  '''

  # check that path exists and load everything to return
  assert os.path.isfile(filepath), '=> No file found at {}'
  checkpoint = torch.load(filepath)
  state_dict = checkpoint.pop('model')
  new_dict = OrderedDict()
  for k, v in state_dict.items():
    new_dict[k[7:]] = v
  model.load_state_dict(new_dict)
  # if isinstance(model, (DataParallel, DistributedDataParallel)):
  #   model.module.load_state_dict(checkpoint.pop('model'))
  #   # model.load_state_dict(checkpoint.pop('model'))
  # else:
  #   model.load_state_dict(checkpoint.pop('model'))
  logger.info('=> Model loaded at {}'.format(filepath))
  return model

def save(path, model, optimizer, scheduler, epoch, config):
  '''
  Save checkpoint file
  '''

  # Remove recursively if epoch_last folder exists and create new one
  # _remove_recursively(path)
  _create_directory(path)

  weights_fpath = os.path.join(path, 'weights_epoch_{}.pth'.format(str(epoch).zfill(3)))

  torch.save({
    'startEpoch': epoch+1,  # To start on next epoch when loading the dict...
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'config_dict': config
  }, weights_fpath)

  return weights_fpath