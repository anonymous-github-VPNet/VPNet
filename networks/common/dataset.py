from torch.utils.data import DataLoader, DistributedSampler

from networks.data.SemanticKITTI import SemanticKITTI_dataloader, collate_fn_BEV, collate_fn_BEV_test
from networks.data.SemanticPOSS import SemanticPOSS_dataloader

def get_dataset(_cfg):

    if _cfg._dict['DATASET']['TYPE'] == 'SemanticKITTI':
        ds_train = SemanticKITTI_dataloader(_cfg._dict['DATASET'], 'train')
        ds_val = SemanticKITTI_dataloader(_cfg._dict['DATASET'], 'val')
        ds_test = SemanticKITTI_dataloader(_cfg._dict['DATASET'], 'test')
    

    if _cfg._dict['DATASET']['TYPE'] == 'SemanticPOSS':
        ds_train = SemanticPOSS_dataloader(_cfg._dict['DATASET'], 'train')
        ds_val = SemanticPOSS_dataloader(_cfg._dict['DATASET'], 'val')
        ds_test = SemanticPOSS_dataloader(_cfg._dict['DATASET'], 'test')

    _cfg._dict['DATASET']['SPLIT'] = {'TRAIN': len(ds_train), 'VAL': len(ds_val), 'TEST': len(ds_test)}

    dataset = {}

    train_batch_size = _cfg._dict['TRAIN']['BATCH_SIZE']
    val_batch_size = _cfg._dict['VAL']['BATCH_SIZE']
    num_workers = _cfg._dict['DATALOADER']['NUM_WORKERS']
    sampler = DistributedSampler(ds_train)
    dataset['train'] = DataLoader(ds_train, batch_size=train_batch_size, num_workers=num_workers, sampler=sampler, collate_fn=collate_fn_BEV)
    dataset['val'] = DataLoader(ds_val, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=collate_fn_BEV)
    dataset['test'] = DataLoader(ds_test, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=collate_fn_BEV_test)

    return dataset, sampler