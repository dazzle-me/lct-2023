def batch_to_device(batch, device='cuda:0'):
    for k, v in batch.items():
        if k in ['input', 'input_1', 'input_2', 'input1', 'input2']:
            for k1, v in batch[k].items():
                batch[k][k1] = v.to(device, non_blocking=True)
        elif k in ['label', 'mask', 'embedding', 'vision_embedding', 'text_embedding', 'vision_embedding_1', 'vision_embedding_2', 'text_embedding_2', 'text_embedding_1']:
            batch[k] = v.to(device, non_blocking=True)
    return batch

def describe_batch(batch):
    for k, v in batch.items():
        try:
            print(k, v.shape)
        except:
            print(k, len(v))