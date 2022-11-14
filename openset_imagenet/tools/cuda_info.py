import logging
import torch

# instantiate module logger
logger = logging.getLogger('eosa.experiments.eval_algos')


def get_cuda_info(device):
    logger.debug('\n')
    logger.info(f'Using device: {device}')

    if device.type == 'cuda':
        logger.info(torch.cuda.get_device_name(device))
        logger.info('Memory Usage:')
        logger.info(
            f'Allocated: {round(torch.cuda.memory_allocated(device)/1024**3,1)} GB')
        logger.info(
            f'Cached: {round(torch.cuda.memory_reserved(device)/1024**3, 1)} GB')


def get_model_status(model):
    logger.info(
        f'Model sent to GPU: {all(p.is_cuda for p in model.parameters())}')
