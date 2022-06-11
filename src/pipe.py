from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn

# imagenet_path = 0
# img_paths = 0
# img_labels = 0


@pipeline_def
def create_dali_pipeline(data_dir, im_paths, im_labels, crop, size, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(
        file_root=data_dir,
        files=im_paths,
        labels=im_labels,
        # shard_id=shard_id,
        # num_shards=num_shards,
        random_shuffle=True,
        # shuffle_after_epoch=True,
        name="Reader"
        )

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'

    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid
    # re-allocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid re-allocations
    # in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device,
                                               output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=1,
                                               random_area=1,
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_LINEAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_LINEAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      scale=1/255.0,
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels
