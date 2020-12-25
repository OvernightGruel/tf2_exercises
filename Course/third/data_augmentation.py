import tensorflow as tf
import tensorflow_datasets as tfds


ds, ds_info = tfds.load('colorectal_histology',
                        split='train',
                        shuffle_files=True,
                        download=True,
                        with_info=True)
print(ds_info)
