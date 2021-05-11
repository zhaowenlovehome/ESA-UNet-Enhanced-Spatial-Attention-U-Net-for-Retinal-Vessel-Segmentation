import tensorflow as tf
import keras.backend as K
import os

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)


    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)


    return [flops.total_float_ops, params.total_parameters]

from UNet import Ghost_UNet
from UNet import ESA_UNet
model = ESA_UNet(input_size=(544, 544, 3), start_neurons=16, lr=1e-3, keep_prob=1, block_size=1)
model.summary()
weight = "Model/DRIVE/MSFF+AR.h5"

if os.path.isfile(weight):
    model.load_weights(weight)
print(get_flops(model))



