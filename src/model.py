from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense

import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf

from constants import NUM_FRAMES

@tf.function
def avg(f,n):
    return f[:,:,::n]

class Concatf0(keras.layers.Layer):
    def __init__(self, n=1):
        self.n = n
        super(Concatf0, self).__init__()
    def call(self, inputs):
        x, f0 = inputs
        f0 = avg(f0, self.n)
        shape = tf.shape(x)
        f0 = tf.repeat(f0, shape[1], 1)
        return tf.concat([x, f0], axis=-1)
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], input_shape[0][2], input_shape[0][3] + 257)
    
class MyConcat(keras.layers.Layer):
    def __init__(self):
        super(MyConcat, self).__init__()
    def call(self, inputs):
        x, emb = inputs
        shape = tf.shape(x)
        emb = tf.repeat(emb, shape[1], 1)
        emb = tf.repeat(emb, shape[2], 2)
        return tf.concat([x, emb], axis=-1)
    
    def compute_output_shape(self, input_shape):
        shape = (None, input_shape[0][1],
                input_shape[0][2], input_shape[0][3] + input_shape[1][3])
        return shape
    
class ACVAE:
    def __init__(self):
        emb_dim = 512
        inputs = Input(shape=(64, None, 1), name="input_enc")
        label = Input(shape=(emb_dim,),)
        emb = Reshape((1,1,emb_dim))(label)

        x = MyConcat()([inputs, emb])

        conv1 = Conv2D(filters=16,
                       kernel_size=(3,9), 
                       strides=(1,1),
                       padding="same")(x)
        conv1_bn = BatchNormalization(axis=1)(conv1)
        conv1_gated = Conv2D(filters=16, 
                             kernel_size=(3,9),
                             strides=(1,1),
                             padding="same")(x)
        conv1_gated_bn = BatchNormalization(axis=1)(conv1_gated)
        conv1_sigmoid = Multiply()([conv1_bn, Activation(sigmoid)(conv1_gated_bn)])

        x = MyConcat()([conv1_sigmoid, emb])

        conv2 = Conv2D(filters=32,
                       kernel_size=(4,8), 
                       strides=(2,2),
                       padding="same")(x)
        conv2_bn = BatchNormalization(axis=1)(conv2)
        conv2_gated = Conv2D(filters=32, 
                             kernel_size=(4,8),
                             strides=(2,2),
                             padding="same")(x)
        conv2_gated_bn = BatchNormalization(axis=1)(conv2_gated)
        conv2_sigmoid = Multiply()([conv2_bn, Activation(sigmoid)(conv2_gated_bn)])

        x = MyConcat()([conv2_sigmoid, emb])

        conv3 = Conv2D(filters=32, 
                       kernel_size=(4,8), 
                       strides=(2,2),
                       padding="same")(x)
        conv3_bn = BatchNormalization(axis=1)(conv3)
        conv3_gated = Conv2D(filters=32,
                             kernel_size=(4,8),
                             strides=(2,2),
                             padding="same")(x)
        conv3_gated_bn = BatchNormalization(axis=1)(conv3_gated)
        conv3_sigmoid = Multiply()([conv3_bn, Activation(sigmoid)(conv3_gated_bn)])

        x = MyConcat()([conv3_sigmoid, emb])
        padding = tf.constant([[0,0], [0,0],[2, 2],[0,0]])
        x = tf.pad(x, padding, "CONSTANT")

        conv4_mu = Conv2D(filters=8,
                          kernel_size=(16,5), 
                          strides=(16,1),
                          padding="valid")(x)
        conv4_logvar = Conv2D(filters=8,
                              kernel_size=(16,5),
                              strides=(16,1),
                              padding="valid")(x)
            
        self.encoder = Model(inputs=[inputs, label], outputs=[conv4_mu, conv4_logvar])       
        # Decoder
        inputs = Input(shape=(1,None, 8), name="input_dec")
        label = Input(shape=(512,), name="label_dec")
        emb = Reshape((1,1,emb_dim))(label)
        f0 = Input(shape=(1,None,257), name="log_f0")
       
        x = MyConcat()([inputs, emb])
        x = Concatf0(4)([x,f0])

        upconv1 = Conv2DTranspose(filters=32, 
                                  kernel_size=(16,5), 
                                  strides=(16,1),
                                  padding="valid")(x)
        upconv1_bn = BatchNormalization(axis=1)(upconv1)
        upconv1_gated = Conv2DTranspose(filters=32, 
                                        kernel_size=(16,5),
                                        strides=(16,1),
                                        padding="valid")(x)
        upconv1_gated_bn = BatchNormalization(axis=1)(upconv1_gated)
        upconv1_sigmoid = Multiply()([upconv1_bn, Activation(sigmoid)(upconv1_gated_bn)])

        upconv1_sigmoid = upconv1_sigmoid[:,:,2:-2,:]
        x = MyConcat()([upconv1_sigmoid, emb])
        x = Concatf0(4)([x,f0])
        
        upconv2 = Conv2DTranspose(filters=32,
                                  kernel_size=(4,8), 
                                  strides=(2,2),
                                  padding="same")(x)
        upconv2_bn = BatchNormalization(axis=1)(upconv2)
        upconv2_gated = Conv2DTranspose(filters=32,
                                        kernel_size=(4,8),
                                        strides=(2,2),
                                        padding="same")(x)
        upconv2_gated_bn = BatchNormalization(axis=1)(upconv2_gated)
        upconv2_sigmoid = Multiply()([upconv2_bn, Activation(sigmoid)(upconv2_gated_bn)])

        x = MyConcat()([upconv2_sigmoid, emb])
        x = Concatf0(2)([x,f0])

        upconv3 = Conv2DTranspose(filters=16,
                                  kernel_size=(4,8),
                                  strides=(2,2), 
                                  padding="same")(x)
        upconv3_bn = BatchNormalization(axis=1)(upconv3)
        upconv3_gated = Conv2DTranspose(filters=16, 
                                        kernel_size=(4,8),
                                        strides=(2,2), 
                                        padding="same")(x)
        upconv3_gated_bn = BatchNormalization(axis=1)(upconv3_gated)
        upconv3_sigmoid = Multiply()([upconv3_bn, Activation(sigmoid)(upconv3_gated_bn)])

        x = MyConcat()([upconv3_sigmoid, emb])
        x = Concatf0()([x,f0])

        upconv4_mu = Conv2DTranspose(filters=1,
                                     kernel_size=(3,9),
                                     strides=(1,1),
                                     padding="same")(x)

        self.decoder = Model(inputs=[inputs, label, f0], outputs=upconv4_mu)
        #reconstruction model
        inputs = Input(shape=(64, None, 1))
        emb_s = Input(shape=(512,))
        emb_t = Input(shape=(512,))
        f0_t = Input(shape=(1,None,257))
       
        mu_enc, logvar_enc = self.encoder([inputs, emb_s])
        z_enc = self.reparameterize(mu_enc, logvar_enc)
        mu_dec = self.decoder([z_enc, emb_t, f0_t])
        self.model = Model(inputs=[inputs, emb_s, emb_t, f0_t], outputs=[mu_dec, mu_enc, logvar_enc])
    
    def reparameterize(self, mu, logvar):
        return tf.random.normal(shape=tf.shape(logvar),
                                mean=mu, 
                                stddev=tf.exp(0.5 * logvar))