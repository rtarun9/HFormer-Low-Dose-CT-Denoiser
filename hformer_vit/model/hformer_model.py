import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np

from tf_data_importer import load_training_tf_dataset

from sklearn.model_selection import train_test_split

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# For sake of efficiency, tf.image has functions that will extract patches from images.
# Creating a custom layer for ease of use when creating the model.
# https://dzlab.github.io/notebooks/tensorflow/vision/classification/2021/10/01/vision_transformer.html


# Takes as input patch_size, which will be same along width and height
# As hformer uses overlapping slices to increase number of training samples, set the stride to a values less than patch_size.

# NOTE : The output shape is num_patches, patch_height, patch_width, patch_depth.
# The Fig1 of paper mentions input to input projection layer is 1 X H X W. Output of this layer is H X W X 1

class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size, stride,name):
        super(PatchExtractor, self).__init__(name=name)
        self.patch_size = patch_size
        self.stride = stride

    def call(self, images):
        # batch_size : number of images, which is not used.
        batch_size = tf.shape(images)[0] 
        # Expected to always be 1.
        patch_depth = images.shape[-1]
        
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [-1, self.patch_size, self.patch_size, patch_depth])
        return patches
    
# NOTE : The paper does not mention what constitutes the input projection block.
# The ONLY thing mentioned in input is each image patch (that is 64x64). Input to the input projection block is 1 X H X W, and output is C X H X W

# NOTE (ASSUMPTIONS) : The input 'projection' block uses convolution (3x3) and C is the number of kernels.
# I chose 3x3 since the paper does not say what H, W is, and C is assumed to be 64 (not mentioned in paper again)

# num_output_channels = C in the block diagram. NO activation function is used (assumed linear).
class InputProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, num_output_channels, kernel_size=(3, 3), name="input_projection_layer", **kwargs):
        super(InputProjectionLayer, self).__init__(name=name, **kwargs)

        # Define the convolutional layer
        self.convolution = tf.keras.layers.Conv2D(filters=num_output_channels,
                                                  kernel_size=kernel_size,
                                                  padding='same', name="convolution_layer")

    def call(self, inputs):
        output = self.convolution(inputs)
        return output
    
    
    
    
# NOTE : The paper does not mention what constitutes the output projection block.
# The ONLY thing mentioned in input is of shape C X H X W and output is 1 X H X W

# NOTE (ASSUMPTIONS) : The output 'projection' block uses transposed convolution (3x3) and C is the value of C (num_features_maps).

# NO activation function is used (assumed linear).
# kernel_size must MATCH whatever was given to InputProjectionLayer
class OutputProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3, 3), name="output_projection_layer", **kwargs):
        super(OutputProjectionLayer, self).__init__(name=name, **kwargs)

        # Define the deconvolutional layer
        self.transpose_conv2d = tf.keras.layers.Conv2DTranspose(filters=1,
                                                             kernel_size=kernel_size,
                                                             padding='same', name="transpose_convolution_2d_layer")

    def call(self, inputs):
        output = self.transpose_conv2d(inputs)
        return output




# Custom tf layer for convolutional block.

# Official / Unofficial documentation for the constituent layers / blocks:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D
# https://github.com/dongkyuk/ConvNext-tensorflow/blob/main/model/convnext.py
# https://keras.io/examples/vision/edsr/
# https://github.com/martinsbruveris/tensorflow-image-models/blob/d5f54175fb91e587eb4d2acf81bb0a7eb4424f4a/tfimm/architectures/convnext.py#L146

# io_num_channels : Number of channels per image / patch for input and output.
class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, io_num_channels, name):
        super(ConvolutionBlock, self).__init__(name=name)
        self.layers = tf.keras.Sequential([
            # Layer 0 : Padding to ensure that the output size is same as input size.
            # Padding is done by 3 because the DWConv will always be 7x7
            tf.keras.layers.ZeroPadding2D(padding=3, name="zero_padding_2d"),
            
            # Layer 1 : Depth wise convolution (7x7)
            tf.keras.layers.DepthwiseConv2D(7, depth_multiplier=1, name="depthwise_conv_2d"),
            
            # Layer 2 : Layer normalization
            tf.keras.layers.LayerNormalization(name="layer_normalization"),
        
            # Layer 3 : Linear (1)
            tf.keras.layers.Conv2D(io_num_channels, kernel_size=1, name="linear_1x1_conv_2d_1"),
            
            # Layer 4 : GELU activation
            tf.keras.layers.Activation('gelu', name="gelu_activation"),
            
            # Layer 5 : Linear (2)
            tf.keras.layers.Conv2D(io_num_channels, kernel_size=1, name="linear_1x1_conv_2d_2"),
        ])
        
    def call(self, images):
        residual = images
                
        # Here, assuming that images are of the shape (num_images, image_height, image_width, num_channels)
        output = self.layers(images) + residual
        return output        
    
    
    
# Note  : Max pooling kernel size is taken as value for stride in the max pooling layer.
# This is because we want NO overlaps between the 'pools' used in maxpooling.

# NOTE : depth_wise_conv_kernel_size will always be 7x7.
# NOTE : Max pool kernel size is basically 'k' from the diagram
# Attention scaling factor depends on the network depth. It is the (1 / sqrt(dk)) that is mentioned in the paper.
# num_channels = (C in the diagram)
class HformerBlock(tf.keras.layers.Layer):
    def __init__(self, maxpool_kernel_size, attention_scaling_factor, num_channels, name, **kwargs):
        super(HformerBlock, self).__init__(name=name, **kwargs)

        # Saving the layer input parameters.
        self.k = maxpool_kernel_size
        self.dk = attention_scaling_factor
        self.c = num_channels

        # Defining the layers required by the HformerBlock.

        # Layers for path A (i.e DWConv, Norm, Linear, Gelu, Linear)        
     
        # Depth wise conv (i.e applying a separate filter for each image channel)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            (7, 7),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            name="depth_wise_conv_2d")
    
        # Defining the layers for path B (i.e max pooling, self attention, linear)

        # Max pooling layers (will be used in attention computation for Keys and Queries)
        self.maxpool = tf.keras.layers.MaxPooling2D((maxpool_kernel_size, maxpool_kernel_size), strides=(maxpool_kernel_size, maxpool_kernel_size), name="max_pooling_layer")

        # (Q, K, V) are used for attention.
        # This is clear from the paper which mentions :
        # The (original) self-attentive mechanism first generates the corresponding query e
        # (q), Key (K) and Value (V)
        # The output of K and Q linear are : HW/k^2 X C
        # Note that the vectors (output of self.k/q/v_linear) are PER channel (C).        
        self.k_linear = tf.keras.layers.Dense(self.c, use_bias=False, name="k_linear")
        self.q_linear = tf.keras.layers.Dense(self.c, use_bias=False, name="q_linear")
        self.v_linear = tf.keras.layers.Dense(self.c, use_bias=False, name="v_linear")

        self.self_attention_linear = tf.keras.layers.Conv2D(num_channels, kernel_size=1, name="self_attention_linear")
        
        # Defining the layers for the stage where input is output of path A and path B
        
        self.norm = tf.keras.layers.LayerNormalization(name="layer_normalization")        
        self.linear_1 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, name="linear_1x1_conv_2d_1")
        self.gelu = tf.keras.layers.Activation('gelu', name="gelu_activation")
        self.linear_2 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, name="linear_1x1_conv_2d_2")

    def call(self, images):
        # NOTE : The shape of images MUST be num_images, image_height, image_width, num_channels (which is already set in the constructor)
        
        # In the Hformer block, the input takes 2 paths.
        # In path A, images undergoe depth wise convolution (DWConv based perceptual module)
        # In path B, images undergoe a transformer module with lightweight self attention module (i.e self attention after maxpooling)
        
        # Code for path A.
        path_a_output = self.depthwise_conv(images)
                
        # Code for path B.
        max_pooled_images = self.maxpool(images)
                    
        # Attention module computation.
        # Q, K, V all pass through linear layers. 
        # q and k are reshaped into HW / (max pool kernel size)^2 X C
        
        # The shape of max pooled images is : (num_images, height', width', channels).
        # We are converting that into (num_images, height' * width', channel)
        flattened_max_pooled_images = tf.reshape(max_pooled_images, (tf.shape(max_pooled_images)[0], max_pooled_images.shape[1] * max_pooled_images.shape[2], self.c))
        
        q = self.q_linear(flattened_max_pooled_images)
        # For the computation of attention map, we need shape of q to be num_images, num_channels, HW.
        # But now, it is num_images, HW, num_channels. The last 2 dimensions must be reversed.
        q = tf.transpose(q, perm=[0, 2, 1])
            
        k = self.k_linear(flattened_max_pooled_images)

        flattened_images = tf.reshape(images, (tf.shape(images)[0], images.shape[1] * images.shape[2], self.c))
                
        v = self.v_linear(flattened_images)

        # Computation of attention
        # attention = ((K'T Q) / sqrt(dk)) V
        # Ignoring num_images, shape of Q is (num_channels)
        attention = tf.matmul(q, k)
                
        # As per paper, after performing softmax, we obtain a attention score matrix with dimension C x C.
        # The scaling factor mentioned in the paper (i.e 1/sqrt(dk)) is based on network depth.
        attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(self.dk, tf.float32)))

        # Now, the final attention map is obtained by multiplied v and attention.
        attention = tf.matmul(v, attention)

        # Now, attention is reshaped into the same dimensions as the input image.
        path_b_output = tf.reshape(attention, (-1, path_a_output.shape[1], path_a_output.shape[2], path_a_output.shape[3]))
        
        combined_path_output = path_a_output + path_b_output 
                    
        x = combined_path_output
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
                
        return x
    

# Documentation of subclassing API : https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing

# Note : Assumes x and y data are already split into image patches (overlapping or non overlapping)
class HformerModel(tf.keras.Model):
    def __init__(self, num_channels_to_be_generated, name):
        super().__init__(name=name)
        
        self.input_projection = InputProjectionLayer(num_output_channels=num_channels_to_be_generated, kernel_size=(3,3), name="input_projection_layer")
        self.output_projection = OutputProjectionLayer(kernel_size=(3,3), name="output_projection_layer")
        
        self.conv_net_block_1 = ConvolutionBlock(io_num_channels=num_channels_to_be_generated, name="conv_net_block_1")
        self.conv_net_block_2 = ConvolutionBlock(io_num_channels=num_channels_to_be_generated, name="conv_net_block_2")
        
        # As per the paper, 2x2 conv layers are used for both upsampling and downsampling.
        self.down_sampling_layer_1 = tf.keras.layers.Conv2D(num_channels_to_be_generated * 2, (2,2), (2,2), name="downsampling_layer_1")
        self.down_sampling_layer_2 = tf.keras.layers.Conv2D(num_channels_to_be_generated * 4, (2,2), (2,2), name="downsampling_layer_2")
        
        # Conv2D transpose : Deconv layer
        self.up_sampling_layer_1 = tf.keras.layers.Conv2DTranspose(num_channels_to_be_generated * 2, (2, 2), (2, 2), name="upsampling_layer_1")
        self.up_sampling_layer_2 = tf.keras.layers.Conv2DTranspose(num_channels_to_be_generated, (2, 2), (2, 2), name="upsampling_layer_2")
        
        self.hformer_block_1 = HformerBlock(maxpool_kernel_size=2, attention_scaling_factor=1, num_channels=num_channels_to_be_generated * 2, name="hformer_block_1")
        self.hformer_block_2 = HformerBlock(maxpool_kernel_size=2, attention_scaling_factor=1, num_channels=num_channels_to_be_generated * 4, name="hformer_block_2")
        self.hformer_block_3 = HformerBlock(maxpool_kernel_size=2, attention_scaling_factor=1, num_channels=num_channels_to_be_generated * 2, name="hformer_block_3")
    
    def call(self, images):
        
        # Split image into patches
        # image_patches = self.patch_extraction(images)
        # The model assumes this has been done already.
        image_patches = images
        
        x = self.input_projection(image_patches)
        
        # First conv block filtering
        conv_block_1_output = self.conv_net_block_1(x)
        
        # Downsampling images from (C X H X W) to (2C X H/2 X W/2)
        x = self.down_sampling_layer_1(conv_block_1_output)
        
        # Hformer block application
        hformer_block_1_output = self.hformer_block_1(x)

        # Downsampling imagesm from (2C X H/2 X W/2) to (4C X H/4 X W/4)
        x = self.down_sampling_layer_2(hformer_block_1_output)
        
        x = self.hformer_block_2(x)
        
        # Upsampling block 1
        x = self.up_sampling_layer_1(x)
        
        # Hformer block application and skip connection.
        x = self.hformer_block_3(x) + hformer_block_1_output
        
        # Upsampling image to 2C X H/2 X W/2
        x = self.up_sampling_layer_2(x)
        
        # Conv block filtering + skip connection.
        x = self.conv_net_block_2(x) + conv_block_1_output        
        x = self.output_projection(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "C": self.C
        })
        return config

    @classmethod
    def from_config(cls, config):
        layer = cls(**config)
        layer.build(input_shape=config["input_shape"])
        return layer
    
def get_hformer_model(num_channels_to_be_generated, name):
    return HformerModel(num_channels_to_be_generated=num_channels_to_be_generated, name=name)