import tensorflow as tf
import numpy as np
from network_func import get_weight, conv2d, dense, apply_bias, apply_dense_bias, leaky_relu, upscale2d, downscale2d, batchnorm, lerp_clip

#tf.enable_eager_execution()
#tf.executing_eagerly()

class cGAN(object):
    def __init__(
        self, 
        weight_l1           = 0.5,
        weight_GAN          = 0.5,
        lod_in              = 6.0,                         
        num_channels        = 3,            # Number of input color channels. 
        resolution          = 256,          # Input resolution. 
        fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        dtype               = 'float32',    # Data type to use for activations and outputs.
        structure           = 'linear',     # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
        channel_first       = False,        # Channel_first = True: NCHW structure (for GPU use)
        EPS                 = 1e-12,        # Numerical stability. 
        **kwargs):

        self.resolution      = resolution
        self.resolution_log2 = int(np.log2(self.resolution))
        self.num_channels    = num_channels
        self.channel_first   = channel_first
        self.fmap_base       = fmap_base
        self.fmap_max        = fmap_max
        self.fmap_decay      = fmap_decay
        self.structure       = structure
        self.dtype           = dtype 
        self.lod_in          = tf.Variable(name="lod_in", initial_value=6.0, trainable=False, dtype=tf.float32)



        if self.channel_first:
            self.conc_axis = 1
        else:
            self.conc_axis = 3

    def generator(self, labels_in):


        assert self.resolution == 2**self.resolution_log2 and self.resolution >= 4    
        def nf(stage): return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

        # First Layer from Image: A x A x 3 ==> A x A x Channels(A) 
        # -------------------------------------------------------------------------- 
        def fromrgb(x, res): 
            with tf.variable_scope('Gen_Enc_FromRGB_lod%d' % (self.resolution_log2 - res)):
                return leaky_relu(apply_bias(conv2d(x, 
                    fmaps=nf(res-1), kernel=1, cf = self.channel_first), cf = self.channel_first))

        # Building Blocks Encoder: Input --> Convolution (+Bias) --> Batchnormalisation
        # --> Activation function --> Downsample by factor 2
        # -------------------------------------------------------------------------- 
        def block_e(x,res): 
            with tf.variable_scope('Gen_Enc%dx%d' % (2**res, 2**res)):
                with tf.variable_scope('Conv'):
                    x = leaky_relu(batchnorm(apply_bias(conv2d(x, fmaps = nf(res-2), kernel = 3, 
                        cf = self.channel_first), cf = self.channel_first), cf = self.channel_first))
                x = downscale2d(x, cf = self.channel_first)
                return x

        # =========================================================================
        # Encoder
        # =========================================================================

        if self.structure == 'linear':
            skip = []
            img = labels_in
            x = fromrgb(img, self.resolution_log2)
            #print(x.shape)
            for res in range(self.resolution_log2, 4, -1):
                lod = self.resolution_log2 - res
                x = block_e(x, res)
                #print(x.shape)
                img = downscale2d(img, cf = self.channel_first)
                y = fromrgb(img, res - 1)
                with tf.variable_scope('Gen_Enc_Grow_lod%d' % lod):
                    x = lerp_clip(x, y, self.lod_in - lod)
                skip.append(x) #
                #print('Encoder after', 2**res, "conv", x.shape)

            for res in range(4,0,-1):
                lod = self.resolution_log2 - res
                x = block_e(x, res)
                #print(x.shape)
                if res >= 2: skip.append(x)
            #print("Encoder Output", x.shape)


            combo_out = x 


        # ========================================================================
        # Decoder
        # ========================================================================

        # Last Layer to Image: A x A x Channels(A) ==> A x A x 3 
        # -------------------------------------------------------------------------- 
        def torgb(x, res): # res = 2..resolution_log2
            lod = self.resolution_log2 - res
            with tf.variable_scope('Gen_Dec_ToRGB_lod%d' % lod):
                return apply_bias(conv2d(x, fmaps=self.num_channels, kernel=1, cf = self.channel_first), cf = self.channel_first)

        # Building Blocks Encoder: Input --> Upsampling by factor 2 --> Convolution (+Bias) 
        # --> Batchnormalisation --> Activation 
        # -------------------------------------------------------------------------- 
        def block_d(x,res): 
            layers = []
            with tf.variable_scope('Gen_Dec_%dx%d' % (2**res, 2**res)):
                x = upscale2d(x, cf = self.channel_first)
                with tf.variable_scope('Conv'):
                    x = leaky_relu(batchnorm(apply_bias(conv2d(x, fmaps = nf(res-1), kernel = 3, cf = self.channel_first),
                     cf = self.channel_first), cf = self.channel_first))
                return x

        # Growing the Decoder 
        # ---------------------------------------------------------------------------  
        if self.structure == 'linear':
            #print('Decode Input:', x.shape)
            x = combo_out
            #print('start decoder',x.shape)
            x = block_d(x, 1)                                   # 1x1x512  ==> 2x2x512
            #print(x.shape)
            x = tf.concat([x,skip[-1]], axis=self.conc_axis)    # concat   ==> 2x2x1024
            x = block_d(x, 2)                                   # 2x2x1024 ==> 4x4x512
            #print(x.shape)
            x = tf.concat([x,skip[-2]], axis=self.conc_axis)
            x = block_d(x, 3)                                   # 4x4x1024 ==> 8x8x512
            #print(x.shape)
            x = tf.concat([x,skip[-3]], axis=self.conc_axis)  
            x = block_d(x, 4)                                   # 8x8x1024 ==> 16x16x512
            #print(x.shape)
            #print("-----")
            images_out = torgb(x,4)                             # Extracted Output layer 16x16
            #print("Image Out:", images_out.shape)
            x = tf.concat([x,skip[-4]], axis=self.conc_axis)    # concat   ==> 16x16x1024
            #print('Decode after const:', x.shape)
            for res in range(5,self.resolution_log2+1):  
                lod = self.resolution_log2 - res 
                x   = block_d(x,res)
                #print(x.shape)
                img = torgb(x,res)
                if res  < self.resolution_log2: 
                    x  = tf.concat([x,skip[-res]], axis = self.conc_axis)   
                images_out = upscale2d(images_out, cf = self.channel_first)
                with tf.variable_scope('Gen_Dec_Grow_lod%d' % lod):
                	images_out = lerp_clip(img, images_out, self.lod_in - lod)
                #print("Images Out:", images_out.shape)
                #print('Decode res:', 2**res,'shape x:', x.shape, "shape img out:", images_out.shape)
        #print('output',images_out.shape)
        assert images_out.dtype == tf.as_dtype(self.dtype)
        images_out = tf.identity(images_out, name='images_out')
        return images_out


    def discriminator(self, unknown, input_image):
        

        assert self.resolution == 2**self.resolution_log2 and self.resolution >= 4    
        def nf(stage): return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)


        def fromrgb(x, res): 
            with tf.variable_scope('Disc_FromRGB_lod%d' % (self.resolution_log2 - res)):
                return leaky_relu(apply_bias(conv2d(x, 
                    fmaps=nf(res-1), kernel=1, cf = self.channel_first), cf = self.channel_first))

        def block(x,res): 
            layers = []
            with tf.variable_scope('Disc_%dx%d' % (2**res, 2**res)):
                if res > 4:
                    with tf.variable_scope('Conv'):
                        x = leaky_relu(batchnorm(apply_bias(conv2d(x, fmaps = nf(res-2), kernel = 3, 
                                      cf = self.channel_first), cf = self.channel_first), cf = self.channel_first))
                    x = downscale2d(x, cf = self.channel_first)  
                else:
                    with tf.variable_scope('Patch'):
                        x = tf.sigmoid(apply_bias(conv2d(x, fmaps = 1, kernel = 3, 
                                      cf = self.channel_first), cf = self.channel_first))
                        
                return x
                
        if self.structure == 'linear':
            img = tf.concat([input_image,unknown], axis=self.conc_axis)
            x   = fromrgb(img, self.resolution_log2)
            for res in range(self.resolution_log2, 4, -1):
                #print(res, x.shape, nf(res-2))
                lod = self.resolution_log2 - res
                x   = block(x, res)
                img = downscale2d(img, cf = self.channel_first)
                y   = fromrgb(img, res - 1)
                with tf.variable_scope('Disc_Grow_lod%d' % lod):
                    x = lerp_clip(x, y, self.lod_in - lod)


            x = block(x, 4)

            return x 

#x1 = tf.random_normal([1,256,256,3], mean=0, stddev=1) 
#x2 = tf.random_normal([1,256,256,3], mean=0, stddev=1) 

#outputs_test = cGAN().generator(x1)
# Examples to understand Clipping
    # -----------------------------------------------------------
    # Example 1: Final lod_reso = 8 (256 pixel). We're in 1. stage (4x4). 
    # res = 8, lod = 0 so we're in the 256x256x64 stage:
    # 100k Iters in TRAINING Phase: lod_in = lod_tr -  max(100-600,0)/600 = lod_tr = 6. 
    # ===> lerp_clip(x,y,lod_in - lod) = lerp_clip(x,y,6) = x + (y-x)*1 = y 
    # ===> output is only downscaled image, as layer doesn't exist!
    #
    # Example 2: Final lod_reso = 8 (256 pixel). We're in transition (4x4) to (8x8).
    # res = 3, lod = 5 so we're in the 8x8x512 stage:
    # 100k Iters in TRANISITION Phase: lod_in = 6 - (700-600)/600 = 5.83.
    # ===> lerp_clip(x,y,lod_in - lod) = lerp_clip(x,y,0.83) = x + (y-x)*0.83 = 0.17x + 0.83y 
    # ===> output INTERPOLATED: composed of 83% image and 17% output of layer 8x8!
    # -----------------------------------------------------------
