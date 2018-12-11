# coding=utf-8
from keras import backend as K
from keras.layers import *
from keras.activations import relu
from keras.initializers import RandomNormal
from demo.instance_normalization import InstanceNormalization
from demo.GroupNormalization import GroupNormalization
from keras.models import Model
from tensorflow.contrib.distributions import Beta
from keras.optimizers import Adam

import time
import tensorflow as tf
import Augmentor
import numpy

mnist_train = np.load('../datasets/MNIST/mnist_train.npy')
mnist_test = np.load('../datasets/MNIST/mnist_test.npy')

usps_train = np.load('../datasets/USPS/usps_train.npy')
usps_test = np.load('../datasets/USPS/usps_test.npy')


K.set_learning_phase(1)

channel_axis = -1
channel_first = False

IMAGE_SHAPE = (28, 28, 1)
nc_in = 1
nc_D_inp = 1
n_dim_style = 8
n_resblocks = 3
n_adain = 2 * n_resblocks
n_dim_adain = 256
nc_base = 64
n_downscale_content = 2
n_downscale_style = 4
use_groupnorm = False
w_l2 = 1e-4

# optimization configs
use_perceptual_loss = True
use_lsgan = True
use_mixup = True
mixup_alpha = 0.2
batchSize = 1
conv_init_dis = RandomNormal(0, 0.02)
conv_init = 'he_normal'
lrD = 1e-4
lrG = 1e-4
opt_decay = 0
TOTAL_ITERS = 300000

# loss weights for generators
w_D = 1
w_recon = 10
w_recon_latent = 1
w_cycrecon = 0.3


def conv_block(input_tensor, f, k=3, strides=2, use_norm=False):
    x = input_tensor
    x = ReflectPadding2D(x)
    x = Conv2D(f, kernel_size=k, strides=strides, kernel_initializer=conv_init, kernel_regularizer=regularizers.l2(w_l2), use_bias=(not use_norm), padding='valid')(x)
    if use_norm:
        x = InstanceNormalization(epsilon=1e-5)(x)
    x = Activation("relu")(x)
    return x

def conv_block_d(input_tensor, f, use_norm=False):
    x = input_tensor
    x = ReflectPadding2D(x, 2)
    x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init_dis,
               kernel_regularizer=regularizers.l2(w_l2),
               use_bias=(not use_norm), padding="valid")(x)
    if use_norm:
        x = InstanceNormalization(epsilon=1e-5)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def ReflectPadding2D(x, pad=1):
    x = Lambda(lambda x: tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT'))(x)
    return x


def upscale_nn(inp, f, use_norm=False):
    x = inp
    x = UpSampling2D()(x)
    x = ReflectPadding2D(x, 2)
    x = Conv2D(f, kernel_size=5, kernel_initializer=conv_init,
               kernel_regularizer=regularizers.l2(w_l2),
               use_bias=(not use_norm), padding='valid')(x)
    if use_norm:
        if use_groupnorm:
            x = GroupNormalization(group=32)(x)
        else:
            x = GroupNormalization(group=f)(x) # group=f equivalant to layer norm
    x = Activation('relu')(x)
    return x


def Encoder_style_MUNIT(nc_in=1, input_size=IMAGE_SHAPE[0], n_dim_adain=n_dim_adain,
                        n_dim_style=n_dim_style, nc_base=nc_base, n_downscale_style=n_downscale_style):
    inp = Input(shape=(input_size, input_size, nc_in))
    x = ReflectPadding2D(inp, 3)
    x = Conv2D(64, kernel_size=7, kernel_initializer=conv_init, kernel_regularizer=regularizers.l2(w_l2), use_bias=True, padding='valid')(x)
    x = Activation('relu')(x)

    dim = 1
    for i in range(n_downscale_style):
        dim = 4 if dim >= 4 else dim * 2
        x = conv_block(x, dim * nc_base)
    x = GlobalAveragePooling2D()(x)
    style_code = Dense(n_dim_style, kernel_regularizer=regularizers.l2(w_l2))(x)
    return Model(inp, style_code)


def Encoder_content_MUNIT(nc_in=1, input_size=IMAGE_SHAPE[0], n_downscale_content=n_downscale_content, nc_base=nc_base):
    def res_block_content(input_tensor, f):
        x = input_tensor
        x = ReflectPadding2D(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, kernel_regularizer=regularizers.l2(w_l2),
                   use_bias=False, padding='valid')(x)
        x = InstanceNormalization(epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = ReflectPadding2D(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, kernel_regularizer=regularizers.l2(w_l2),
                   use_bias=False, padding='valid')(x)
        x = InstanceNormalization(epsilon=1e-5)(x)
        x = add([x, input_tensor])
        return x

    inp = Input(shape=(input_size, input_size, nc_in))
    x = ReflectPadding2D(inp, 3)
    x = Conv2D(64, kernel_size=7, kernel_initializer=conv_init, kernel_regularizer=regularizers.l2(w_l2), use_bias=True, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    dim = 1
    ds = 2 ** n_downscale_content
    for i in range(n_downscale_content):
        dim = 4 if dim >= 4 else dim * 2
        x = conv_block(x, dim * nc_base, use_norm=True)
    for i in range(n_resblocks):
        x = res_block_content(x, dim * nc_base)
    content_code = x
    return Model(inp, content_code)


def MLP_MUNIT(n_dim_style=n_dim_style, n_dim_adain=n_dim_adain, n_blk=3, n_adain=2 * n_resblocks):
    # MLP for AdaIN parameters
    inp_style_code = Input(shape=(n_dim_style,))

    adain_params = Dense(n_dim_adain, kernel_regularizer=regularizers.l2(w_l2), activation='relu')(inp_style_code)
    for i in range(n_blk - 2):
        adain_params = Dense(n_dim_adain, kernel_regularizer=regularizers.l2(w_l2), activation='relu')(adain_params)
    adain_params = Dense(2 * n_adain * n_dim_adain, kernel_regularizer=regularizers.l2(w_l2))(
        adain_params)  # No output activation
    return Model(inp_style_code, [adain_params])



def Decoder_MUNIT(nc_in=256, input_size=IMAGE_SHAPE[0]//(2 ** n_downscale_content), n_dim_adain=256, n_resblocks=n_resblocks):
    def op_adain(inp):
        x = inp[0]
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        adain_bias = inp[1]
        adain_bias = K.reshape(adain_bias, (-1, 1, 1, n_dim_adain))
        adain_weight = inp[2]
        adain_weight = K.reshape(adain_weight, (-1, 1, 1, n_dim_adain))
        out = tf.nn.batch_normalization(x, mean, var, adain_bias, adain_weight, variance_epsilon=1e-7)
        return out

    def AdaptiveInstanceNorm2d(inp, adain_params, idx_adain):
        assert inp.shape[-1] == n_dim_adain
        x = inp
        idx_head = idx_adain * 2 * n_dim_adain
        adain_weight = Lambda(lambda x: x[:, idx_head:idx_head + n_dim_adain])(adain_params)
        adain_bias = Lambda(lambda x: x[:, idx_head+n_dim_adain:idx_head + 2 * n_dim_adain])(adain_params)
        out = Lambda(op_adain)([x, adain_bias, adain_weight])
        return out

    def res_block_adain(inp, f, adain_params, idx_adain):
        x = inp
        x = ReflectPadding2D(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init,
                   kernel_regularizer=regularizers.l2(w_l2), bias_regularizer=regularizers.l2(w_l2),
                   use_bias=False, padding="valid")(x)
        x = Lambda(lambda x: AdaptiveInstanceNorm2d(x[0], x[1], idx_adain))([x, adain_params])
        x = Activation('relu')(x)
        x = ReflectPadding2D(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init,
                   kernel_regularizer=regularizers.l2(w_l2), bias_regularizer=regularizers.l2(w_l2),
                   use_bias=False, padding="valid")(x)
        x = Lambda(lambda x: AdaptiveInstanceNorm2d(x[0], x[1], idx_adain + 1))([x, adain_params])

        x = add([x, inp])
        return x

    inp_style = Input((n_dim_style, ))
    style_code = inp_style
    mlp = MLP_MUNIT()
    adain_params = mlp(style_code)

    inp_content = Input(shape=(input_size, input_size, nc_in))
    content_code = inp_content
    x = content_code

    for i in range(n_resblocks):
        x = res_block_adain(x, nc_in, adain_params, 2 * i)

    dim = 1
    for i in range(n_downscale_content):
        dim = dim if nc_in // dim <= 64 else dim * 2
        x = upscale_nn(x, nc_in//dim, use_norm=True)
    x = ReflectPadding2D(x, 3)
    out = Conv2D(1, kernel_size=7, kernel_initializer=conv_init,
                 kernel_regularizer=regularizers.l2(w_l2),
                 padding='valid', activation="tanh")(x)
    return Model([inp_style, inp_content], [out, style_code, content_code])


def Discriminator(nc_in, input_size=IMAGE_SHAPE[0]):
    inp = Input(shape=(input_size, input_size, nc_in))
    x = conv_block_d(inp, 64, False)
    x = conv_block_d(x, 128, False)
    x = conv_block_d(x, 256, False)
    x = ReflectPadding2D(x, 2)
    out = Conv2D(1, kernel_size=5, kernel_initializer=conv_init_dis,
                 kernel_regularizer=regularizers.l2(w_l2),
                 use_bias=False, padding="valid")(x)
    if not use_lsgan:
        x = Activation('sigmoid')(x)
    return Model(inputs=[inp], outputs=out)


def Discriminator_MS(nc_in, input_size=IMAGE_SHAPE[0]):
    # Multi-scale discriminator architecture
    inp = Input(shape=(input_size, input_size, nc_in))

    def conv2d_blocks(inp, nc_base=64, n_layers=3):
        x = inp
        dim = nc_base
        for _ in range(n_layers):
            x = conv_block_d(x, dim, False)
            dim *= 2
        x = Conv2D(1, kernel_size=1, kernel_initializer=conv_init_dis,
                   kernel_regularizer=regularizers.l2(w_l2),
                   use_bias=True, padding="valid")(x)
        if not use_lsgan:
            x = Activation('sigmoid')(x)
        return x

    x0 = conv2d_blocks(inp)
    ds1 = AveragePooling2D(pool_size=(3, 3), strides=2)(inp)
    x1 = conv2d_blocks(ds1)
    ds2 = AveragePooling2D(pool_size=(3, 3), strides=2)(ds1)
    x2 = conv2d_blocks(ds2)
    return Model(inputs=[inp], outputs=[x0, x1, x2])


encoder_style_A = Encoder_style_MUNIT()
encoder_content_A = Encoder_content_MUNIT()
encoder_style_B = Encoder_style_MUNIT()
encoder_content_B = Encoder_content_MUNIT()
decoder_A = Decoder_MUNIT()
decoder_B = Decoder_MUNIT()

x = Input(shape=IMAGE_SHAPE) # dummy input tensor
netGA = Model(x, decoder_A([encoder_style_A(x), encoder_content_A(x)]))
netGB = Model(x, decoder_B([encoder_style_B(x), encoder_content_B(x)]))

netDA = Discriminator_MS(nc_D_inp)
netDB = Discriminator_MS(nc_D_inp)

def model_paths(netEnc_content, netEnc_style, netDec):
    fn_content_code = K.function([netEnc_content.inputs[0]], [netEnc_content.outputs[0]])
    fn_style_code = K.function([netEnc_style.inputs[0]], [netEnc_style.outputs[0]])
    fn_decoder_rgb = K.function(netDec.inputs, [netDec.outputs[0]])

    fake_output = netDec.outputs[0]
    fn_decoder_out = K.function(netDec.inputs, [fake_output])
    return fn_content_code, fn_style_code, fn_decoder_out


def translation(src_imgs, style_image, fn_content_code_src, fn_style_code_tar, fn_decoder_rgb_tar, rand_style=False):
    # Cross domain translation function
    # This funciton is for visualization purpose
    """
    Inputs:
        src_img: source domain images, shape=(input_batch_size, h, w, c)
        style_image: target style images,  shape=(input_batch_size, h, w, c)
        fn_content_code_src: Source domain K.function of content encoder
        fn_style_code_tar: Target domain K.function of style encoder
        fn_decoder_rgb_tar: Target domain K.function of decoder
        rand_style: sample style codes from normal distribution if set True.
    Outputs:
        fake_rgb: output tensor of decoder having chennels [R, G, B], shape=(input_batch_size, h, w, c)
    """
    batch_size = src_imgs.shape[0]
    content_code = fn_content_code_src([src_imgs])[0]
    if rand_style:
        style_code = np.random.normal(size=(batch_size, n_dim_style))
    elif style_image is None:
        style_code = fn_style_code_tar([src_imgs])[0]
    else:
        style_code = fn_style_code_tar([style_image])[0]
    fake_rgb = fn_decoder_rgb_tar([style_code, content_code])[0]
    return fake_rgb

real_A = Input(shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
real_B = Input(shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))

path_content_code_A, path_style_code_A, path_decoder_A = model_paths(encoder_content_A, encoder_style_A, decoder_A)
path_content_code_B, path_style_code_B, path_decoder_B = model_paths(encoder_content_B, encoder_style_B, decoder_B)

if use_lsgan:
    loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
else:
    loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

# random sampled style code
s_a = K.random_normal(shape=(batchSize, n_dim_style), mean=0., stddev=1.)
s_b = K.random_normal(shape=(batchSize, n_dim_style), mean=0., stddev=1.)
# encode
c_a, s_a_prime = encoder_content_A(real_A), encoder_style_A(real_A)
c_b, s_b_prime = encoder_content_B(real_B), encoder_style_B(real_B)
# decode (within domain)
x_a_recon = decoder_A([s_a_prime, c_a])[0]
x_b_recon = decoder_B([s_b_prime, c_b])[0]
# decode (cross domain)
x_ba = decoder_A([s_a, c_b])[0]
x_ab = decoder_B([s_b, c_a])[0]
# encode again
c_b_recon, s_a_recon = encoder_content_A(x_ba), encoder_style_A(x_ba)
c_a_recon, s_b_recon = encoder_content_B(x_ab), encoder_style_B(x_ab)
# decode again (if needed)
x_aba = decoder_A([s_a_prime, c_a_recon])[0]
x_bab = decoder_B([s_b_prime, c_b_recon])[0]

loss_GA = loss_GB = loss_DA = loss_DB = loss_adv_GA = loss_adv_GB = 0

# reconstruction loss
loss_x_a_recon = w_recon * K.mean(K.abs(x_a_recon - real_A))
loss_x_b_recon = w_recon * K.mean(K.abs(x_b_recon - real_B))
loss_s_a_recon = w_recon_latent * K.mean(K.abs(s_a_recon - s_a))
loss_s_b_recon = w_recon_latent * K.mean(K.abs(s_b_recon - s_b))
loss_c_a_recon = w_recon_latent * K.mean(K.abs(c_a_recon - c_a))
loss_c_b_recon = w_recon_latent * K.mean(K.abs(c_b_recon - c_b))
loss_cycrecon_x_a = w_cycrecon * K.mean(K.abs(x_aba - real_A))
loss_cycrecon_x_b = w_cycrecon * K.mean(K.abs(x_bab - real_B))
loss_GA += (loss_x_a_recon + loss_s_a_recon + loss_c_a_recon)
loss_GB += (loss_x_b_recon + loss_s_b_recon + loss_c_b_recon)

# GAN loss
if use_mixup:
    dist_beta = Beta(mixup_alpha, mixup_alpha)
    lam_A = dist_beta.sample()
    mixup_A = lam_A * real_A + (1 - lam_A) * x_ba
    outputs_xba_DA = netDA(x_ba)
    outputs_mixup_DA = netDA(mixup_A)
    for output in outputs_mixup_DA:
        loss_DA += loss_fn(output, lam_A * K.ones_like(output))
    for output in outputs_xba_DA:
        loss_adv_GA += w_D * loss_fn(output, K.ones_like(output))

    lam_B = dist_beta.sample()
    mixup_B = lam_B * real_B + (1 - lam_B) * x_ab
    outputs_xab_DB = netDB(x_ab)
    outputs_mixup_DB = netDB(mixup_B)
    for output in outputs_mixup_DB:
        loss_DB += loss_fn(output, lam_B * K.ones_like(output))
    for output in outputs_xab_DB:
        loss_adv_GB += w_D * loss_fn(output, K.ones_like(output))
else:
    outputs_DA_real = netDA(real_A)
    outputs_DA_fake = netDA(x_ba)
    for output_real, output_fake in zip(outputs_DA_real, outputs_DA_fake):
        loss_DA += loss_fn(output_real, K.ones_like(output_real))
        loss_DA += loss_fn(output_fake, K.zeros_like(output_fake))
        loss_adv_GA += w_D * loss_fn(output_fake, K.ones_like(output_fake))
    outputs_DB_real = netDB(real_B)
    outputs_DB_fake = netDB(x_ab)
    for output_real, output_fake in zip(outputs_DB_real, outputs_DB_fake):
        loss_DB += loss_fn(output_real, K.ones_like(output_real))
        loss_DB += loss_fn(output_fake, K.zeros_like(output_fake))
        loss_adv_GB += w_D * loss_fn(output_fake, K.ones_like(output_fake))
loss_GA += loss_adv_GA
loss_GB += loss_adv_GB

# L2 weight regularization
# https://github.com/keras-team/keras/issues/2662
for loss_tensor in netGA.losses:
    loss_GA += loss_tensor
for loss_tensor in netGB.losses:
    loss_GB += loss_tensor
for loss_tensor in netDA.losses:
    loss_DA += loss_tensor
for loss_tensor in netDB.losses:
    loss_DB += loss_tensor


# Trainable weights
weightsDA = netDA.trainable_weights
weightsGA = netGA.trainable_weights
weightsDB = netDB.trainable_weights
weightsGB = netGB.trainable_weights

# Define training function
lr_factor = 1

training_updates = Adam(lr=lrD*lr_factor, beta_1=0.5).get_updates(weightsDA+weightsDB,[],loss_DA+loss_DB)
netD_train = K.function([real_A, real_B],
                        [loss_DA, loss_DB],
                        training_updates)

training_updates = Adam(lr=lrG*lr_factor, beta_1=0.5, decay=opt_decay).get_updates(weightsGA+weightsGB,[], loss_GA+loss_GB)
netG_train = K.function([real_A, real_B],
                        [loss_GA, loss_GB,
                         loss_x_a_recon, loss_x_b_recon,
                         loss_s_a_recon, loss_s_b_recon,
                         loss_c_a_recon, loss_c_b_recon,
                         loss_adv_GA, loss_adv_GB],
                        training_updates)


def get_unit_test_weights():
    # Get the weight values of the first Conv2D layer of encoders, decoders, and discriminators
    global encoder_style_A
    global encoder_content_A
    global encoder_style_B
    global encoder_content_B
    global decoder_A
    global decoder_B
    global netDA
    global netDB

    w_enc_content_A = encoder_content_A.layers[2].get_weights()
    w_enc_style_A = encoder_style_A.layers[2].get_weights()
    w_dec_A = decoder_A.layers[3].get_weights()
    w_DA = netDA.layers[6].get_weights()
    w_enc_content_B = encoder_content_B.layers[2].get_weights()
    w_enc_style_B = encoder_style_B.layers[2].get_weights()
    w_dec_B = decoder_B.layers[3].get_weights()
    w_DB = netDB.layers[6].get_weights()

    return [w_enc_content_A,
            w_enc_style_A,
            w_dec_A,
            w_DA,
            w_enc_content_B,
            w_enc_style_B,
            w_dec_B,
            w_DB,
            ]


def unit_test(prev_weights):
    # Test if weights are updated after an iter of training
    global encoder_style_A
    global encoder_content_A
    global encoder_style_B
    global encoder_content_B
    global decoder_A
    global decoder_B
    global netDA
    global netDB

    w_enc_content_A = encoder_content_A.layers[2].get_weights()
    w_enc_style_A = encoder_style_A.layers[2].get_weights()
    w_dec_A = decoder_A.layers[3].get_weights()
    w_DA = netDA.layers[6].get_weights()
    w_enc_content_B = encoder_content_B.layers[2].get_weights()
    w_enc_style_B = encoder_style_B.layers[2].get_weights()
    w_dec_B = decoder_B.layers[3].get_weights()
    w_DB = netDB.layers[6].get_weights()

    name_nets = ['encoder_content_A',
                 'encoder_style_A',
                 'decoder_A',
                 'netDA',
                 'encoder_content_B',
                 'encoder_style_B',
                 'decoder_B',
                 'netDB'
                 ]

    weights = [w_enc_content_A,
               w_enc_style_A,
               w_dec_A,
               w_DA,
               w_enc_content_B,
               w_enc_style_B,
               w_dec_B,
               w_DB,
               ]

    print("[Unit test]")
    for prev_w, w, net in zip(prev_weights, weights, name_nets):
        print(f"Testing {net}")
        assert (not np.array_equal(prev_w[0], w[
            0])), f"Unit test failed! The weights of {net} are exactly the same after a batch of training."

    return None


def showG(test_A, test_B):
    sample_imgs_pBtA = []
    sample_imgs_pAtB = []
    imgs_pAtA = np.squeeze(np.array([
        translation(test_A[i:i + 1], None, path_content_code_A, path_style_code_A, path_decoder_A)[0]
        for i in range(test_A.shape[0])]))
    imgs_pBtA = np.squeeze(np.array([
        translation(test_A[i:i + 1], test_B[i:i + 1], path_content_code_A, path_style_code_B, path_decoder_B)[0]
        for i in range(test_A.shape[0])]))
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_A[i:i + 1], test_B[i:i + 1], path_content_code_A, path_style_code_B, path_decoder_B,
                        rand_style=True)[0]
            for i in range(test_A.shape[0])]))
        sample_imgs_pBtA.append(im)

    imgs_pBtB = np.squeeze(np.array([
        translation(test_B[i:i + 1], None, path_content_code_B, path_style_code_B, path_decoder_B)[0]
        for i in range(test_B.shape[0])]))
    imgs_pAtB = np.squeeze(np.array([
        translation(test_B[i:i + 1], test_A[i:i + 1], path_content_code_B, path_style_code_A, path_decoder_A)[0]
        for i in range(test_B.shape[0])]))
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_B[i:i + 1], test_A[i:i + 1], path_content_code_B, path_style_code_A, path_decoder_A,
                        rand_style=True)[0]
            for i in range(test_B.shape[0])]))
        sample_imgs_pAtB.append(im)

    figure_A = np.concatenate([
        np.squeeze(test_A),
        imgs_pAtA,
        imgs_pBtA,
        sample_imgs_pBtA[0],
        sample_imgs_pBtA[1],
        sample_imgs_pBtA[2],
        sample_imgs_pBtA[3],
        sample_imgs_pBtA[4]
    ], axis=-2)
    figure_A = figure_A.reshape(batchSize * IMAGE_SHAPE[0], 8 * IMAGE_SHAPE[1], 3)
    figure_A = np.clip((figure_A + 1) * 255 / 2, 0, 255).astype('uint8')
    figure_B = np.concatenate([
        np.squeeze(test_B),
        imgs_pBtB,
        imgs_pAtB,
        sample_imgs_pAtB[0],
        sample_imgs_pAtB[1],
        sample_imgs_pAtB[2],
        sample_imgs_pAtB[3],
        sample_imgs_pAtB[4]
    ], axis=-2)
    figure_B = figure_B.reshape(batchSize * IMAGE_SHAPE[0], 8 * IMAGE_SHAPE[1], 3)
    figure_B = np.clip((figure_B + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = np.concatenate([figure_A, figure_B], axis=1)
    #display(Image.fromarray(figure))


t0 = time.time()
gen_iterations = 0
errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
errGA_x_recon_sum = errGA_s_recon_sum = errGA_c_recon_sum = 0
errGB_x_recon_sum = errGB_s_recon_sum = errGB_c_recon_sum = 0
errGA_adv_sum = errGB_adv_sum = 0

display_iters = 300


def generate_next(data):
    indexes = np.random.choice(range(data.shape[0]), size=1)
    data = data[indexes]
    data = data.reshape(-1, 28, 28, 1)
    return data

while gen_iterations < TOTAL_ITERS:
    #imgs_A, _ = next(gA)
    # imgs_A = imgs_A * 2 - 1  # transform [0, 1] to [-1, 1]
    # imgs_B, _ = next(gB)
    # imgs_B = imgs_B * 2 - 1
    imgs_A = generate_next(mnist_train)
    imgs_B = generate_next(usps_train)

    if gen_iterations == 2:
        unit_test_weights = get_unit_test_weights()

    # Train dicriminators for one batch
    errDs = netD_train([imgs_A, imgs_B])
    errDA_sum += errDs[0]
    errDB_sum += errDs[1]

    # Train generators for one batch
    errGs = netG_train([imgs_A, imgs_B])
    errGA_sum += errGs[0]
    errGB_sum += errGs[1]
    errGA_x_recon_sum += errGs[2]
    errGB_x_recon_sum += errGs[3]
    errGA_s_recon_sum += errGs[4]
    errGB_s_recon_sum += errGs[5]
    errGA_c_recon_sum += errGs[6]
    errGB_c_recon_sum += errGs[7]
    errGA_adv_sum += errGs[8]
    errGB_adv_sum += errGs[9]

    if gen_iterations == 2:
        unit_test(unit_test_weights)
        print("Unit test passed.")
        unit_test_weights = None

    gen_iterations += 1

    # Visualization
    if gen_iterations % display_iters == 0:
        #clear_output()
        print('[Iter. %d] Loss_DA: %f | Loss_DB: %f | Loss_GA: %f | Loss_GB: %f'
              % (gen_iterations, errDA_sum / display_iters, errDB_sum / display_iters,
                 errGA_sum / display_iters, errGB_sum / display_iters))
        print('[Reconstruction losses (image)] x_a_recon: %f | x_b_recon: %f'
              % (errGA_x_recon_sum / display_iters, errGB_x_recon_sum / display_iters))
        print('[Reconstruction losses (style code)] s_a_recon: %f | s_b_recon: %f'
              % (errGA_s_recon_sum / display_iters, errGB_s_recon_sum / display_iters))
        print('[Reconstruction losses (content code)] c_a_recon: %f | c_b_recon: %f'
              % (errGA_c_recon_sum / display_iters, errGB_c_recon_sum / display_iters))
        print('[Adversarial losses] GA_adv: %f | GB_adv: %f'
              % (errGA_adv_sum / display_iters, errGB_adv_sum / display_iters))
        print('[Elapsed time] %f sec.' % (time.time() - t0))

        # for i in range(8 // batchSize):
        #     vis_A, _ = gA.send(batchSize)
        #     vis_A = vis_A * 2 - 1
        #     vis_B, _ = gB.send(batchSize)
        #     vis_B = vis_B * 2 - 1
        #     showG(vis_A, vis_B)
        errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
        errGA_x_recon_sum = errGA_s_recon_sum = errGA_c_recon_sum = 0
        errGB_x_recon_sum = errGB_s_recon_sum = errGB_c_recon_sum = 0
        errGA_adv_sum = errGB_adv_sum = 0

        # # Save models
        # encoder_style_A.save_weights("models/encoder_style_A.h5")
        # encoder_content_A.save_weights("models/encoder_content_A.h5")
        # encoder_style_B.save_weights("models/encoder_style_B.h5")
        # encoder_content_B.save_weights("models/encoder_content_B.h5")
        # decoder_A.save_weights("models/decoder_A.h5")
        # decoder_B.save_weights("models/decoder_B.h5")
        # # netGA.save_weights("models/netGA.h5")
        # # netGB.save_weights("models/netGB.h5")
        # netDA.save_weights("models/netDA.h5")
        # netDB.save_weights("models/netDB.h5")



