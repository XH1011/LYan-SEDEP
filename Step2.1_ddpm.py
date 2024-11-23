# pip install tensorflow_addons
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import h5py
import numpy as np
import math
import os
import random
import scipy.io as scio
from sklearn.preprocessing import normalize
os.environ["TF_KERAS"] = '1'

# from __future__ import division

import numpy as np
from scipy import linalg

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array

def set_seed(seed=31):

  os.environ['PYTHONHASHSEED']=str(0)
  random.seed(seed)
  tf.random.set_seed(seed)
  # tf.keras.utils.set_random_seed(seed)
  tf.compat.v1.set_random_seed(seed)
  np.random.seed(seed)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'

# set the parameters for dataset
batch_size = 32
num_epochs = 1000
timesteps = 1000
norm_groups = 8
learning_rate=2e-4

img_size=2048
img_channels=1
clip_min=-1.0
clip_max=1.0

first_conv_channels=8
channel_multipier=[1,2,4,8]
widths=[first_conv_channels* mult for mult in channel_multipier]
has_attention=[False,False,False,False]

num_res_blocks = 2


# data preprocessing
# Tensorflow preprocessing operations.
def preprocess(x,y):
    return tf.image.resize(tf.cast(x, tf.float32) / 127.5 - 1, (32, 32))

def load_data_train(file,BATCH_SIZE):
    f = open(file, 'rb')
    x0=pickle.load(f)[0].astype(np.float32)
    x = tf.data.Dataset.from_tensor_slices(x0)
    x = x.shuffle(5000).batch(BATCH_SIZE)
    return x, x0
def load_data_test(file,BATCH_SIZE):
    f = open(file, 'rb')
    x0=pickle.load(f)[1].astype(np.float32)
    x = tf.data.Dataset.from_tensor_slices(x0)
    x = x.shuffle(5000).batch(BATCH_SIZE)
    return x,x0

# Set random seeds for noise
def set_key(key):
    np.random.seed(key)

#load data
# file_name='D:/Desktop/paper3/dataset/chopper/QuantileTransformer/x1.pkl'
# file_name='D:/Desktop/paper3/dataset/chopper/x1.pkl'
# file_name='D:/Desktop/paper3/dataset/dataset_pu/QuantileTransformer/x1.pkl'
file_name='./LYan-SEDEP/data_pu/x1.pkl'
train_ds,x0=load_data_train(file=file_name,BATCH_SIZE=32)

class GaussianDiffusion:
    """Gaussian diffusion utility.

    Args:
        beta_start: Start value of the scheduled variance
        beta_end: End value of the scheduled variance
        timesteps: Number of time steps in the forward process
    """

    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define the linear variance schedule
        self.betas = betas = np.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=np.float64,  # Using float64 for better precision
        )
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.log_one_minus_alphas_cumprod = tf.constant(
            np.log(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32
        )

        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_mean_variance(self, x_start, t):
        """Extracts the mean, and the variance at current timestep.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """Diffuse the data.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        if noise is None:
            noise = tf.random.normal(shape=x_start.shape)
        x_start_shape = tf.shape(x_start)
        return (
                self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
                + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
                * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """

        x_t_shape = tf.shape(x_t)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffusion model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        # No noise when t == 0
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1]
        )
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise

def Add_NoiseTo_image(x_eval,noise_eval,i):
    gdf = GaussianDiffusion(beta_start=1e-4, beta_end=0.02,timesteps=1000+1)
    noise_image = gdf.q_sample(x_eval, i, noise_eval)
    noise_image=(tf.reshape(noise_image, shape=[2048, ])).numpy()
    return noise_image
def show_Added_NoiseImage(x,noise):
    fig = plt.figure(figsize=(15,15))
    for index, i in enumerate([0, 30, 500, 900]): #这四个数的长度决定于变量timesteps的大小
        if index<1:
            noisy_im=(tf.reshape(x, shape=[2048, ])).numpy()#没有添加噪声
            print('List index:',index,'Without any noises')
        else:
            print('List index:',index,'The added noise is:', i)
            noisy_im=Add_NoiseTo_image(x,noise,i)#依次添加噪声
        plt.subplot(4, 1, index + 1)
        plt.xlim((0, 2048))
        plt.xlabel('Data points')
        plt.ylabel('Amplitude')
        plt.plot(noisy_im.T)
        plt.tick_params(axis='both', which='both', direction='in')
    plt.show()


#NN
# Unet Operate
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = tfa.layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj

class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, None, :]

        x = tfa.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        x = layers.Add()([x, temb])
        x = tfa.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])

        return x

    return apply

def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=(1, 2),
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply

def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=(1, 2), interpolation=interpolation)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply

def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        # print(temb.shape)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        # print(temb.shape)
        return temb

    return apply

# Build a Unet model
def build_model(
        img_size,
        img_channels,
        widths,
        has_attention,
        first_conv_channels,
        num_res_blocks=2,
        norm_groups=8,
        interpolation="nearest",
        activation_fn=keras.activations.swish,
):
    image_input = layers.Input(
        shape=(1, img_size, img_channels), name="image_input"
    )  # (None,1,3072,1)

    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")  # *(None,)

    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=(3, 3),
        strides=(1, 2),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)

    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)  # (None,8)

    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)  # (None,8)

    skips = [x]
    print('latent vector to encoder', x.shape)
    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])

            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)
    print('latent vector after encoder', x.shape)

    # MiddleBlock
    x = ResidualBlock(widths[-4], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    # x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-4], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    print('latent vector to decoder', x.shape)

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):

            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            # print('Concatenate',x.shape)
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)
    print('latent vector after decoder', x.shape)
    # End block
    x = tfa.layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.UpSampling2D(size=(1, 2), interpolation=interpolation)(x)
    x = layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))(x)
    print('output', x.shape)
    return keras.Model([image_input, time_input], x, name="unet")


class DiffusionModel(keras.Model):
    def __init__(self, img_size, img_channels, network, ema_network, timesteps, gdf_util,
                 ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
        self.img_size = img_size
        self.img_channels = img_channels

    def train_step(self, images):
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)

            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t], training=True)

            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}

    def plot_images(
            self, epoch=None, logs=None, num_rows=2, num_cols=8, figsize=(12, 5)
    ):
        """Utility to plot images using the diffusion model during training."""
        generated_samples = self.generate_images(num_images=num_rows * num_cols)
        generated_samples = (
            tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0)
                .numpy()
                .astype(np.uint8)
        )

        _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i, image in enumerate(generated_samples):
            if num_rows == 1:
                ax[i].imshow(image)
                ax[i].axis("off")
            else:
                ax[i // num_cols, i % num_cols].imshow(image)
                ax[i // num_cols, i % num_cols].axis("off")

        plt.tight_layout()
        plt.show()


# --------------->>>Training Phase<<<---------------------------
def train_loss(img_batch):
    # 2. Sample timesteps uniformly
    ema = 0.999
    t = tf.random.uniform(
        minval=0, maxval=timesteps, shape=(img_batch.shape[0],), dtype=tf.int64)
    with tf.GradientTape() as tape:
        # 3. Sample random noise to be added to the images in the batch
        noise = tf.random.normal(shape=tf.shape(img_batch), dtype=img_batch.dtype)
        # 4. Diffuse the images with noise
        images_t = gdf_util.q_sample(img_batch, t, noise)  # (32,1,1024,1)
        # 5. Pass the diffused images and time steps to the network
        pred_noise = network([images_t, t], training=True)
        # 6. Calculate the loss
        loss = mseloss(noise, pred_noise)

    # 7. Get the gradients
    gradients = tape.gradient(loss, network.trainable_weights)
    # 8. Update the weights of the network
    opt.apply_gradients(zip(gradients, network.trainable_weights))
    # 9. Updates the weight values for the network with EMA weights
    for weight, ema_weight in zip(network.weights, ema_network.weights):
        ema_weight.assign(ema * ema_weight + (1 - ema) * weight)
    # 10. Return loss values
    return loss.numpy()


def train_on_step(image_batch):
    image_batch = tf.reshape(image_batch, shape=[-1, 1, 2048, 1])
    loss = train_loss(image_batch)
    return loss




# global setting
# seed='21'
# set_seed(int(seed))
# print(int(seed))
# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

# configure the GPU
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

# for visuialization
batch=1
sample_fig = tf.constant(next(iter(train_ds))[1],tf.float32)
x_eval=tf.reshape(sample_fig,shape=[-1,1,2048,1])
# print(x_eval.shape)
noise_eval=tf.random.normal(shape=tf.shape(x_eval),dtype=sample_fig.dtype)
# print(noise_eval.shape)
# show_Added_NoiseImage(x_eval,noise_eval)

# build model
print('Built network../')
network = build_model(
    img_size=2048,
    img_channels=1,
    widths=widths,
    has_attention=has_attention,
    first_conv_channels=first_conv_channels,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)

print('Built ema network../')
ema_network = build_model(
    img_size=2048,
    img_channels=1,
    widths=widths,
    has_attention=has_attention,
    first_conv_channels=first_conv_channels,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)

ema_network.set_weights(network.get_weights()) #To keep exactly the same weitghts from random initializer
gdf_util = GaussianDiffusion(timesteps=timesteps)

# Optimizer and loss function
opt=keras.optimizers.Adam(learning_rate=learning_rate)
mseloss = keras.losses.MeanSquaredError()
print('Network Summary-->')
ema_network.summary()

# Run
loss_list=[]
for epoch in range(num_epochs):
    for images_batch in train_ds:
        loss=train_on_step(images_batch)
    loss_list.append(loss)
    print(epoch,loss)

    # if epoch%200==0:
    if epoch % 200 == 0:
      # Save the model weights
      save_dir='./LYan-SEDEP/ddpm_pu/model/'
      # save_dir = './LYan-SEDEP/ddpm_chopper/model/'
      os.makedirs(save_dir, exist_ok=True)
      network.save_weights(save_dir+'model_'+str(epoch)+'.ckpt')

# Save the model weights (last step)
save_dir='./LYan-SEDEP/ddpm_pu/model/'
# save_dir='./LYan-SEDEP/ddpm_chopper/model/'
os.makedirs(save_dir, exist_ok=True)
network.save_weights(save_dir+'model_last_'+str(epoch)+'.ckpt')

loss_curve=np.array(loss_list)
np.savetxt('./LYan-SEDEP/ddpm_pu/model/loss.txt',loss_curve)
# np.savetxt('./LYan-SEDEP/ddpm_chopper/model/loss.txt',loss_curve)


# #---GENERATING SAMPLES FORM DDPM----
# Load the model weights
# dir='./LYan-SEDEP/ddpm_pu/model/model_last_999.ckpt'
dir='./LYan-SEDEP/ddpm_chopper/model/model_last_999.ckpt'
print('Load weights from ',dir)
ema_network.load_weights(dir)
print('load weights successfully!!!')
ema_network.summary()
# encoder nework
# latent_network=tf.keras.models.Model(inputs=ema_network.input,outputs=ema_network.get_layer('add_63').output)

def noised_images(x):
    t = tf.cast(tf.fill(len(x), 999), dtype=tf.int64)
    x = tf.reshape(x, shape=(-1, 1, 2048, 1))
    noise = tf.random.normal(shape=tf.shape(x), dtype=x.dtype)
    noised_images = gdf_util.q_sample(x, t, noise)
    return noised_images

def generate_images_x(x,num_images=16):
    # 1. Randomly sample noise (starting point for reverse process)
    x = noised_images(x)
    samples=tf.reshape(x, shape=(-1, 1, 2048, 1))
    # samples = tf.random.normal(
    #     shape=(num_images, 1, img_size, img_channels), dtype=tf.float32
    # )
    # 2. Sample from the model iteratively
    for t in reversed(range(0, timesteps)):
        tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
        pred_noise = ema_network.predict([samples, tt], verbose=0, batch_size=num_images)

        samples = gdf_util.p_sample(
            pred_noise, samples, tt, clip_denoised=True
        )
    # 3. Return generated samples
    return samples


#---------------------------------------------------------
# # 功能1：生成数据
#原始数据
for i in range(1,31):
    file_name = f'x{i}'
    save_faults.append(file_name)
for save_fault in save_faults:
    data_name = './LYan-SEDEP/data_pu/' + save_fault + '.pkl'
    _, x_train=load_data_train(file=data_name,BATCH_SIZE=32)
    _, x_test = load_data_test(file=data_name, BATCH_SIZE=32)
    new_x_train = generate_images_x(x_train, num_images=679)
    new_x_test = generate_images_x(x_test, num_images=291)
    with open('./LYan-SEDEP/results_ddpm/data_pu/gen_'+save_fault+'.pkl', 'wb') as f:
    # with open('./LYan-SEDEP/results_ddpm/data_chopper/gen_' + save_fault + '.pkl', 'wb') as f:
        pickle.dump([new_x_train,new_x_test], f, pickle.HIGHEST_PROTOCOL)
    print(save_fault)

