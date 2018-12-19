from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import glob
import math
import os
import random
import time

import numpy as np
import tensorflow as tf

from network import cGAN
from train_func import preprocess, deprocess, preprocess_lab, augment, rgb_to_lab, \
    append_index, save_images
import memory_saving_gradients

tf.logging.set_verbosity(tf.logging.ERROR)

# parser is the same as flags. We can use these to vary different parameters before training

parser = argparse.ArgumentParser()
# Basic arguments
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
# variables
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
# Additional stuff that might be unnecessary
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--reset_opti", type=bool, default=True, help="reset optimizer after each phase")
parser.add_argument("--l1_weight", type=float, default=10.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

a = parser.parse_args()

# a = parser.parse_args([
#     '--input_dir', 'facades',
#     '--mode', 'train',
#     '--output_dir', '1_facades_output',
#     '--which_direction', 'BtoA',
#     '--max_epochs', '2048',
#     # '--checkpoint', '1_facades_output',
#     '--checkpoint', None,
#     '--reset_opti', 'True',
#     '--batch_size', '5', ])

EPS = 1e-10
CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model",
                               "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def load_examples():
    # Make sure Images are are in the folder either as jpg or png
    if a.input_dir is not None and os.path.isfile(a.input_dir + "/checkpoint"):
        raise Exception("input_dir does not exist")
    # glob.glob() returns all paths with the specified pattern, so all images in training folder
    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg

    # if no images are detected as jpg, try the same with png
    if len(input_paths) == 0:
        # input_paths contains all image paths, in total 400
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    # Trainig images not detected
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    # Return the name of the path or of the image if an image path is plugged in
    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    # load images
    with tf.name_scope("load_images"):
        # Not clear what path queue does, somehow takes the input paths and shuffles them if mode==train
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        # We extract the names and the values of each Image
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        # in raw_input we have the rgb channels from the images
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        # Check if the 3rd dimension of the image is 3 for the rgb channels
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        # lab_colorization is False per default
        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        # split the concatinated image into A=real images
        # B=conditional Images
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1]  # [height, width, channels]
            a_images = preprocess(raw_input[:, :width // 2, :])
            b_images = preprocess(raw_input[:, width // 2:, :])

    if a.which_direction == "AtoB":  # input=real, targets=cond
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":  # input=cond, target=real
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images],
                                                              batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_model(inputs, targets):
    cgan = cGAN()
    # cgan.lod_in = tf.assign(cgan.lod_in, )
    with tf.variable_scope("generator"):
        outputs = cgan.generator(inputs)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = cgan.discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = cgan.discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        # discrim_loss = tf.reduce_mean(-(tf.log(tf.maximum(predict_real,EPS)) + tf.log(tf.maximum(1 - predict_fake, EPS))))
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        # gen_loss_GAN = tf.reduce_mean(-tf.log(tf.maximum(predict_fake, EPS)))
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    # Define Adam Optimizer Settings as Variables to be reset after Phase Transition 
    learning_rate = tf.Variable(name="learning_rate", initial_value=a.lr, trainable=False, dtype=tf.float32)
    beta_1 = tf.Variable(name="beta_1", initial_value=a.beta1, trainable=False, dtype=tf.float32)

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(learning_rate, beta_1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(learning_rate, beta_1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


class TrainingSchedule:
    def __init__(
            self,
            cur_nimg,  # Number of current Image
            training_images=400,  # Number of available training Images
            lod_initial_resolution=16,  # Image Resolution used at the Beginning.
            lod_training_img=16000,  # Images to show before doubling the Resolution => ~(3-6)*available Images
            lod_transition_img=16000,  # Images to show when fading in new Layers.
            batch_size=5,  # Batch Size.
            final_resolution_log2=8  # log_2 of final Resolution (256)
    ):
        # Training phase.
        self.img = cur_nimg
        phase_dur = lod_training_img + lod_transition_img  # Transition + Stabelizing Phase: 2400 Images
        phase_idx = int(np.floor(self.img / phase_dur)) if phase_dur > 0 else 0  # Current overall Layer Phase Number
        phase_img = self.img - phase_idx * phase_dur  # Number of Images within current Phase

        # Level-of-detail and resolution.
        self.lod = final_resolution_log2  # log_2 of max. Network Resolution:     8 [256x256]
        self.lod -= np.floor(
            np.log2(lod_initial_resolution))  # Subtract log_2 of initial Resolution: 6 [4x4 -> 256x256]
        self.lod -= phase_idx  # Subtract the Number of passed Phases
        if lod_transition_img > 0:
            self.lod -= max(phase_img - lod_training_img,
                            0.0) / lod_transition_img  # Subtract passed Transition Phase Percentage.
        self.lod = max(self.lod, 0.0)  # Ensure minimum of zero
        self.resolution = 2 ** (final_resolution_log2 - int(np.floor(self.lod)))
        self.stored_cur_nimg = cur_nimg


def main():
    # Set Seed for reproducing results

    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    for i, j in a._get_kwargs():  # See current configuration
        print(i, "=", j)

    examples = load_examples()  # load images
    print("examples count = %d" % examples.count)
    model = create_model(examples.inputs, examples.targets)  # Feed images into Network

    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # with tf.name_scope("inputs_summary"):
    #    tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    # tf.summary.scalar("predict_real",model.predict_real)
    # tf.summary.scalar("predict_fake", model.predict_fake)
    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    lod_in_sum = [v for v in tf.global_variables() if v.name == "lod_in:0"][0]
    tf.summary.scalar("lod", lod_in_sum)

    # lr_sum = [v for v in tf.global_variables() if v.name == "learning_rate:0"][0]
    # tf.summary.scalar("Learning_Rate", lr_sum)
    # beta1_sum = [v for v in tf.global_variables() if v.name == "beta_1:0"][0]
    # tf.summary.scalar("Beta1", beta1_sum)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    saver = tf.train.Saver(max_to_keep=1)
    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    lod_old = 6.0
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None and os.path.isfile(a.output_dir + "/checkpoint"):
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results, a)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets, a)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)

        else:

            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                # ---------------------------------------------------------------------
                # Training Schedule and LOD update
                # ---------------------------------------------------------------------
                # Adapt Interpolation Parameter lod_in:
                TS = TrainingSchedule(int(results["global_step"] * a.batch_size))
                lod_new = TS.lod

                if lod_new != lod_old:
                    lod_in_var = [v for v in tf.global_variables() if v.name == "lod_in:0"][0]
                    lod_in_var.load(lod_new, sess)

                # Reset Optimizers if there is a Phase Transition
                if a.reset_opti:
                    if ((lod_new % 1.0 == 0) or (lod_old % 1.0 == 0)) and (lod_old != lod_new):
                        # print(lod_old, lod_new, "Optimizer is reset")
                        lr_var = [v for v in tf.global_variables() if v.name == "learning_rate:0"][0]
                        lr_var.load(a.lr, sess)
                        beta1_var = [v for v in tf.global_variables() if v.name == "beta_1:0"][0]
                        beta1_var.load(a.beta1, sess)

                lod_old = lod_new
                # =====================================================================



                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])
                '''
                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    steps_per_epoch                append_index(filesets, step=True)
                '''

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("epoch %d,  step %d, cur_nimg %d,  image/sec %0.1f,  remaining minutes %d, lod: %d" % \
                          (train_epoch, train_step, TS.stored_cur_nimg, rate, remaining / 60, lod_new))

                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):
                    # print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                # if step % len(examples.paths) - 1 == 0:

                if sv.should_stop():
                    break


main()
