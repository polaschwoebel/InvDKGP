import invgp
from gpflow.utilities import set_trainable
import numpy as np
import tensorflow as tf


def initialize_orbit(args, samples=None):
    samples = args.orbit_size[0] if samples is None else samples
    if args.use_orbit == 'quantized_rot':
        angle = 359. if args.angle is None else args.angle
        print('angle is ', angle)
        print('use stn is', args.use_stn)
        orbit = invgp.kernels.orbits.ImageRotQuant(
            orbit_size=samples,  # always start with the first one (GP training)
            angle=angle,
            input_dim=args.image_size,
            interpolation_method="BILINEAR", 
            use_stn=args.use_stn)

    elif args.use_orbit == 'sampled_rot':
        angle = 1. if args.angle is None else args.angle
        orbit = invgp.kernels.orbits.ImageRotation(
            angle=angle,
            input_dim=tf.cast(args.image_size, np.float32),
            minibatch_size=samples,
            use_stn=args.use_stn,
            radians=args.radians,
            interpolation_method="BILINEAR")

    elif args.use_orbit == 'inverse_angle_rot':
        angle = 1. if args.angle is None else args.angle
        orbit = invgp.kernels.orbits.InverseAngleImageRotation(
            angle=angle,
            input_dim=tf.cast(args.image_size, np.float32),
            minibatch_size=samples,
            use_stn=args.use_stn,
            radians=args.radians,
            interpolation_method="BILINEAR")

    elif args.use_orbit == 'general_affine':
        orbit = invgp.kernels.orbits.GeneralSpatialTransform(
            input_dim=args.image_size,
            minibatch_size=samples, initialization=args.affine_init)

    elif args.use_orbit == 'seven_param_affine':
        orbit = invgp.kernels.orbits.InterpretableSpatialTransform(
            theta_min=np.array([-args.radian_init, 1., 1., 0., 0., 0., 0.]) - args.affine_init * np.array([0., 1., 1., 1., 1., 1., 1.]),  # play with initialization
            theta_max=np.array([args.radian_init, 1., 1., 0., 0., 0., 0.]) + args.affine_init * np.array([0., 1., 1., 1., 1., 1., 1.]),  # play with initialization
            input_dim=args.image_size,
            # img_size=args.image_shape,
            minibatch_size=samples,
            radians=args.radians,
            colour=args.dataset in ["CIFAR10", "PCAM", "PCAM32"]) 

    elif args.use_orbit == 'colorspace':
        orbit = invgp.kernels.orbits.ColorTransform(
            input_dim=args.image_size,
            minibatch_size=samples,
            log_lims_contrast=[-1., 1.],
            log_lims_brightness=[-1., 1.],)

    else:
        print('Please pass valid orbit!')
        exit()

    if args.fix_angle:
        set_trainable(orbit.angle, False)
    if args.fix_orbit:
        set_trainable(orbit, False)

    return orbit
