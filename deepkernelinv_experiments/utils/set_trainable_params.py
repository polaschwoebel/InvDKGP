from gpflow.utilities import set_trainable


def set_trainable_basekern_var(args, model, truth_val):
    if args.use_model in ['GP', 'GPR', 'SGPR', 'SVGP', 'SampleSVGP'] and not args.deepkernel:
        set_trainable(model.kernel.variance, truth_val)

    # both invariant and deep
    elif args.invariant and args.deepkernel:
        set_trainable(model.kernel.basekern.basekern.variance, truth_val)

    # either invariant or deep
    elif args.invariant or args.deepkernel:
        set_trainable(model.kernel.basekern.variance, truth_val)


def set_trainable_basekern_cnn(args, model, truth_val):
    # both invariant and deep
    if args.invariant and args.deepkernel:
        set_trainable(model.kernel.basekern.cnn, truth_val)

    # deep
    elif args.invariant or args.deepkernel:
        set_trainable(model.kernel.cnn, truth_val)

    else:
        print('Attempting to toggle CNN for shallow model - exit.')
        exit()