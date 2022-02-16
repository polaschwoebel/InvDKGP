# DeepKernelInvariances

Example run: invariant deep kernel GP on rotated MNIST.

1. pretrain CNN:
`python runner_script.py -use_model CNN  -path 'experiment_results/demorun/' -dataset 'rotMNIST'  
-nr_epochs 200 -GPU 0 -CNN_architecture small 
 -optimizer adam -learning_rate 0.001 -batch_size 200`


2. train invariant deep kernel GP on top
`python runner_script.py -use_model SVGP -deepkernel -path 'experiment_results/demorun/' -dataset 'rotMNIST' 
-batch_size 200 -nr_inducing_points 750 -ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 2 
-pretrained_CNN_path 'experiment_results/demorun/model_CNN_rotMNIST_small_STN_preprocessing_lr_0.001_batch_size_200_adam'
-invariant -use_orbit seven_param_affine -orbit_size 90 
-likelihood_variance 0.05 -experiment_config 'coord_ascent_training' -toggle_after_steps 15000 -basekern_lengthscale 50 
 -full_batch_frequency 5000 -CNN_architecture small -fix_likelihood -fix_kernel_variance`


For NeurIPS paper runs see batch scripts in `deepkerneinv_experiments.experiment_scripts.bash_scripts/*`.# InvDKL
