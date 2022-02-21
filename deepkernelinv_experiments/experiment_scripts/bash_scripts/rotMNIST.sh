# #!/bin/sh

#M1:
python runner_script.py -use_model CNN  -path 'experiment_results/rotMNIST_experiment/' -nr_epochs 200 -GPU 1 -batch_size 200 -dataset rotMNIST

#M2: 
python runner_script.py -use_model SVGP -path 'experiment_results/rotMNIST_experiment/' -dataset 'rotMNIST' -batch_size 200 -nr_inducing_points 1200 \
-ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 1 -likelihood_variance 0.02 -basekern_lengthscale 10 \
-full_batch_frequency 5000 -learning_rate 0.001 

# M3: 
python runner_script.py -use_model SampleSVGP -matheron -path 'experiment_results/rotMNIST_experiment/' -dataset 'rotMNIST' -batch_size 200 -nr_inducing_points 1200 \
-ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 2 -likelihood_variance 0.02 -basekern_lengthscale 10 \
-full_batch_frequency 5000 -learning_rate 0.001 -likelihood Softmax

#M4: 
python runner_script.py -use_model SVGP -path 'experiment_results/rotMNIST_experiment/' -dataset 'rotMNIST' -batch_size 200 -nr_inducing_points 1200 \
-ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 2 -invariant -use_orbit seven_param_affine  -use_stn -orbit_size 120 -likelihood_variance 0.02 -basekern_lengthscale 10 \
-full_batch_frequency 5000 -learning_rate 0.001 

# M5: 
python runner_script.py -use_model SampleSVGP -matheron -path 'experiment_results/rotMNIST_experiment/' -dataset 'rotMNIST' -batch_size 200 -nr_inducing_points 1200 \
-ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 6 -invariant -use_orbit seven_param_affine  -use_stn -orbit_size 120 -likelihood_variance 0.02 -basekern_lengthscale 10 \
-full_batch_frequency 5000 -learning_rate 0.001 -likelihood Softmax

#M6: 
python runner_script.py -use_model SVGP -deepkernel -path 'experiment_results/rotMNIST_experiment/' -dataset 'rotMNIST' -batch_size 200 -nr_inducing_points 1200 \
-ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 0 -pretrained_CNN_path 'experiment_results/rotMNIST_experiment/model_CNN_rotMNIST_small_lr_0.001_0.001_batch_size_200_adam_adam' \
-CNN_architecture small -learning_rate 0.001 -likelihood Gaussian

# M7: 
python runner_script.py -use_model SampleSVGP -matheron -deepkernel -path 'experiment_results/rotMNIST_experiment/' -dataset 'rotMNIST' -batch_size 200 -nr_inducing_points 1200 \
-ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 1 -pretrained_CNN_path 'experiment_results/rotMNIST_experiment/model_CNN_rotMNIST_small_lr_0.001_0.001_batch_size_200_adam_adam' \
-CNN_architecture small -learning_rate 0.001 -likelihood Softmax -basekern_lengthscale 20  

# M8:
python runner_script.py -use_model SVGP -deepkernel -path 'experiment_results/rotMNIST_experiment/' -dataset 'rotMNIST' -batch_size 200 200 -nr_inducing_points 1200 \
 -ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 4 -pretrained_CNN_path 'experiment_results/rotMNIST_experiment/model_CNN_rotMNIST_small_lr_0.001_0.001_batch_size_200_adam_adam' \
 -invariant -use_orbit seven_param_affine  -use_stn -orbit_size 120 120 -likelihood_variance 0.05 -experiment_config 'coord_ascent_training' -toggle_after_steps 30000 -basekern_lengthscale 50 \
 -full_batch_frequency 5000 -CNN_architecture small -learning_rate 0.003 0.003 -decay_lr steps cyclic -fix_kernel_variance -fix_likelihood