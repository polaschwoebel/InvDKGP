# #!/bin/sh

# CNN: 
python runner_script.py -use_model CNN -CNN_architecture 'litjens_vgg'  -path 'experiment_results/PCAM_experiment/' -dataset PCAM \
 -nr_classes 2  -nr_epochs 5 -batch_size 64 -optimizer adam  -learning_rate 0.001  -GPU 2 -decay_lr steps -full_batch_frequency 2000 -dropout_prob 0.5\
 -latent_dim 50 -precision double  

# non inv. DKGP:
python runner_script.py -use_model SampleSVGP -matheron -likelihood Softmax -deepkernel -path 'experiment_results/PCAM_experiment/' \
-dataset 'PCAM' -batch_size 32 32 -nr_inducing_points 750 -ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 2 \
-pretrained_CNN_path 'experiment_results/PCAM_experiment/model_CNN_PCAM_litjens_vgg_dropout_0.5_lr_0.001_decay_lr_steps_batch_size_64_adam_ltdm_50_float64' -likelihood_variance 0.05 \
 -basekern_lengthscale 10 -full_batch_frequency 5000 -CNN_architecture litjens_vgg -learning_rate 0.001 0.0001 -latent_dim 50 -nr_classes 2 -ARD -precision double  \
 -experiment_config 'coord_ascent_training' -toggle_after_steps 50000 

# CNN + STN/DA preprocessing:
python runner_script.py -use_model CNN -CNN_architecture litjens_vgg  -path 'experiment_results/PCAM_experiment/' \
-dataset PCAM -nr_classes 2  -nr_epochs 5 -batch_size 64 -optimizer adam  -learning_rate 0.001  -GPU 6 -decay_lr steps -full_batch_frequency 2000 \
-DA_rotate -precision double -radians True  -use_stn 

# InvDGKP + STN/DA preprocessing + learned invariance
python runner_script.py -use_model SampleSVGP -matheron -likelihood Softmax -deepkernel -path 'experiment_results/PCAM_experiment/'  -dataset 'PCAM' -batch_size 32 32 -nr_inducing_points 750 -ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 0 -pretrained_CNN_path 'experiment_results/25_05_PCAM_full_da_pretrain/model_CNN_PCAM_litjens_vgg_lr_0.001_decay_lr_steps_batchs_64_adam_DA_rotate_0.0_ltdm_50' -basekern_lengthscale 1 -full_batch_frequency 5000 -CNN_architecture litjens_vgg  -learning_rate 0.001  0.0001 -latent_dim 50 -nr_classes 2 -precision double -nr_eval_batches 1000 -orbit_size 20 20 -use_orbit sampled_rot -angle 0.31 -invariant -use_stn -experiment_config 'coord_ascent_training' -toggle_after_steps 75000 -name_appendix '_init_1' -mode_test

# InvDGKP + STN/DA preprocessing + fixed invariance
python runner_script.py -use_model SampleSVGP -matheron -likelihood Softmax -deepkernel -path 'experiment_results/PCAM_experiment/' \
-dataset 'PCAM' -batch_size 32 32 -nr_inducing_points 750 -ind_point_init 'inducing-init' -nr_epochs 10000 -GPU 2 \
-pretrained_CNN_path 'experiment_results/PCAM_experiment/model_CNN_PCAM_litjens_vgg_lr_0.001_decay_lr_steps_batchs_64_adam_DA_rotate_0.0_ltdm_50'\
 -basekern_lengthscale 10 -full_batch_frequency 5000 -CNN_architecture litjens_vgg  -learning_rate 0.001  0.0001 -latent_dim 50 -nr_classes 2 -precision double \
 -nr_eval_batches 1000 -orbit_size 20 20 -use_orbit sampled_rot -fix_orbit -angle 0.31 -invariant -use_stn -experiment_config 'coord_ascent_training' -toggle_after_steps 50000 -mode test

