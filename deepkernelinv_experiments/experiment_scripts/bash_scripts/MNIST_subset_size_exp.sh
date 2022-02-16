# #!/bin/sh

SIZES=(312 1250 5000 20000 60000)
NR_EPOCHS=(6400 3200 800 400 100)

for DATASETSIZE in {0..4}
do
       python runner_script.py -use_model CNN  -path 'experiment_results/MNIST_subsets_experiment/' -dataset 'MNIST' \
       -nr_epochs ${NR_EPOCHS[$DATASETSIZE]} -GPU 6 -batch_size 200 -CNN_architecture small -subset_size ${SIZES[$DATASETSIZE]}
done


SIZES=(312 1250 5000 20000 60000)
NR_EPOCHS=(80000 12000 3000 1000 260)
FILEPATHS=(\
 	'experiment_results/MNIST_subsets_experiment/model_CNN_MNIST_312_small_lr_0.001_0.001_batch_size_200_adam_adam' \
 	'experiment_results/MNIST_subsets_experiment/model_CNN_MNIST_1250_small_lr_0.001_0.001_batch_size_200_adam_adam' \
 	'experiment_results/MNIST_subsets_experiment/model_CNN_MNIST_5000_small_lr_0.001_0.001_batch_size_200_adam_adam' \
 	'experiment_results/MNIST_subsets_experiment/model_CNN_MNIST_20000_small_lr_0.001_0.001_batch_size_200_adam_adam' \
 	'experiment_results/MNIST_subsets_experiment/model_CNN_MNIST_60000_small_lr_0.001_0.001_batch_size_200_adam_adam' \
 	)

for DATASETSIZE in {0..4}
do
 	python runner_script.py -use_model SVGP -deepkernel -path 'experiment_results/MNIST_subsets_experiment/' -dataset 'MNIST' -subset_size ${SIZES[$DATASETSIZE]} \
 	-batch_size 200 200  -learning_rate 0.001 0.001 -nr_inducing_points 1200 -likelihood_variance 0.05 -full_batch_frequency 5000 \
 	-CNN_architecture small -GPU 5 -basekern_lengthscale 10 \
 	-pretrained_CNN_path ${FILEPATHS[$DATASETSIZE]} \
 	-invariant  -orbit_size 120 120 -use_orbit seven_param_affine -affine_init 0.02 \
 	-nr_epochs ${NR_EPOCHS[$DATASETSIZE]} -experiment_config coord_ascent_training -toggle_after_steps 25000 -fix_likelihood -fix_kernel_variance -run 2
done
