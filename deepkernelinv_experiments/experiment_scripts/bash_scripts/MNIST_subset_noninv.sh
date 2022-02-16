# #!/bin/sh

SIZES=(312 1250 5000 20000 60000)
NR_EPOCHS=(80000 12000 3000 1000 260)
INDP=(312 1200 1200 1200 1200)
FILEPATHS=(\
	'experiment_results/03_05_MNIST_subsets_neurips/model_CNN_MNIST_312_small_lr_0.001_0.001_batch_size_200_adam_adam' \
	'experiment_results/03_05_MNIST_subsets_neurips/model_CNN_MNIST_1250_small_lr_0.001_0.001_batch_size_200_adam_adam' \
	'experiment_results/03_05_MNIST_subsets_neurips/model_CNN_MNIST_5000_small_lr_0.001_0.001_batch_size_200_adam_adam' \
	'experiment_results/03_05_MNIST_subsets_neurips/model_CNN_MNIST_20000_small_lr_0.001_0.001_batch_size_200_adam_adam' \
	'experiment_results/03_05_MNIST_subsets_neurips/model_CNN_MNIST_60000_small_lr_0.001_0.001_batch_size_200_adam_adam' \
	)

for DATASETSIZE in {0..4}
do
	python runner_script.py -use_model SVGP -deepkernel -path 'experiment_results/17_05_MNIST_subsets_neurips/' -dataset 'MNIST' -subset_size ${SIZES[$DATASETSIZE]} \
	-batch_size 200 -learning_rate 0.001 -nr_inducing_points ${INDP[$DATASETSIZE]} -likelihood_variance 0.05 -full_batch_frequency 5000 \
	-CNN_architecture small -GPU 7 -basekern_lengthscale 10 \
	-pretrained_CNN_path  ${FILEPATHS[$DATASETSIZE]} \
	-nr_epochs ${NR_EPOCHS[$DATASETSIZE]} 
done
