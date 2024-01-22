# from google_cloud_gpu_utils import *

# # Define hyperparameters
# epsilons = [1.0, 1.5, 8.0]
# clippings = [1.0, 0.9]
# sparsities = [0.01, 0.1, 0.5]
# epochss = [200, 100]
# batch_sizes = [5000, 500]
# classifier_lrs = [0.8]
# lrs = [0.01]
# adaptive_bools = [False]

# # Generate jobs
# jobs = []
# for task_id in range(48, 72):  # Adjust the range as per your requirements
#     epsilon = epsilons[task_id % len(epsilons)]
#     task_id //= len(epsilons)

#     clipping = clippings[task_id % len(clippings)]
#     task_id //= len(clippings)

#     sparsity = sparsities[task_id % len(sparsities)]
#     task_id //= len(sparsities)

#     epochs = epochss[task_id % len(epochss)]
#     task_id //= len(epochss)

#     batch_size = batch_sizes[task_id % len(batch_sizes)]
#     task_id //= len(batch_sizes)

#     classifier_lr = classifier_lrs[task_id % len(classifier_lrs)]
#     task_id //= len(classifier_lrs)

#     lr = lrs[task_id % len(lrs)]
#     task_id //= len(lrs)

#     adaptive_bool = adaptive_bools[task_id % len(adaptive_bools)]
#     task_id //= len(adaptive_bools)

#     command = f"python3 -m train_cifar --dataset 'cifar10' --batch_size {batch_size} --model 'resnet18' --num_classes 10 --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --use_adaptive_lr {adaptive_bool} --wd 0.0 --momentum 0.9 --lr_schedule_type 'warmup_cosine' --clip_gradient False --num_epochs {epochs} --finetune_strategy 'all_layers' --use_gn True --use_magnitude_mask True --sparsity {sparsity} --use_dp True --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir 'dp_finegrained_to_add' --out_file 'out_file.txt' --seed 0  --SLURM_JOB_ID google_cloud_magnitude_mask --TASK_ID {task_id}"
#     jobs.append(command)

# run_all_jobs(jobs)
