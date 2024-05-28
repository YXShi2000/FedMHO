## FedMHO: Heterogeneous One-Shot Federated Learning Towards Computing Resource-Constrained Clients



This repository is the official implementation of the paper “FedMHO: Heterogeneous One-Shot Federated Learning Towards Computing Resource-Constrained Clients”.



**Note: We use PyTorch with MPI backend for a Master-Worker computation/communication topology. Therefore, do not use the "pip install" command to install PyTorch!**



Train FedMHO-MD on EMNIST

```bash
python run.py     --arch vgg9 --complex_arch master=vgg9,worker=vgg9:cvae,num_clients_per_model=5 --experiment heterogeneous     --data emnist --pin_memory True --batch_size 64 --num_workers 8     --partition_data non_iid_dirichlet --non_iid_alpha 0.5   --train_data_ratio 1 --val_data_ratio 0    --n_clients 10 --participation_ratio 1 --n_comm_rounds 1 --local_n_epochs 200   --world_conf 0,0,1,1,100 --on_cuda True     --fl_aggregate scheme=semi_supervised,clean_rate=0.8     --optimizer sgd --lr 1e-3 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150     --lr_scheduler MultiStepLR --lr_decay 0.01     --weight_decay 0 --use_nesterov False --momentum_factor 0.9     --track_time True --display_tracked_time True --python_path  /home/dzyao/anaconda3/envs/pytorch/bin/python    --hostfile hostfile     --manual_seed 2023  --pn_normalize False --same_seed_process True     --use_hog_feature False   --pin_memory False  --latent_size 20  --num_synthesis_sample 12000   --is_md True
```



Train FedMHO-SD on EMNIST

```bash
python run.py     --arch vgg9 --complex_arch master=vgg9,worker=vgg9:cvae,num_clients_per_model=5 --experiment heterogeneous     --data emnist --pin_memory True --batch_size 64 --num_workers 8     --partition_data non_iid_dirichlet --non_iid_alpha 0.5   --train_data_ratio 1 --val_data_ratio 0    --n_clients 10 --participation_ratio 1 --n_comm_rounds 1 --local_n_epochs 200   --world_conf 0,0,1,1,100 --on_cuda True     --fl_aggregate scheme=semi_supervised,clean_rate=0.8     --optimizer sgd --lr 1e-3 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150     --lr_scheduler MultiStepLR --lr_decay 0.01     --weight_decay 0 --use_nesterov False --momentum_factor 0.9     --track_time True --display_tracked_time True --python_path  /home/dzyao/anaconda3/envs/pytorch/bin/python    --hostfile hostfile     --manual_seed 2023  --pn_normalize False --same_seed_process True     --use_hog_feature False   --pin_memory False  --latent_size 20  --num_synthesis_sample 12000   --is_sd True
```



Train FedAvg on EMNIST

```bash
python run.py     --arch vgg9  --complex_arch master=vgg9,worker=vgg9:cnn,num_clients_per_model=5  --experiment heterogeneous     --data emnist --pin_memory True --batch_size 64 --num_workers 8     --partition_data non_iid_dirichlet --non_iid_alpha 0.5   --train_data_ratio 1 --val_data_ratio 0    --n_clients 10 --participation_ratio 1 --n_comm_rounds 1 --local_n_epochs 200    --world_conf 0,0,1,1,100 --on_cuda True     --fl_aggregate scheme=federated_average_ensemble     --optimizer sgd --lr 1e-3 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150     --lr_scheduler MultiStepLR --lr_decay 0.01     --weight_decay 0 --use_nesterov False --momentum_factor 0.9     --track_time True --display_tracked_time True --python_path /home/dzyao/anaconda3/envs/pytorch/bin/python3.8     --hostfile hostfile     --manual_seed 2020 --pn_normalize False --same_seed_process True   --use_hog_feature False   --pin_memory False
```



Train FedDF on EMNIST

```bash
python run.py     --arch vgg9  --complex_arch master=vgg9,worker=vgg9:cnn,num_clients_per_model=5  --experiment heterogeneous     --data emnist --pin_memory True --batch_size 64 --num_workers 8     --partition_data non_iid_dirichlet --non_iid_alpha 0.5   --train_data_ratio 1 --val_data_ratio 0.1    --n_clients 10 --participation_ratio 1 --n_comm_rounds 200 --local_n_epochs 1    --world_conf 0,0,1,1,100 --on_cuda True     --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=random_sampling,data_name=mnist,data_percentage=1.0,total_n_server_pseudo_batches=6000,eval_batches_freq=200,early_stopping_server_batches=400     --optimizer sgd --lr 1e-3 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150     --lr_scheduler MultiStepLR --lr_decay 0.01     --weight_decay 0 --use_nesterov False --momentum_factor 0.9     --track_time True --display_tracked_time True --python_path /home/dzyao/anaconda3/envs/pytorch/bin/python3.8     --hostfile hostfile     --manual_seed 2023 --pn_normalize False --same_seed_process True   --use_hog_feature False   --pin_memory False
```



**Besides, we are based on DENSE's official code with minor modifications to suit our experimental setup, and the code is provided in the DENSE-code folder.**
