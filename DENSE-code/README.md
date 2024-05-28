

Modified based on the official code of DENSE ([zj-jayzhang/DENSE: Official PyTorch implementation of DENSE (NeurIPS 2022) (github.com)](https://github.com/zj-jayzhang/DENSE))



Train DENSE on EMNIST

```
python3 heter_fl.py  --type=pretrain  --batch_size=64 --lr=1e-3 --model=cnn --dataset=emnist --beta=0.5 --seed=2020 --num_users=10 --local_ep=200 --device cuda:1


python heter_fl.py --type=kd_train --epochs=200 --lr=1e-3 --batch_size 64 --synthesis_batch_size=64 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/emnist --other=emnist --model=cnn --dataset=emnist --adv=1 --beta=0.5 --seed=2020 --device cuda:0

```

