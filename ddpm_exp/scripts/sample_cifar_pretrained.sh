python prune.py \
--config cifar10.yml \
--exp run/sample/ddim_cifar10_official \
--sample \
--use_pretrained \
--timesteps 100 \
--eta 0 \
--ni \
--doc sample_100k \
--skip_type quad  \
--pruning_ratio 0.0 \
--fid \
--use_ema