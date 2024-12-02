
# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = False #this used to be set to true but I don't want it messing with our run
wandb_project = 'owt'
wandb_run_name='gpt2-124M'


# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# model
n_layer = 12
n_head = 12
n_embd = 768

learning_rate = 6e-5 # max learning rate
# this makes total number of tokens be 300B
max_iters = 20000
lr_decay_iters = 20000
warmup_iters = 0

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

# weight decay
weight_decay = 1e-1
shouldUseOpenWebText = True

#coefficients to determine where we should put the two centroids
shiftCoefficient0 = 0.253
shiftCoefficient1 = 0.743
haveReadShiftCoefficients = True
shouldUseOpenWebText = True
