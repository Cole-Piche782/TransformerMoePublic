# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'gpt2'


#coefficients to determine where we should put the two centroids
shiftCoefficient0 = 0.253
shiftCoefficient1 = 0.743
haveReadShiftCoefficients = True
shouldUseOpenWebText = True
