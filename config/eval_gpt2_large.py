# evaluate the base gpt2
# n_layer=36, n_head=20, n_embd=1280
# 774M parameters
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'gpt2-large'

#coefficients to determine where we should put the two centroids
shiftCoefficient0 = 0.253
shiftCoefficient1 = 0.743
haveReadShiftCoefficients = True
shouldUseOpenWebText = True
