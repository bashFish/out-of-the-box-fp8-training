from nanoGPT.model import GPT
from notebook_utils import config, train
from unit_scaling.transforms import simulate_fp8, unit_scale

import torch

gpt = GPT(config)  # model unchanged from original nanoGPT
unit_scaled_gpt = unit_scale(gpt)
fp8_gpt = simulate_fp8(gpt)
unit_scaled_fp8_gpt = unit_scale(fp8_gpt)
# unit_scaled_fp8_gpt = torch.compile(unit_scale(fp8_gpt))

models = [unit_scaled_gpt, gpt, unit_scaled_fp8_gpt, fp8_gpt]
# import pdb;pdb.set_trace()
for model in models:
    train(model, mode="file")
