# Simple startup -
Works for Mac, but not optimized at all.

1. Run `poetry install` to install required packages
2. Run `poetry run python traditional_lora_finetune.py` to train the model. Model to train is mentioned in this file along with the fine tunning dataset. 
3. Once training is finished, run `poetry run python talk_to_fine_model.py` to get chatty with the model.


## Path Ahead -
Optimize the shit out of this.
Explore Packages like Unsloth that a way faster than Lora/Qlora. and can support in fine tunning bog models on small machines.
Building a training pipeline that does the standard two stepped approach for training LLMs -
- RL using input dataset
- Human in loop for further enhancement

