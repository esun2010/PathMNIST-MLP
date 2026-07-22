# Instructions to run
IMPORTANT: must create folder named "checkpoints"
Run train.py for training and the test.py for validation.
# Misc info
model.py includes model + helper functions. 

train.py will save the model and running it again will continue training from last epoch.
# Model details
Architecture: 3x28x28 -> 512 -> 512 -> 128 -> 9

Loss Function: Cross Entropy Loss

Activation Function: ReLU

Optimizer: Adam
