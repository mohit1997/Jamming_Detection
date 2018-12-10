# Jamming_Detection

1. `gen_data.py` generates data for two taps where one tap is always attacked and one is never affected externally.
2. `quantizer.py` will generate data for various input probability and then after quantization at the reciever, use basic algorithm to detect jamming attack assuming the receiver knows when there is any attack.
3. `train.py` trains a neural network to detect jamming attack.


![Attack Detection Curve](Accuracy.png?raw=true)
