# Jamming_Detection

1. [quantizer.py](quantizer.py) will generate data for various input probability and then after quantization at the reciever, use basic algorithm to detect jamming attack assuming the receiver knows when there is any attack. Following curve is generated:
![Attack Detection Curve](Accuracy.png?raw=true)
2. [gen_data.py](gen_data.py) generates data for two taps where one tap is always attacked and one is never affected externally.
3. [train.py](train.py) trains a neural network to detect jamming attack based on the data generated by running [gen_data.py](gen_data.py).

To train a neural network
1. Edit [gen_data.py](gen_data.py) by changing the following values
```python
p = 0.4 # Input Probability Distribution Followed
SNR = 1.0 # SNR of the both the channels
N = 100000 # Number of samples transmitted from each channel
```
2. Run `python gen_data.py`.

3. Run `python train.py`

## Requirements
1. Python 2/3
2. Matplotlib
3. Tensorflow 1.8 (GPU/CPU)
4. Keras 2.2.2



