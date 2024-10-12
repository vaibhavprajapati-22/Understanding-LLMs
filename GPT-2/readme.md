## GPT-2 Architecture 
![gpt2 architecture](https://github.com/user-attachments/assets/0484396c-e1c4-444f-8f2e-5222fb847e0b)
<p align="center">
  <a href="https://medium.com/@hsinhungw/gpt-2-detailed-model-architecture-6b1aad33d16b">Image Source</a>
</p>

## File Description 
1. fineweb.py : Python script to download and save 100M tokens as numpy array.
2. gpt2.py : Architecture of GPT-2 model
3. train.ipynb : Jupyter Notebook showing the pre-training of gpt2 model
4. train_gpt2.py : Python script for the pre-training of the gpt-2 model

## Key Learnings 
1. Shared Weights Between Embedding and Logits Layers:
   <br>
   In GPT-2, the word embedding layer and the logits layer share the same weights. This weight sharing helps reduce the overall parameter count and improves model     efficiency.
2. Precision Optimization in Deep Learning Models:
   <br>
   Although most deep learning models use float32 by default, they can often tolerate lower precision without loss in accuracy. This can improve performance and memory usage:
   * torch.set_float32_matmul_precision('high'): Ensures computations are done in TF32 instead of float32, enhancing performance on supported hardware.
   * Using torch.autocast(device_type=device.type, dtype=torch.bfloat16): This allows automatic mixed precision, which further reduces memory usage and speeds up training.
3. Compilation with torch.compile():
   <br>
   torch.compile() enables ahead-of-time (AOT) compilation for PyTorch models, optimizing the computation graph and improving the performance of training and inference.
4. Cosine Learning Rate Scheduler:
   <br>
   This technique smoothly decays the learning rate using a cosine function, which helps in gradually lowering the learning rate during training, leading to better model convergence and preventing overshooting of the optimal weights.
5. Weight Decay for Regularization:
   <br>
   Applying weight decay helps in regularizing the model by penalizing large weights, preventing overfitting, and ensuring the model generalizes better on unseen data.
6. Gradient Accumulation for Large Batch Sizes:
   <br>
   To simulate larger batch sizes without increasing memory usage, gradient accumulation is used. Instead of updating weights after every batch, gradients are accumulated over multiple batches before updating, allowing more efficient training on limited resources.

## References 
[Let's reproduce GPT-2 (124M) by Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU&t=13410s)
