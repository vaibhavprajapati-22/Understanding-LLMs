from gpt2 import GPT, GPTConfig
import torch
import numpy as np
import time
import os
import math


class DataLoader:

    def __init__(self, B, T):
        self.B = B
        self.T = T
        # Loading Tokens
        tokens = np.load('fineweb_edu_tokenized/tokenized_100M.npy')
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T

        if self.current_position + B * T >= len(self.tokens):
            self.current_position = 0

        return x, y


total_batch_size = 1024 * 4  #524288
B = 4  # mini batch size
T = 1024  # max sequence length
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size : {total_batch_size}")
print(f"=> calculated gradient accumulated steps : {grad_accum_steps}")

train_loader = DataLoader(B, T)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GPT(GPTConfig())
model.to(device)
# model = torch.compile(model) Needed GPU A100 or latest (Kaggle GPUs do not work)

torch.set_float32_matmul_precision('high')

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device.type)

# Directory to save logs
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"log.txt")

# Directory to save checkpoints
checkpoint_dir = "checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# Function to save a checkpoint
def save_checkpoint(step, model, optimizer, loss_acc, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"gpt2_checkpoint_step_{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_acc,
    }, checkpoint_path)
    print(f"Checkpoint saved at step {step}")


max_steps = 200

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 50


def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    elif step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_acc = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device.type, dtype=torch.float16):
        #     logits, loss = model(x, y) Needed GPU A100 (Kaggle GPUs do not work)
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_acc += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Get learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    # torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_seconds = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)
    print(
        f'Step {step}, Loss: {loss_acc.item():.4f}, Norm : {norm}, dt : {dt:.2f}ms, tokens/sec : {tokens_per_seconds:.2f}s')
    with open(log_file, 'a') as f:
        f.write(f"{step} train {loss_acc.item():.6f}\n")

    if (step + 1) % 25 == 0:
        save_checkpoint(step, model, optimizer, loss_acc, checkpoint_dir)
