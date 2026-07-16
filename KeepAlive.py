# keep_alive.py
import time
import torch

print("Keep-alive started. Will run for 12 hours.")
tensor = torch.zeros(1000, 1000, device="cuda")

start = time.time()
hours = 12
while time.time() - start < hours * 3600:
    tensor = tensor + 1
    time.sleep(10)

print("Keep-alive done.")