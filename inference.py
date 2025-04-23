import tiktoken
import torch
from torch import nn

from src.utils.trainer import autodetect_device


def generate(model: nn.Module, prompt: str, max_len: int = 32, repeats: int = 1, enc_type="gpt2"):
    model.eval()
    device = autodetect_device()
    num_generations = repeats

    enc = tiktoken.get_encoding(enc_type)

    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_generations, 1)
    x = tokens.to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_len:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(top_k_probs, 1)
            xcol = torch.gather(top_k_indices, -1 , ix)
            x = torch.cat((x, xcol), dim=1)

    generations = []
    for i in range(num_generations):
        tokens = x[i, :max_len].tolist()
        decoded = enc.decode(tokens)
        generations.append(decoded)
    return generations