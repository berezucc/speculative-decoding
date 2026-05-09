import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
