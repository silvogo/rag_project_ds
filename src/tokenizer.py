from transformers import AutoTokenizer
from llama_index.core import set_global_tokenizer

def set_up_tokenizer():
    set_global_tokenizer(AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer"))
    print("Tokenizer is set!")