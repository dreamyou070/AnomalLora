from transformers import CLIPTokenizer

def load_tokenizer(args):
    original_path = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(original_path)
    return tokenizer