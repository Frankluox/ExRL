import os
from transformers import AutoModel, pipeline, AutoTokenizer, CLIPTextModelWithProjection

os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
tokenizer = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"
LANG_ENABLED = os.environ.get("ROBOMIMIC_ENABLE_LANG", "1") == "1"
if LANG_ENABLED:
    lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
        tokenizer,
        cache_dir=os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip"))
    ).eval()
    tz = AutoTokenizer.from_pretrained(tokenizer, TOKENIZERS_PARALLELISM=True)
else:
    lang_emb_model = None
    tz = None

LANG_EMB_OBS_KEY = "lang_emb"

def get_lang_emb(lang):
    if not LANG_ENABLED:
        return None
    if lang is None:
        return None
    
    tokens = tz(
        text=lang,                   # the sentence to be encoded
        add_special_tokens=True,             # Add [CLS] and [SEP]
        max_length=25,  # maximum length of a sentence
        padding="max_length",
        return_attention_mask=True,        # Generate the attention mask
        return_tensors="pt",               # ask the function to return PyTorch tensors
    )
    lang_emb = lang_emb_model(**tokens)['text_embeds'].detach()[0]

    return lang_emb

def get_lang_emb_shape():
    if not LANG_ENABLED:
        raise RuntimeError("Language embedding is disabled (ROBOMIMIC_ENABLE_LANG=0).")
    return list(get_lang_emb('dummy').shape)
