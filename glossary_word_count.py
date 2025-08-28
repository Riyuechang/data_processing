import json

import opencc
from transformers import AutoTokenizer


GLOSSARY_PATH = "./translation/sakura_gpt_dict_沈黙の魔女の隠しごと.json"

MODEL_NAME = "Sakura-1.5B-Qwen2.5-v1.0-HF"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/LLMModel/{MODEL_NAME}"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

opencc_converter = opencc.OpenCC('tw2s.json')

with open(GLOSSARY_PATH, "r", encoding="utf-8") as file:
    glossary_dict: list[dict[str, str]] = json.load(file)

glossary_list = [
    f"{glossary['jp']}->{opencc_converter.convert(glossary['tw'])} #{opencc_converter.convert(glossary['info'])}" if glossary["info"] else f"{glossary['jp']}->{opencc_converter.convert(glossary['tw'])}"
    for glossary in glossary_dict
]
glossary_prompt = "\n".join(glossary_list)

print(glossary_prompt)
print(len(glossary_prompt))
print(len(tokenizer(glossary_prompt)["input_ids"]))
