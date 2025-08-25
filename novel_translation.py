import os
import json

import opencc
from tqdm import tqdm
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoTokenizer


MAX_TOKENS = 2048 #8192
TEMPERATURE = 0.1
MIN_P = 0.1
TOP_P = 0.3
FREQUENCY_PENALTY = 0.2

MAX_REQUESTS = 32 #8 64 128 256 
MAX_BATCHED_TOKENS = 32768 #8192 16384 65536
VRAM_UTILIZATION = 0.95

MODEL_NAME = "Sakura-7B-Qwen2.5-v1.0-GGUF/sakura-7b-qwen2.5-v1.0-iq4xs.gguf"
#MODEL_NAME = "Sakura-GalTransl-7B-v3.7/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/LLMModel/{MODEL_NAME}"

TOKENIZER_NAME = "Sakura-1.5B-Qwen2.5-v1.0-HF"
TOKENIZER_PATH = f"/media/ifw/GameFile/linux_cache/LLMModel/{TOKENIZER_NAME}"

#NOVEL_NAME = "test"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
NOVEL_PATH = f"./output/{NOVEL_NAME}"

USE_GLOSSARY = True
GLOSSARY_PATH = "./translation/sakura_gpt_dict.json"

SAVE_DIR_PATH = f"./translation/{NOVEL_NAME}"


def vllm_add_request(input_text: str, request_id: str):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"},
            {"role": "user", "content": f"根据以下术语表（可以为空）：\n{glossary_prompt}\n将下面的日文文本翻译成中文：{input_text}" if USE_GLOSSARY else f"将下面的日文文本翻译成中文：{input_text}"}
        ], 
        add_generation_prompt=True, 
        tokenize=False
    )

    params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        min_p=MIN_P,
        top_p=TOP_P,
        #top_k=50,
        frequency_penalty=FREQUENCY_PENALTY,
        #seed=None
    )
    llm_engine.add_request(
        request_id=request_id,
        prompt=prompt, 
        params=params
    )


engine_args = EngineArgs(
    model=MODEL_PATH, 
    tokenizer=TOKENIZER_PATH,
    #dtype=torch.float16, 
    #quantization="bitsandbytes", 
    #load_format="bitsandbytes",
    gpu_memory_utilization=VRAM_UTILIZATION,
    max_model_len=MAX_TOKENS,
    max_num_batched_tokens=MAX_BATCHED_TOKENS,
    max_num_seqs=MAX_REQUESTS,
    enable_prefix_caching=True,
    enforce_eager=True,
    #swap_space=8,
    #max_seq_len_to_capture=8192,
    #cpu_offload_gb=1,
)
llm_engine = LLMEngine.from_engine_args(engine_args)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

opencc_converter = opencc.OpenCC('tw2s.json')

if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

if USE_GLOSSARY:
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as file:
        glossary_dict: list[dict[str, str]] = json.load(file)
    
    glossary_list = [
        f"{glossary['jp']}->{opencc_converter.convert(glossary['tw'])} #{opencc_converter.convert(glossary['info'])}" if glossary["info"] else f"{glossary['jp']}->{opencc_converter.convert(glossary['tw'])}"
        for glossary in glossary_dict
    ]
    glossary_prompt = "\n".join(glossary_list)

tqdm_progress = tqdm(novel_file_list)
for novel_file in tqdm_progress:
    tqdm_progress.set_description(novel_file)

    with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
        dataset: list[dict[str, str | list[str]]] = json.load(file)

    for chapter_index, chapter in enumerate(dataset):
        for chunk_index, chunk in enumerate(chapter["content"]):
            if chunk[-1] == "\n":
                newline = "newline"
            else:
                newline = "None"

            vllm_add_request(
                input_text=chunk.strip("\n"),
                request_id=f"{chapter_index}:{chunk_index}#{newline}"
            )

    while True:
        request = llm_engine.step()

        for output in request:
            if output.finished:
                data_index, newline = output.request_id.split("#")
                chapter_index, chunk_index = data_index.split(":")

                if newline == "newline":
                    add_newline = "\n"
                else:
                    add_newline = ""

                dataset[int(chapter_index)]["content"][int(chunk_index)] = {
                    "jp": dataset[int(chapter_index)]["content"][int(chunk_index)],
                    "translation": output.outputs[0].text + add_newline
                }

        if not llm_engine.has_unfinished_requests():
            break

    with open(f"{SAVE_DIR_PATH}/{novel_file}", 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)