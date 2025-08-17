import json

import torch
from tqdm import tqdm
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoTokenizer


MAX_TOKENS = 2048 #8192
TEMPERATURE = 0.1
MIN_P = 0.1

MAX_REQUESTS = 256 #128 8 32
MAX_BATCHED_TOKENS = 65536 #32768 8192 16384
VRAM_UTILIZATION = 0.95

MODEL_NAME = "Sakura-1.5B-Qwen2.5-v1.0-HF"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/LLMModel/{MODEL_NAME}"

NOVEL_NAME = "關於我在無意間被隔壁的天使變成廢柴這件事_6"
JP_TW_NOVEL_ALIGNMENT_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/jp_tw_novel_alignment_clean/{NOVEL_NAME}_clean.json"

DATA_OUTPUT_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/jp_tw_novel_translation_alignment/{NOVEL_NAME}_translation.json"


def vllm_add_request(input_text: str, request_id: str):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"},
            {"role": "user", "content": f"将下面的日文文本翻译成中文：{input_text}"}
        ], 
        add_generation_prompt=True, 
        tokenize=False
    )

    params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        min_p=MIN_P,
        #top_p=0.9,
        #top_k=50,
        #seed=None
    )
    llm_engine.add_request(
        request_id=request_id,
        prompt=prompt, 
        params=params
    )


engine_args = EngineArgs(
    model=MODEL_PATH, 
    dtype=torch.float16, 
    quantization="bitsandbytes", 
    load_format="bitsandbytes",
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

with open(JP_TW_NOVEL_ALIGNMENT_PATH, "r", encoding="utf-8") as file:
    dataset: list[dict[str, str]] = json.load(file)

for chapter_index, chapter in enumerate(dataset):
    for alignment_index, alignment in enumerate(chapter["alignment"]):
        vllm_add_request(
            input_text=alignment["jp"],
            request_id=f"{chapter_index}:{alignment_index}"
        )

with tqdm(desc="正在生成中...", total=sum([len(data["alignment"]) for data in dataset])) as tqdm_ber:
    while True:
        request = llm_engine.step()

        for output in request:
            if output.finished:
                chapter_index, alignment_index = output.request_id.split(":")
                dataset[int(chapter_index)]["alignment"][int(alignment_index)]["tw_translation"] = output.outputs[0].text
                tqdm_ber.update()

        if not llm_engine.has_unfinished_requests():
            break

with open(DATA_OUTPUT_PATH, 'w', encoding='utf-8') as file:
    json.dump(dataset, file, indent=4, ensure_ascii=False)