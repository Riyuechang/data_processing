import os
import json

from tqdm import tqdm
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoTokenizer


MAX_TOKENS = 2048 #8192
TEMPERATURE = 0.1
MIN_P = 0.1
TOP_P = 0.3

MAX_REQUESTS = 32 #8 64 128 256 
MAX_BATCHED_TOKENS = 32768 #8192 16384 65536
VRAM_UTILIZATION = 0.95

MODEL_NAME = "Sakura-7B-Qwen2.5-v1.0-GGUF/sakura-7b-qwen2.5-v1.0-iq4xs.gguf"
#MODEL_NAME = "Sakura-GalTransl-7B-v3.7/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/LLMModel/{MODEL_NAME}"

TOKENIZER_NAME = "Sakura-1.5B-Qwen2.5-v1.0-HF"
TOKENIZER_PATH = f"/media/ifw/GameFile/linux_cache/LLMModel/{TOKENIZER_NAME}"

NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
NOVEL_PATH = f"./output/{NOVEL_NAME}"

SAVE_DIR_PATH = f"./translation/{NOVEL_NAME}"


def vllm_add_request(input_text: str, request_id: str):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"},
            #{"role": "system", "content": "你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。"},
            {"role": "user", "content": f"将下面的日文文本翻译成中文：{input_text}"}
            #{"role": "user", "content": f"参考以下术语表（可为空，格式为src->dst #备注）：\n\n根据以上术语表的对应关系和备注，结合历史剧情和上下文，将下面的文本从日文翻译成简体中文：\n{input_text}"}
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

if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

tqdm_progress = tqdm(novel_file_list)
for novel_file in tqdm_progress:
    tqdm_progress.set_description(novel_file)

    with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
        dataset: list[dict[str, str | list[str]]] = json.load(file)

    for chapter_index, chapter in enumerate(dataset):
        for chunk_index, chunk in enumerate(chapter["content"]):
            vllm_add_request(
                input_text=chunk,
                request_id=f"{chapter_index}:{chunk_index}"
            )

    while True:
        request = llm_engine.step()

        for output in request:
            if output.finished:
                chapter_index, chunk_index = output.request_id.split(":")
                dataset[int(chapter_index)]["content"][int(chunk_index)] = output.outputs[0].text

        if not llm_engine.has_unfinished_requests():
            break
    
    for chapter in dataset:
        chapter["content"] = "".join(chapter["content"])

    with open(f"{SAVE_DIR_PATH}/{novel_file}", 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)