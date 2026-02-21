import os
import re
import json

from tqdm import tqdm
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoTokenizer


TEMPERATURE = 0.7
TOP_K =20
TOP_P = 0.8
MIN_P = 0.0
#FREQUENCY_PENALTY = 0.2

MAX_TOKENS = 8192 #6144
MAX_REQUESTS = 16
MAX_BATCHED_TOKENS = 32768
VRAM_UTILIZATION = 0.95

MODEL_NAME = "Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/LLMModel/{MODEL_NAME}"

TOKENIZER_NAME = MODEL_NAME
TOKENIZER_PATH = f"/media/ifw/GameFile/linux_cache/LLMModel/{TOKENIZER_NAME}"

NOVEL_NAME = "test"
#NOVEL_NAME = "關於我在無意間被隔壁的天使變成廢柴這件事_v01-10_alignment"
NOVEL_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/{NOVEL_NAME}"

PROPMT_PATH = "./propmt/jp_tw_llm_alignment_check_propmt.md"

SAVE_DIR_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/{NOVEL_NAME}_llm_alignment_check"


def vllm_add_request(input_text: str, request_id: str):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": input_text}
        ], 
        add_generation_prompt=True, 
        tokenize=False
    )

    llm_engine.add_request(
        request_id=request_id,
        prompt=prompt, 
        params=generation_parameters
    )


generation_parameters = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    min_p=MIN_P,
    #frequency_penalty=FREQUENCY_PENALTY,
    #seed=None
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

with open(PROPMT_PATH, "r", encoding="utf-8") as file:
    alignment_propmt = file.read()

if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

tqdm_progress = tqdm(novel_file_list)
for novel_file in tqdm_progress:
    tqdm_progress.set_description(novel_file)

    with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
        dataset: list[dict[str, str | list[dict[str, str]]]] = json.load(file)

    for chapter_index, chapter in enumerate(dataset):
        for chunk_index, chunk in enumerate(chapter["alignment"]):
            vllm_add_request(
                input_text=alignment_propmt.format(jp=chunk["jp"], tw=chunk["tw"]),
                request_id=f"{chapter_index}:{chunk_index}"
            )

    not_pass_tag = ""
    while True:
        request = llm_engine.step()

        for output in request:
            if not output.finished:
                continue

            llm_response = output.outputs[0].text
            answer = re.findall(r"<answer>\n(True|False)\n</answer>", llm_response)
            answer = answer[-1] if answer else "True_unaligned"

            if "True" in answer:
                not_pass_tag = "not_pass_"

            chapter_index, chunk_index = output.request_id.split(":")

            dataset[int(chapter_index)]["alignment"][int(chunk_index)]["llm_response"] = llm_response if answer == "True" else ""
            dataset[int(chapter_index)]["alignment"][int(chunk_index)]["unaligned"] = answer

        if not llm_engine.has_unfinished_requests():
            break

    with open(f"{SAVE_DIR_PATH}/{not_pass_tag}{novel_file}", 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)