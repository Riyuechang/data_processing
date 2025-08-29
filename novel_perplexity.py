import os
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache


SLIDING_WINDOW_SIZE = 4096
MAX_SENTENCE_LEN = 256

#MODEL_NAME = "gemma-2-2b"
MODEL_NAME = "llm-jp-3.1-1.8b"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/LLMModel/{MODEL_NAME}"

NOVEL_NAME = "test"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
#NOVEL_NAME = "[依空まつり]_サイレント・ウィッチ_沈黙の魔女の隠しごと_第09巻_epub"
NOVEL_PATH = f"./epub_chapter_content/{NOVEL_NAME}"

SAVE_DIR_PATH = f"./novel_perplexity/{NOVEL_NAME}"


def total_proportion(numbers: list[int | float]):
    number_sum = sum(numbers)
    return [number / number_sum for number in numbers]

def pre_segmentation(
    text: str, 
    seg_symbol: list[str] = ["\n"]
):
    text_seg: list[str] = []
    new_text = text[0]

    for word in text[1:]:
        if word not in seg_symbol and new_text[-1] in seg_symbol:
            text_seg.append(new_text)
            new_text = ""

        new_text += word
    
    if new_text:
        text_seg.append(new_text)

    return text_seg

def sentence_segmentation(text: str):
    SEGMENTATION_SYMBOL_1 = ["\n"]
    SEGMENTATION_SYMBOL_2 = ["\n", "」", "』", "）", ")"]
    SEGMENTATION_SYMBOL_3 = ["\n", "。", ".", "！", "!", "？", "?", "；", ";"]
    SEGMENTATION_SYMBOL_4 = ["\n", "，", ",", "：", ":", "、"]

    sentences = pre_segmentation(text, SEGMENTATION_SYMBOL_1)

    def sentence_len_check(
        sentences: list[str], 
        seg_symbol: list[str]
    ):
        new_sentences: list[str] = []

        for sentence in sentences:
            if len(sentence) <= MAX_SENTENCE_LEN:
                new_sentences.append(sentence)
                continue

            if seg_symbol is None:
                new_sentences.extend([
                    sentence[i:i + MAX_SENTENCE_LEN] 
                    for i in range(0, len(sentence), MAX_SENTENCE_LEN)
                ])
            else:
                new_sentences.extend(pre_segmentation(sentence, seg_symbol))
        
        return new_sentences

    sentences = sentence_len_check(sentences, SEGMENTATION_SYMBOL_2)
    sentences = sentence_len_check(sentences, SEGMENTATION_SYMBOL_3)
    sentences = sentence_len_check(sentences, SEGMENTATION_SYMBOL_4)
    sentences = sentence_len_check(sentences, None)

    return sentences

def calculate_perplexity(
    text: str, 
    past_key_values: DynamicCache=None,
    use_cache: bool=True,
) -> tuple[float, float, DynamicCache]:
    with torch.no_grad():
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            add_special_tokens=True if past_key_values else False
        ).to("cuda")

        if past_key_values and past_key_values.get_seq_length() + len(inputs["input_ids"]) > SLIDING_WINDOW_SIZE:
            new_cache = DynamicCache()
            for index in range(len(past_key_values.layers)):
                new_key = past_key_values.layers[index].keys[:, :, -SLIDING_WINDOW_SIZE + len(inputs["input_ids"][0]):, :]
                new_value = past_key_values.layers[index].values[:, :, -SLIDING_WINDOW_SIZE + len(inputs["input_ids"][0]):, :]

                new_cache.update(
                    key_states=new_key, 
                    value_states=new_value, 
                    layer_idx=index
                )

            past_key_values = new_cache

        outputs = model(
            **inputs, 
            labels=inputs["input_ids"], 
            past_key_values=past_key_values, 
            use_cache=use_cache
        )
    return (
        torch.exp(outputs.loss).item(),
        float(outputs.loss),
        outputs.past_key_values if use_cache else None
    )

def perplexity_distribution(
    sentences: list[str], 
    use_cache: bool=True,
):
    distributions_perplexity: list[float] = []
    distributions_loss: list[float] = []
    past_key_values = None

    for sentence in sentences:
        perplexity, loss, past_key_values = calculate_perplexity(
            sentence,
            past_key_values=past_key_values if use_cache else None,
            use_cache=use_cache
        )
        distributions_perplexity.append(perplexity)
        distributions_loss.append(loss)

    return distributions_perplexity, distributions_loss


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )

    if not os.path.isdir(SAVE_DIR_PATH):
        os.mkdir(SAVE_DIR_PATH)

    novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

    tqdm_progress = tqdm(novel_file_list)
    for novel_file in tqdm_progress:
        tqdm_progress.set_description(novel_file)

        with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
            dataset: list[dict[str, str]] = json.load(file)

        for data in dataset:
            sentences = sentence_segmentation(data["content"])
            perplexitys, loss = perplexity_distribution(sentences, use_cache=True)
            perplexitys[0] = 0
            loss[0] = 0
            data["content"] = sentences
            data["loss"] = loss
            data["perplexity"] = perplexitys
            #data["content"] = small_chunk_merging(text_chunking([sentences], [perplexitys]))

        with open(f"{SAVE_DIR_PATH}/{novel_file}", 'w', encoding='utf-8') as file:
            json.dump(dataset, file, indent=4, ensure_ascii=False)