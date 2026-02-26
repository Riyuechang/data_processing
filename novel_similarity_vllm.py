import os
import json

import torch
from torch.nn.functional import cosine_similarity

from vllm import LLM
from tqdm import tqdm
from transformers import AutoTokenizer


VLLM_PROGRESS_BAR = False

MAX_SENTENCE_LEN = 256
SLIDING_WINDOW_SIZE = 512
BACKWARD_WINDOW_SIZE = 64
MAX_TOKENS = SLIDING_WINDOW_SIZE + 2 

MAX_REQUESTS = 32 #32 96 256
MAX_BATCHED_TOKENS = MAX_REQUESTS * MAX_TOKENS #8192 16384 32768 65536 131072
VRAM_UTILIZATION = 0.9

#MODEL_NAME = "jina-embeddings-v3"
#MODEL_NAME = "Qwen3-Embedding-0.6B"
MODEL_NAME = "Qwen3-Embedding-8B"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/embedding_model/{MODEL_NAME}"

NOVEL_NAME = "test"
#NOVEL_NAME = "[北山結莉] 精霊幻想記 第27巻 ep"
#NOVEL_NAME = "Otonari_no_Tenshisama_ni_Itsu_v01-10_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
#NOVEL_NAME = "[依空まつり]_サイレント・ウィッチ_沈黙の魔女の隠しごと_第09巻_epub"
NOVEL_PATH = f"./epub_chapter_content/{NOVEL_NAME}"

SAVE_DIR_PATH = f"./novel_similarity/{NOVEL_NAME}"

SEGMENTATION_SYMBOL_1 = ["\n"]
SEGMENTATION_SYMBOL_2 = ["」", "』", "）", ")"]
SEGMENTATION_SYMBOL_3 = ["。", ".", "！", "!", "？", "?", "；", ";"]
SEGMENTATION_SYMBOL_4 = ["，", ",", "：", ":", "、"]
SEGMENTATION_SYMBOL_ALL = (
    SEGMENTATION_SYMBOL_1
    + SEGMENTATION_SYMBOL_2
    + SEGMENTATION_SYMBOL_3
    + SEGMENTATION_SYMBOL_4
)


def truncate_middle(text, max_length, ellipsis="..."):
    if len(text) <= max_length:
        return text

    remaining_len = max_length - len(ellipsis)

    front_len = remaining_len // 2
    back_len = remaining_len - front_len

    return text[:front_len] + ellipsis + text[-back_len:]

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

def text_to_tokenize(text: str) -> list[str]:
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode([input_id]) for input_id in input_ids]

def sentence_truncation_for_token(
    sentence: str, 
    max_token_count: int,
    reverse: bool = False,
    discard_truncated_char: bool = False
):
    if not max_token_count:
        return ""
    
    sentence_token: list[str] = text_to_tokenize(sentence)
    
    if reverse:
        sentence_token.reverse()

    new_sentence_token = sentence_token[-max_token_count:]

    find_the_truncated_char = False
    for token_index, token in enumerate(new_sentence_token):
        new_token = token

        if reverse:
            new_token = token[::-1]

        for char_index, char in enumerate(new_token):
            if char not in SEGMENTATION_SYMBOL_ALL:
                continue

            find_the_truncated_char = True
            new_token_index = token_index
            new_char_index = char_index
            new_token = new_token[char_index + int(discard_truncated_char):]

            if reverse:
                new_token = new_token[::-1]

            if find_the_truncated_char:
                break

        if find_the_truncated_char:
            break
    
    if find_the_truncated_char:
        new_sentence_token = new_sentence_token[new_token_index:]
        new_sentence_token[new_char_index] = new_token

    if reverse:
        new_sentence_token.reverse()

    return "".join(new_sentence_token)

def similarity_distribution(sentences: list[str]):
    sentences_token_count = [len(tokenizer(sentence, add_special_tokens=False)) for sentence in sentences]

    contexts: list[str] = []
    for sentence_index in range(1, len(sentences)):
        context = ""
        context_token_count = 0

        for context_sentence, context_sentence_token_count in reversed(list(zip(
            sentences[:sentence_index], 
            sentences_token_count[:sentence_index]
        ))):
            if context_token_count + context_sentence_token_count > SLIDING_WINDOW_SIZE:
                remaining_window_size = SLIDING_WINDOW_SIZE - context_token_count
                new_context_sentence = sentence_truncation_for_token(
                    context_sentence, 
                    remaining_window_size,
                    discard_truncated_char=True
                )
                context = new_context_sentence + context
                break

            context = context_sentence + context
            context_token_count += context_sentence_token_count
        
        contexts.append(context)

    sentences_backward = []
    for sentence_index in range(1, len(sentences)):
        backward_sentence = sentences[sentence_index]
        backward_sentence_token_count = sentences_token_count[sentence_index]

        for context_sentence, context_sentence_token_count in zip(
            sentences[sentence_index + 1:], 
            sentences_token_count[sentence_index + 1:]
        ):
            if backward_sentence_token_count + context_sentence_token_count > BACKWARD_WINDOW_SIZE:
                remaining_window_size = SLIDING_WINDOW_SIZE - backward_sentence_token_count
                new_backward_sentence = sentence_truncation_for_token(
                    context_sentence, 
                    remaining_window_size,
                    reverse=True
                )
                backward_sentence += new_backward_sentence
                break

            backward_sentence += context_sentence
            backward_sentence_token_count += context_sentence_token_count
        
        sentences_backward.append(backward_sentence)

    contexts_embeddings, sentences_embeddings = [
        torch.tensor([
            output.outputs.embedding 
            for output in model.embed(
                texts,
                use_tqdm=VLLM_PROGRESS_BAR
            )
        ])
        for texts in [contexts, sentences_backward]
    ]

    distributions: list[float] = [1]
    for context_embedding, sentence_embedding in zip(contexts_embeddings, sentences_embeddings):
        distributions.append(float(cosine_similarity(
            context_embedding.unsqueeze(0), 
            sentence_embedding.unsqueeze(0)
        )[0]))

    return distributions


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = LLM(
        model=MODEL_PATH, 

        dtype="bfloat16",
        quantization="bitsandbytes", 
        #load_format="bitsandbytes",

        gpu_memory_utilization=VRAM_UTILIZATION,
        max_model_len=MAX_TOKENS,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        max_num_seqs=MAX_REQUESTS,

        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        task="embed"
    )

    if not os.path.isdir(SAVE_DIR_PATH):
        os.mkdir(SAVE_DIR_PATH)

    novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

    tqdm_progress = tqdm(novel_file_list)
    for novel_file in tqdm_progress:
        with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
            dataset: list[dict[str, str]] = json.load(file)

        for data_index,  data in enumerate(dataset):
            tqdm_progress.set_description(f"{data_index}/{len(dataset)} {truncate_middle(novel_file, 32)}")

            sentences = sentence_segmentation(data["content"])
            similarity = similarity_distribution(sentences)
            data["content"] = sentences
            data["similarity"] = similarity

        with open(f"{SAVE_DIR_PATH}/{novel_file}", 'w', encoding='utf-8') as file:
            json.dump(dataset, file, indent=4, ensure_ascii=False)