import json
import statistics

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache


THRESHOLD = 2
THRESHOLD_DECAY = 0.99

SLIDING_WINDOW_SIZE = 8192
MAX_SENTENCE_LEN = 256
MAX_CHUNK_SIZE = 768

#MODEL_NAME = "gemma-2-2b"
MODEL_NAME = "llm-jp-3.1-1.8b"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/LLMModel/{MODEL_NAME}"

DATA_PATH = "test_data/[依空まつり] サイレント・ウィッチ IX 沈黙の魔女の隠しごと.txt"
DATA_OUTPUT_PATH = "./output/[依空まつり] サイレント・ウィッチ IX 沈黙の魔女の隠しごと_perplexity_chunking.json"


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
    return_loss: bool=False
) -> tuple[float, DynamicCache]:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True if past_key_values else False).to("cuda")
        outputs = model(
            **inputs, 
            labels=inputs["input_ids"], 
            past_key_values=past_key_values, 
            use_cache=use_cache
        )
    return (
        float(outputs.loss) if return_loss else torch.exp(outputs.loss).item(),
        outputs.past_key_values if use_cache else None
    )

def perplexity_distribution(
    sentences: list[str], 
    use_cache: bool=True,
    return_loss: bool=False
):
    distributions = []
    past_key_values = None

    for sentence_index in tqdm(range(len(sentences))):
        word_count = 0

        for context_index in range(sentence_index + 1):
            if word_count + len(sentences[sentence_index - context_index]) > SLIDING_WINDOW_SIZE:
                if context_index:
                    context_index -= 1
                break

            word_count += len(sentences[sentence_index - context_index])
        
        perplexity, past_key_values = calculate_perplexity(
            sentences[sentence_index] if use_cache else "".join(sentences[sentence_index - context_index:sentence_index + 1]),
            past_key_values=past_key_values if use_cache else None,
            use_cache=use_cache,
            return_loss=return_loss
        )
        distributions.append(perplexity)

        #print(sentences[sentence_index - context_index:sentence_index + 1], 
        #      len("".join(sentences[sentence_index - context_index:sentence_index + 1])),
        #      len(sentences[sentence_index - context_index:sentence_index + 1]))

    return distributions

def re_chunking(
    chunks: list[list[tuple[str, float]]],
    median_perplexity: float, 
    threshold: float
):
    is_exceeding_limits = False
    for chunk in chunks:
        if sum([len(sentence) for sentence, _ in chunk]) > MAX_CHUNK_SIZE:
            is_exceeding_limits = True
            break
    
    if not is_exceeding_limits:
        return chunks
    
    new_chunks: list[list[tuple[str, float]]] = []
    for chunk in chunks:
        if sum([len(sentence) for sentence, _ in chunk]) <= MAX_CHUNK_SIZE:
            new_chunks.append(chunk)
            continue
        
        new_chunk: list[tuple[str, float]] = [chunk[0]]

        for sentence, perplexity in chunk[1:]:
            if perplexity > median_perplexity * threshold:
                new_chunks.append(new_chunk)
                new_chunk = []
            
            new_chunk.append((sentence, perplexity))

        if new_chunk:
            new_chunks.append(new_chunk)
    
    return re_chunking(new_chunks, median_perplexity, threshold * THRESHOLD_DECAY)

def text_chunking(
    text: str, 
    threshold: float
):
    sentences = sentence_segmentation(text)
    distribution = perplexity_distribution(sentences, use_cache=True, return_loss=True)
    proportion = total_proportion(distribution[1:])
    median_perplexity = statistics.median(proportion)

    chunks: list[list[tuple[str, float]]] = []
    new_chunk: list[tuple[str, float]] = [(sentences[0], 0)]
    for sentence, perplexity in zip(sentences[1:], proportion):
        if perplexity > median_perplexity * threshold:
            chunks.append(new_chunk)
            new_chunk = []
        
        new_chunk.append((sentence, perplexity))
    
    if new_chunk:
        chunks.append(new_chunk)

    final_chunks = re_chunking(chunks, median_perplexity, threshold * THRESHOLD_DECAY)
    final_chunks_clean = [[chunk for chunk, _ in final_chunk] for final_chunk in final_chunks]
    
    return ["".join(final_chunk) for final_chunk in final_chunks_clean]


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

    text = """シルフィ 視点
    ルディがアイシャと別れたあと
    「大事な話があるんだ、大事なね、」
    と決心のついたような顔で言った。
    ボクとロキシーとエリスは何だろう、もしかしてまた増えるのかなと思い、聞き返した。
    「大事な話？」
    ルディは
    「うんとっても大事な話なんだ。
    夕食後3人とも寝室に来てくれないか。」
    と言った。
    「うん」
    そう言いながら、ボクらは四人で夕日の中を歩いて行った。

    ルーデウス視点
    俺は、アイシャと前世のことについて話したことを思いながら、リーリャとシルフィの美味しいカレーライスに似たものと、
    「今まで作ってくれていたアイシャ姉の代わりに作る！！」と張り切っていたクリスが作った、でこぼこだが世界一おいしいクッキーを食べた。

    """

    #perplexity = calculate_perplexity(text)
    #print(perplexity)

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data: str = file.read()
        
    """
    sentences = sentence_segmentation(data)
    distribution = perplexity_distribution(sentences, use_cache=True, return_loss=True)
    proportion = total_proportion(distribution[1:])
    median = statistics.median(proportion)

    ans = [number for number in proportion if number > median * THRESHOLD]

    print(median * THRESHOLD)
    print(ans)
    print(len(ans))
    """

    data_chunks = text_chunking(data, THRESHOLD)

    with open(DATA_OUTPUT_PATH, 'w', encoding='utf-8') as file:
        json.dump(data_chunks, file, indent=4, ensure_ascii=False)