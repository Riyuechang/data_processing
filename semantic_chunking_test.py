from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


CHUNKING_THRESHOLD = 0.8
MAX_CHUNKING_TOKEN_SIZE = 8192
#MAX_CHUNKING_SIZE = 3

#MODEL_PATH = "/media/ifw/GameFile/linux_cache/embedding_model/jina-embeddings-v3"
#MODEL_PATH = "/media/ifw/GameFile/linux_cache/embedding_model/multilingual-e5-large"
MODEL_PATH = "/media/ifw/GameFile/linux_cache/embedding_model/multilingual-e5-small"


"""def sliding_window_chunking(text_list: list[str]):
    for index in range(2, len(text_list)):

    return"""

def pre_segmentation(text: str):
    SEGMENTATION_SYMBOL = ["\n"]#["\n", "。", "！", "？", "；", ".", "!", "?", ";"]

    text_seg: list[str] = []
    new_text = text[0]

    for word in text[1:]:
        if word not in SEGMENTATION_SYMBOL and new_text[-1] in SEGMENTATION_SYMBOL:
            text_seg.append(new_text)
            new_text = ""

        new_text += word
    
    if new_text:
        text_seg.append(new_text)

    return text_seg

def text_chunking(text: str):
    text_seg = pre_segmentation(text)

    chunking_list: list[str] = []
    new_chunk: str = text_seg[0]

    for new_text in text_seg[1:]:
        new_chunk_embedding = model.encode(new_chunk)
        new_text_embedding = model.encode(new_text)

        similarity = model.similarity(new_chunk_embedding, new_text_embedding)

        if similarity >= CHUNKING_THRESHOLD:
            new_chunk += new_text
        else:
            chunking_list.append(new_chunk)
            new_chunk = new_text
    
    if new_chunk:
        chunking_list.append(new_chunk)

    return chunking_list


model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)#.bfloat16()#.half()

text1 = "世界百大入侵物種，卻變成夜市美食「藥燉土虱」？"
text2 = "和這樣厲害的美少女比鄰而居，想必有部分男生會相當羨慕，恨不得能置身同樣情境吧。"

embedding1 = model.encode(text1)
embedding2 = model.encode(text2)

#similarity = cosine_similarity([embedding1], [embedding2])#[0][0]
similarity = model.similarity(embedding1, embedding2)

print(similarity)