from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv(override=True)

MODEL = "openai/gpt-5.4-nano"

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 200
wait = wait_exponential(multiplier=1, min=10, max=240)


WORKERS = 5

openai = OpenAI()


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""

    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents


# def make_prompt(document):
#     how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 3
#     return f"""
# You take a document and you split the document into overlapping chunks for a KnowledgeBase.

# The document is from the shared drive of a company called Insurellm.
# The document is of type: {document["type"]}
# The document has been retrieved from: {document["source"]}

# A chatbot will use these chunks to answer questions about the company.
# You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
# This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
# There should be overlap between the chunks as appropriate; typically about 10% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

# For each chunk, you should provide a headline, a summary, and the original text of the chunk.
# Together your chunks should represent the entire document with overlap.

# Here is the document:

# {document["text"]}

# Respond with the chunks.
# """

def get_embeddings(text_list):
    """Batch retrieve embeddings from OpenAI"""
    if not text_list:
        return []
    # Using your defined embedding_model variable
    res = openai.embeddings.create(model=embedding_model, input=text_list).data
    return [e.embedding for e in res]

def semantic_chunker(text, percentile_threshold=85, window_size=3, min_chunk_words=50):
    """
    Advanced semantic chunker using a sliding window to smooth out transitions.
    """
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    if len(sentences) < window_size * 2:
        return [text]

    sentence_embeddings = get_embeddings(sentences)
    
    # 1. Create smoothed "window" embeddings
    # We average the embeddings of 'window_size' sentences to represent a 'topic'
    window_embeddings = []
    for i in range(len(sentence_embeddings) - window_size + 1):
        window = sentence_embeddings[i : i + window_size]
        window_embeddings.append(np.mean(window, axis=0))

    # 2. Calculate distances between consecutive windows
    distances = []
    for i in range(len(window_embeddings) - 1):
        sim = cosine_similarity([window_embeddings[i]], [window_embeddings[i+1]])[0][0]
        distances.append(1 - sim)

    # 3. Determine breakpoint threshold
    # 85th percentile is usually the 'sweet spot' for technical docs
    breakpoint_dist = np.percentile(distances, percentile_threshold)
    
    chunks = []
    current_chunk_indices = list(range(window_size)) # Start with the first window
    
    for i, dist in enumerate(distances):
        # Index of the sentence that would start the next chunk
        next_sentence_idx = i + window_size 
        
        # Break if distance is high AND current chunk isn't too small
        current_text_len = len(". ".join([sentences[j] for j in current_chunk_indices]).split())
        
        if dist >= breakpoint_dist and current_text_len >= min_chunk_words:
            chunks.append(". ".join([sentences[j] for j in current_chunk_indices]) + ".")
            current_chunk_indices = [next_sentence_idx]
        else:
            if next_sentence_idx < len(sentences):
                current_chunk_indices.append(next_sentence_idx)
            
    # Cleanup final chunk
    if current_chunk_indices:
        chunks.append(". ".join([sentences[j] for j in current_chunk_indices]) + ".")
        
    return chunks

@retry(wait=wait)
def process_document(document):
    """
    1. Breaks doc into semantic chunks mathematically.
    2. Uses LLM to 'Enrich' each chunk.
    """
    # Step 1: Mathematical Split
    raw_texts = semantic_chunker(document["text"], percentile_threshold=95)
    
    processed_chunks = []
    
    for raw_text in raw_texts:
        if len(raw_text.split()) < 15: # Skip tiny fragments
            continue

        enrichment_prompt = f"""
        Analyze this document chunk from {document['source']}.
        Provide a headline and a 1-2 sentence summary.
        
        TEXT:
        {raw_text}
        """
        
        try:
            # Note: Using response_format=Chunks because the LLM 
            # outputs a list based on your class definition
            response = completion(
                model=MODEL, 
                messages=[{"role": "user", "content": enrichment_prompt}],
                response_format=Chunks 
            )
            
            # 1. Validate as the plural 'Chunks' class
            raw_reply = response.choices[0].message.content
            validated_data = Chunks.model_validate_json(raw_reply)
            
            # 2. Extract the first chunk from the list (since we sent 1 raw_text)
            if validated_data.chunks:
                llm_chunk = validated_data.chunks[0]
                
                # 3. Re-assemble using our 'raw_text' to ensure no text was lost
                final_chunk = Chunk(
                    headline=llm_chunk.headline,
                    summary=llm_chunk.summary,
                    original_text=raw_text 
                )
                processed_chunks.append(final_chunk.as_result(document))
                
        except Exception as e:
            print(f"Error processing chunk: {e}")
            
    return processed_chunks

# def make_messages(document):
#     return [
#         {"role": "user", "content": make_prompt(document)},
#     ]


# @retry(wait=wait)
# def process_document(document):
#     messages = make_messages(document)
#     response = completion(model=MODEL, messages=messages, response_format=Chunks)
#     reply = response.choices[0].message.content
#     doc_as_chunks = Chunks.model_validate_json(reply).chunks
#     return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


# def create_embeddings(chunks):
#     chroma = PersistentClient(path=DB_NAME)
#     if collection_name in [c.name for c in chroma.list_collections()]:
#         chroma.delete_collection(collection_name)

#     texts = [chunk.page_content for chunk in chunks]
#     emb = openai.embeddings.create(model=embedding_model, input=texts).data
#     vectors = [e.embedding for e in emb]

#     collection = chroma.get_or_create_collection(collection_name)

#     ids = [str(i) for i in range(len(chunks))]
#     metas = [chunk.metadata for chunk in chunks]

#     collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
#     print(f"Vectorstore created with {collection.count()} documents")

def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    # Note: For large datasets, batch these into groups of 100
    texts = [chunk.page_content for chunk in chunks]
    vectors = get_embeddings(texts) 

    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
