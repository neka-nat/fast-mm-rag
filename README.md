# fast-mm-rag

Fast Multimodal RAG on CPU.

This package is an extension package for [llama_index](https://github.com/run-llama/llama_index) and can be used as a plugin for `MultiModalVectorStoreIndex` of llama_index.

This package uses:

* [clip.cpp](https://github.com/monatis/clip.cpp)
* [USearch](https://github.com/unum-cloud/usearch)

By using these packages, it is possible to perform faster RAG computations on the CPU.

## Usage

```py
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index import SimpleDirectoryReader, StorageContext
from usearch.index import Index

from fast_mm_rag import ClipCppEmbedding, USearchVectorStore


usearch_index = Index(ndim=512, metric="cos")
text_store = USearchVectorStore(usearch_index=usearch_index)
image_store = USearchVectorStore(usearch_index=usearch_index)
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

documents = SimpleDirectoryReader("./data/").load_data()
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    image_vector_store=image_store,
    image_embed_model=ClipCppEmbedding,
)
```
