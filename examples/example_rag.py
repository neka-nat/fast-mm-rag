from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index import SimpleDirectoryReader, StorageContext
from usearch.index import Index

from fast_mm_rag import ClipCppEmbedding, USearchVectorStore

from PIL import Image
import matplotlib.pyplot as plt
import os


def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break


client = Index(ndim=512, metric="cos")
text_store = USearchVectorStore(client=client)
image_store = USearchVectorStore(client=client)
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

retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)
retrieval_results = retriever_engine.retrieve("cat")


from llama_index.response.notebook_utils import display_source_node
from llama_index.schema import ImageNode

retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

plot_images(retrieved_image)
