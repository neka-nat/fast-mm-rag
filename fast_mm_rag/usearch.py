import logging
import os
from typing import Any, List, Optional, cast

import fsspec
import numpy as np
from usearch.index import Index
from fsspec.implementations.local import LocalFileSystem

from llama_index.schema import BaseNode
from llama_index.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = logging.getLogger()

DEFAULT_PERSIST_PATH = os.path.join(
    DEFAULT_PERSIST_DIR, f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
)


class USearchVectorStore(VectorStore):
    def __init__(self, usearch_index: Index) -> None:
        self._usearch_index = usearch_index

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "USearchVectorStore":
        persist_path = os.path.join(
            persist_dir,
            f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}",
        )
        # only support local storage for now
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("USearch only supports local storage for now.")
        return cls.from_persist_path(persist_path=persist_path, fs=None)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "USearchVectorStore":

        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("USearch only supports local storage for now.")

        if not os.path.exists(persist_path):
            raise ValueError(f"No existing {__name__} found at {persist_path}.")

        logger.info(f"Loading {__name__} from {persist_path}.")
        usearch_index = Index.restore(persist_path)
        return cls(usearch_index=usearch_index)

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        NOTE: in the Faiss vector store, we do not store text in Faiss.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        new_ids = []
        for node in nodes:
            text_embedding = node.get_embedding()
            text_embedding_np = np.array(text_embedding, dtype="float32")
            new_id = self._usearch_index.size
            self._usearch_index.add(new_id, text_embedding_np)
            new_ids.append(new_id)
        return [str(i) for i in new_ids]

    @property
    def client(self) -> Any:
        """Return the usearch index."""
        return self._usearch_index

    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Save to file.

        This method saves the vector store to disk.

        Args:
            persist_path (str): The save_path of the file.

        """
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("USearch only supports local storage for now.")

        dirpath = os.path.dirname(persist_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        self._usearch_index.save(persist_path)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        raise NotImplementedError("Delete not yet implemented for USearch index.")

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for USearch yet.")

        query_embedding = cast(List[float], query.query_embedding)
        query_embedding_np = np.array(query_embedding, dtype="float32")
        matches = self._usearch_index.search(
            query_embedding_np, query.similarity_top_k
        )
        indices = [m.key for m in matches]
        dists = [m.distance for m in matches]
        # if empty, then return an empty response
        if len(indices) == 0:
            return VectorStoreQueryResult(similarities=[], ids=[])

        # returned dimension is 1 x k
        node_idxs = [str(i) for i in indices[0]]

        return VectorStoreQueryResult(similarities=dists, ids=node_idxs)
