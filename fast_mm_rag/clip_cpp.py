import logging
from typing import Any, Literal

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    Embedding,
)
from llama_index.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.schema import ImageType


logger = logging.getLogger(__name__)

AVAILABLE_CLIP_CPP_MODELS = (
    "CLIP-ViT-B-32-laion2B-s34B-b79K",
    "CLIP-ViT-H-14-laion2B-s32B-b79K",
    "CLIP-ViT-L-14-laion2B-s32B-b82K",
    "clip-vit-base-patch32",
    "clip-vit-large-patch14",
)
DEFAULT_CLIP_CPP_MODEL = "CLIP-ViT-B-32-laion2B-s34B-b79K"


class ClipCppEmbedding(MultiModalEmbedding):
    """CLIP embedding models for encoding text and image for Multi-Modal purpose.

    This class provides an interface to generate embeddings using a model
    deployed in clip_cpp. At the initialization it requires a model name
    of clip_cpp.

    Note:
        Requires `clip_cpp` package to be available in the PYTHONPATH. It can be installed with
        `pip install clip_cpp`.
    """

    embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)

    _model: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "ClipCppEmbedding"

    def __init__(
        self,
        *,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        model_name: str = DEFAULT_CLIP_CPP_MODEL,
        float_type: Literal["fp32", "fp16"] = "fp16",
        verbosity: int = 0,
        **kwargs,
    ):
        """Initializes the ClipCppEmbedding class.

        During the initialization the `clip_cpp` package is imported.

        Args:
            embed_batch_size (int, optional): The batch size for embedding generation. Defaults to 10,
                must be > 0 and <= 100.
            model_name (str): The model name of Clip model.

        Raises:
            ImportError: If the `clip_cpp` package is not available in the PYTHONPATH.
            ValueError: If the model cannot be fetched from huggingface. or if the embed_batch_size
                is not in the range (0, 100].
        """
        if embed_batch_size <= 0:
            raise ValueError(f"Embed batch size {embed_batch_size}  must be > 0.")

        repo_id = f"mys/ggml_{model_name}"
        model_file = f"{model_name}_ggml-model-{float_type}.gguf"
        try:
            from clip_cpp import Clip
        except ImportError:
            raise ImportError("ClipCppEmbedding requires `pip install clip_cpp`.")

        super().__init__(
            embed_batch_size=embed_batch_size, model_name=model_name, **kwargs
        )
        try:
            if self.model_name not in AVAILABLE_CLIP_CPP_MODELS:
                raise ValueError(
                    f"Model name {self.model_name} is not available in clip_cpp."
                )
            self._model = Clip(
                model_path_or_repo_id=repo_id,
                model_file=model_file,
                verbosity=verbosity,
            )
        except Exception as e:
            logger.error(f"Error while loading clip_cpp model.")
            raise ValueError("Unable to fetch the requested embeddings model") from e

    # TEXT EMBEDDINGS

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return [self._model.encode_text(self._model.tokenize(text)) for text in texts]

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)

    # IMAGE EMBEDDINGS

    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return self._get_image_embedding(img_file_path)

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return self._model.load_preprocess_encode_image(img_file_path)
