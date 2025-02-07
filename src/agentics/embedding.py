import numpy as np
from openai import OpenAI
from typing import Union, List, Tuple

class Embedding:
    """
    A class for generating text embeddings using OpenAI's models.

    This class provides an interface to generate embeddings from text and 
    perform similarity ranking using cosine similarity.

    Args:
        model (str, optional): The model identifier to use. Defaults to "text-embedding-3-small".
        client (OpenAI, optional): OpenAI client instance. If None, a new instance is created.

    Attributes:
        client (OpenAI): The OpenAI client instance.
        model (str): The model identifier being used.
    """

    def __init__(self, model: str = "text-embedding-3-small", client: OpenAI = None):
        self.client = client or OpenAI()
        self.model = model

    def __call__(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Callable interface for generating embeddings.

        Args:
            input (Union[str, List[str]]): Text input, either a single string or a list of strings.

        Returns:
            Union[List[float], List[List[float]]]: 
                - If a single string is provided, returns a list of floats (embedding).
                - If a list of strings is provided, returns a list of embeddings.
        """
        return self.embed(input)

    def embed(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text input.

        Args:
            input (Union[str, List[str]]): Text input, either a single string or a list of strings.

        Returns:
            Union[List[float], List[List[float]]]: 
                - If a single string is provided, returns a list of floats (embedding).
                - If a list of strings is provided, returns a list of embeddings.
        """
        response = self.client.embeddings.create(input=input, model=self.model)
        if isinstance(input, str):
            return response.data[0].embedding
        return [data.embedding for data in response.data]

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute the cosine similarity between two embedding vectors.

        Args:
            a (List[float]): The first embedding vector.
            b (List[float]): The second embedding vector.

        Returns:
            float: The cosine similarity score between -1 and 1.
        """
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def rank(self, vector: List[float], vectors: List[List[float]], return_vectors: bool = False) -> List[Tuple[Union[int, List[float]], float]]:
        """
        Rank a list of vectors by similarity to a given vector using cosine similarity.

        Args:
            vector (List[float]): The reference embedding vector.
            vectors (List[List[float]]): A list of embedding vectors to compare against.
            return_vectors (bool): If True, returns the actual vectors instead of their indices.

        Returns:
            List[Tuple[Union[int, List[float]], float]]: A list of tuples where each tuple contains:
                - The index of the vector in the input list (if return_vectors=False).
                - The original embedding vector (if return_vectors=True).
                - The cosine similarity score (higher is more similar).
                The list is sorted in descending order of similarity.
        """
        vector = np.array(vector)  # (d,)
        vectors = np.array(vectors)  # (n, d)
        
        # Compute dot products between the reference vector and all vectors
        dot_products = np.dot(vectors, vector)  # (n,)

        # Compute norms (magnitudes) of vectors
        norm_vector = np.linalg.norm(vector)  # scalar
        norm_vectors = np.linalg.norm(vectors, axis=1)  # (n,)

        # Compute cosine similarities
        similarities = dot_products / (norm_vectors * norm_vector)  # (n,)

        # Sort by highest similarity
        sorted_indices = np.argsort(-similarities)  # Order descending
        
        if return_vectors:
            return [(vectors[idx].tolist(), float(similarities[idx])) for idx in sorted_indices]
        else:
            return [(int(idx), float(similarities[idx])) for idx in sorted_indices]