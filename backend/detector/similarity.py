import torch
import torch.nn.functional as F


class SimilarityEngine:

    @staticmethod
    def cosine_similarity(
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:

        # Remove unnecessary dimensions
        embedding1 = embedding1.squeeze(0)
        embedding2 = embedding2.squeeze(0)

        # Compute cosine similarity
        similarity = F.cosine_similarity(
            embedding1,
            embedding2,
            dim=0
        )

        return similarity.item()