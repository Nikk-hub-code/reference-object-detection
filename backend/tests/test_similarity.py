from detector.clip_encoder import CLIPEncoder
from detector.similarity import SimilarityEngine


def main():

    # Initialize CLIP encoder
    encoder = CLIPEncoder()

    # Encode reference image
    embedding1 = encoder.encode_image(
        "test_images/apple.png"
    )

    # Encode second image
    embedding2 = encoder.encode_image(
        "test_images/fruits.png"
    )

    # Compute similarity
    similarity = SimilarityEngine.cosine_similarity(
        embedding1,
        embedding2
    )

    print("\nSimilarity Score:")
    print(similarity)


if __name__ == "__main__":
    main()