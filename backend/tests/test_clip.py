from detector.clip_encoder import CLIPEncoder

encoder = CLIPEncoder()

embedding = encoder.encode_image("test_images/apple.png")

print("Embedding Shape:")
print(embedding.shape)

print("\nEmbedding:")
print(embedding)