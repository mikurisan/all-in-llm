from sentence_transformers import SentenceTransformer

model_dir = "/opt/repo/all-in-llm/cache"

model = SentenceTransformer(model_dir, device="cpu")

texts = ["你好", "今天天气不错"]
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

print("embeddings shape:", embeddings.shape)
print("first vector head:", embeddings[0][:5])
