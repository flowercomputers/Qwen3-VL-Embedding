from cog import BasePredictor, Input, Path, BaseModel
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
import torch
import time

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...")

        self.model = Qwen3VLEmbedder(
            model_name_or_path="./models/Qwen3-VL-Embedding-2B",
            dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2"
        )

        print("Model loaded successfully!")

    def predict(
            self,
            text: str = Input(
                description="a string to embed",
                default="A woman playing with her dog on a beach at sunset.",
            )
    ) -> list[float]:
        """Run a single prediction on the model"""
        embedStartTime = time.time()
        embeddings = self.model.process([{"text": text}])
        embedEndTime = time.time()
        print("Embedding time:", embedEndTime - embedStartTime)
        embedding = embeddings[0]  # take the single embedding from the batch
        print("Embedding length:", len(embedding))
        print("Embedding shape:", embedding.shape)
        print("Embedding:", embedding)

        return embedding.tolist()