from cog import BasePredictor, Input, Path, BaseModel
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...")

        self.model = Qwen3VLEmbedder(
            model_name_or_path="./models/Qwen3-VL-Embedding-4B",
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2"
        )

        print("Model loaded successfully!")

    def predict(
            self,
            text: str = Input(
                description="Image",
                default="A woman playing with her dog on a beach at sunset.",
            )
    ) -> list[float]:
        """Run a single prediction on the model"""
        embeddings = self.model.process([{"text": text}])
        embedding = embeddings[0]  # take the single embedding from the batch
        print("Embedding length:", len(embedding))
        print("Embedding shape:", embedding.shape)
        print("Embedding:", embedding)

        return embedding.tolist()