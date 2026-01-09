from cog import BasePredictor, Input, Path, BaseModel

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...")

        self.model = Qwen3VLEmbedder(
            model_name_or_path="./models/Qwen3-VL-Embedding-8B",
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
        print(embeddings)
        return embeddings.tolist()