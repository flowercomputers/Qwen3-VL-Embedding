from cog import BasePredictor, Input, Path, BaseModel
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
import torch
import time

class EmbeddingInput(BaseModel):
    type: str # either "text", "image", or "video"
    content: str # is always a string, can be a string or a URL to an image or video

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
            inputs: List[Dict] = Input(
                description="a list of embedding inputs, can be text, image, or video",
                default=[EmbeddingInput(type="text", content="A woman playing with her dog on a beach at sunset.")],
            )
    ) -> list[list[float]]:
        formatted_inputs = []
        for input in inputs:
            if input.type == "text":
                formatted_inputs.append({"text": input.content})
            elif input.type == "image":
                formatted_inputs.append({"image": input.content})
            elif input.type == "video":
                formatted_inputs.append({"video": input.content})
            else:
                raise ValueError(f"Invalid input type: {input.type}")

        return self.model.process(formatted_inputs).tolist()