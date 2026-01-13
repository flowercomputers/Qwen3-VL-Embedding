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
            inputs: List[str] = Input(
                description="a list of strings, can be text, image URLs, or video URLs",
                default=["A woman playing with her dog on a beach at sunset."],
            ),
            types: List[str] | None = Input(
                description="a list of types, can be 'text', 'image', or 'video'",
                default=None,
            )
    ) -> list[list[float]]:
        formatted_inputs = []
        if types is None:
            types = ["text"] * len(inputs)
        if len(types) != len(inputs):
            raise ValueError("The number of types must be the same as the number of inputs")
        for input, type in zip(inputs, types):
            if type == "text":
                formatted_inputs.append({"text": input})
            elif type == "image":
                formatted_inputs.append({"image": input})
            elif type == "video":
                formatted_inputs.append({"video": input})
            else:
                raise ValueError(f"Invalid input type: {type}")
        
        print(formatted_inputs)
        return self.model.process(formatted_inputs).tolist()