from pydoc import text
from dotenv import load_dotenv
import os

#importing a HuggingFace Transformer model for sentiment analysis
from transformers import pipeline

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

# Retrieve the token from the .env file
token = os.getenv("HF_TOKEN")

# Initialize the HuggingFace pipeline with the correct model identifier
transformer_pipeline = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True,
    use_auth_token=token)

text = "Oh great, another mountain of work. I absolutely love spending my Friday nights debugging microservices."

transformer_scores_raw = transformer_pipeline(text)
print(type(transformer_scores_raw))
print(transformer_scores_raw)