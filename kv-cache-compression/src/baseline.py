# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")