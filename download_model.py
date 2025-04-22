from transformers import AutoModel, AutoTokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
AutoModel.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)