from transformers import FlaxAutoModel, AutoTokenizer

class HuggingfaceEmbeddingModel:
    def __init__(self, hf_model_path_id):
        self.model = FlaxAutoModel.from_pretrained(hf_model_path_id)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path_id)

    def last_hidden_state(self, input_ids):

        outputs = self.model(input_ids)
        last_hidden_states = outputs['last_hidden_state']
        return last_hidden_states

    def embed_query(self, query):

        input_ids = self.tokenizer(query, padding=True, truncation=True, return_tensors='jax').input_ids
        outputs = self.last_hidden_state(input_ids)

        return outputs

    def tokenize(self, query):
        return self.tokenizer.tokenizer(query, padding=True, truncation=True, return_tensors='jax').input_ids

