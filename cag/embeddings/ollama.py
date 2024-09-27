from langchain_ollama import OllamaEmbeddings

class OllamaEmbeddings:
    def __init__(self, model : str):
        self.embeddings = OllamaEmbeddings(model = model)
        self.model = model
        self.embedding_shape = None

    def get_embedding_shape(self):
        test_input = 'Test Input'
        test_embeddings = self.embeddings.embed_query(test_input)

        return (len(test_embeddings),)

    def embed_query(self, query):
        return self.embeddings.embed_query(query)

    def embed_documents(self, documents):
        return self.embeddings.embed_documents(documents)