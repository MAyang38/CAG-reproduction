import jax.numpy as jnp
from jax import jit
class VectorCandidates:
    def __init__(self, contexts, questions):
        self.contexts = contexts
        self.questions = questions


    @staticmethod
    @jit  # Use JIT here to optimize the static method
    def _calculate_similarities(contexts, questions):
        context_norms = jnp.linalg.norm(contexts, axis=2, keepdims=True)
        question_norms = jnp.linalg.norm(questions, axis=3, keepdims=True)

        dot_products = jnp.einsum('fij,fikj->fik', contexts, questions)

        cosine_similarities = dot_products / (context_norms * question_norms.squeeze(-1))

        return cosine_similarities

    @staticmethod
    @jit
    def _calculate_query_sims(contexts, query):

        dot_products = jnp.sum(contexts * query, axis=-1)
        query_norm = jnp.linalg.norm(query, axis=-1, keepdims=True)  # Shape (1, 1)
        context_norms = jnp.linalg.norm(contexts, axis=-1)  # Shape (1, 100)
        cosine_similarities = dot_products / (query_norm * context_norms)

        return cosine_similarities


    def query_similarities(self, query):
        return self._calculate_query_sims(self.contexts, query)

    def internal_similarities(self):
        return self._calculate_similarities(self.contexts, self.questions)

