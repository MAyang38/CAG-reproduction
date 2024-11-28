import jax.numpy as jnp
from jax import jit
from typing import List
import jax

class VectorCandidates:
    def __init__(self, contexts,
                 questions):

        self.contexts = jnp.array(contexts)
        self.questions = jnp.array(questions)


    @staticmethod
    @jit  # Use JIT here to optimize the static method
    def _calculate_similarities(contexts : jax.Array,
                                questions : jax.Array) -> jax.Array :

        context_norms = jnp.linalg.norm(contexts, axis=2, keepdims=True)
        question_norms = jnp.linalg.norm(questions, axis=3, keepdims=True)

        dot_products = jnp.einsum('fij,fikj->fik', contexts, questions)

        cosine_similarities = dot_products / (context_norms * question_norms.squeeze(-1))

        return cosine_similarities

    @staticmethod
    @jit
    def _calculate_query_sims(contexts : jax.Array,
                              query : jax.Array) -> jax.Array:

        dot_products = jnp.sum(contexts * query, axis=-1)
        query_norm = jnp.linalg.norm(query, axis=-1, keepdims=True)
        context_norms = jnp.linalg.norm(contexts, axis=-1)
        cosine_similarities = dot_products / (query_norm * context_norms)

        return cosine_similarities


    def query_similarities(self, query : jax.Array ) -> jax.Array :

        return self._calculate_query_sims(self.contexts, query)

    def internal_similarities(self):
        return self._calculate_similarities(self.contexts, self.questions)

    @staticmethod
    @jit
    def _get_policy_output(similarities: jax.Array,
                           percentiles: jax.Array) -> jax.Array:
        return jnp.percentile(similarities, percentiles)

    def get_policy_output(self,percentiles : jax.Array) -> jax.Array :

        return self._get_policy_output(self.internal_similarities(), percentiles)

