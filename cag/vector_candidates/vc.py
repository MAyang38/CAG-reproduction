import jax.numpy as jnp

class VectorCandidates:
    def __init__(self, contexts, questions):
        self.contexts = contexts
        self.questions = questions

    def internal_similarities(self):
        context_norms = jnp.linalg.norm(self.contexts, axis=2, keepdims=True)
        question_norms = jnp.linalg.norm(self.questions, axis=3, keepdims=True)

        dot_products = jnp.einsum('fij,fikj->fik', self.contexts, self.questions)

        cosine_similarities = dot_products / (context_norms * question_norms.squeeze(-1))

        return cosine_similarities

    def internal_euclidean(self):

        # Expand contexts to shape (N, 1, D)
        contexts_expanded = jnp.expand_dims(self.contexts, axis=1)  # (N, 1, D)

        # Compute the Euclidean distances
        distances = jnp.sqrt(jnp.sum((contexts_expanded - self.questions) ** 2, axis=-1))  # (N, 3)

        return distances
