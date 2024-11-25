import jax.numpy as jnp
from cag.vector_candidates.vc import VectorCandidates
import jax
from jax import jit
import jax.numpy as jnp

class VectorCandidatesGate:

    def __init__(self, vc : VectorCandidates,
                 embedding_model,
                 embeddings = None,
                 policy = 95 ,
                 threshold = 0,
                 similarity = 'cosine'):

        self.vc = vc
        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self.policy = policy
        self.threshold = threshold
        self.similarity = similarity

    def __call__(self, query : str):
        D = self.vc.internal_similarities()
        d = self.vc.query_similarities(query)

        policy = 100 - self.policy


