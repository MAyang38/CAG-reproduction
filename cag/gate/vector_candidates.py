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
                 threshold = 0):

        self.vc = vc
        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self.policy = policy
        self.threshold = threshold

    def __call__(self, query : str):

        query = self.embedding_model.embed_query(query)
        query = jnp.array(query)

        # D is calculated within the VC
        # here we calculate d
        d = self.vc.query_similarities(query)

        policy = jnp.array([100 - self.policy])

        policy_output = self.vc.get_policy_output(policy)

        if d.max() >= policy_output - self.threshold:
            return True

        else:
            return False
