import json

from cag.embeddings import SentenceTransformerEmbeddings
from cag.vector_candidates.vc import VectorCandidates
import datasets
from cag.models import ChatOllama
import time
import numpy as np
import copy
from cag.gate.vector_candidates import VectorCandidatesGate
from langchain_community.vectorstores import FAISS



def answer_relevancy(generated_answer, original_query):
    prompt = """i will give you a answer , please generate three question which we can derive from that answer.
    use this format for generation :  start the generation with "---" and end it with "---" too ; between the questions you should include "---" as well . like this format bellow : 

    ---
    Question number 1
    ---
    Question number 2 
    ---
    Question number 3
    --- 

    Here is the answer : 
    Answer : {answer}"""

    prompt = prompt.format(answer=generated_answer)

    generated_questions = llama.invoke(prompt).content
    reason = copy.deepcopy(generated_questions)

    generated_questions = [item for item in generated_questions.split('---') if len(item) > 7]

    # embed the question
    generated_questions = [embeddings_model.embed_query(question) for question in generated_questions]

    # embed the query
    original_query = embeddings_model.embed_query(original_query)

    generated_questions, original_query = np.array(generated_questions), np.array(original_query)

    # Normalize vectors
    vec1_norm = original_query / np.linalg.norm(original_query)
    vec_list_norm = generated_questions / np.linalg.norm(generated_questions, axis=1, keepdims=True)

    # Compute cosine similarity
    cosine_sim = np.dot(vec_list_norm, vec1_norm)

    return reason, np.mean(cosine_sim)


def context_relevancy(retrieved_context, original_query):
    prompt = """this is a context relevancy test. for the given context and question , extract each sentence of the context and determine if that sentence can potentially be helpful to answer the question. for every sentence , describe the relevancy of that sentence and answer in YES or NO terms which that sentence can be helpful to answer the question or not. 

    use this format : 

    Sentence  : a simple description of relevancy to the question : YES or NO

    Here is Question 
    Question : {query}

    Here is the Context :
    Context : {context}"""

    prompt = prompt.format(query=original_query, context=retrieved_context)

    output = qwen.invoke(prompt).content
    reason = copy.deepcopy(output)

    output = output.lower()

    score = output.count('yes') / (output.count('yes') + output.count('no'))

    return reason, score


def pseudo_context_generate(query):
    prompt = """for the given question, generate a simple and small passage that can answer the question.
    Here is the Question :

    Question : {question}
    """

    prompt = prompt.format(question=query)

    output = llama.invoke(prompt).content

    return output


def query_rewriting(query):
    prompt = f"""Please rewrite the query bellow for better retrieval in web search engines or retrieval augmented generation. just generate the rewrited query without any more explaination. generate only one rewrited query, only one.

    Here is the Query :
    Query : {query}
    """

    prompt = prompt.format(query=query)

    rewrited = llama.invoke(prompt).content

    return rewrited


embeddings_model = SentenceTransformerEmbeddings('sentence-transformers/all-mpnet-base-v2', device='cuda')
qwen = ChatOllama(model = 'qwen2.5', temprature = 0.001)
llama = ChatOllama(model = 'llama3.2', temprature = 0.001)

squad = datasets.load_dataset('rajpurkar/squad')
squad = squad['validation'].shuffle()
squad = squad[:100]
with open('E:\\study\\ECNU\\biye\\RAG\\CAG\\Dataset\\CRSB-Texts.json', 'r') as f:
    crsb = json.load(f)

crsb = crsb['amazon_rainforest']

contexts = crsb['contents']
questions = squad['question']


retriever = FAISS.from_texts(texts=contexts,
                             embedding= embeddings_model)

with open('E:\\study\\ECNU\\biye\\RAG\\CAG\\Dataset\\CRSB-Embeddings-MPNET.json', 'r') as f:
    crsb = json.load(f)

crsb_contexts_embeddings = crsb['amazon_rainforest']['contents']
crsb_pseudo_queries_embeddings = crsb['amazon_rainforest']['questions']

VC = VectorCandidates(contexts= [ crsb_contexts_embeddings ], questions= [ crsb_pseudo_queries_embeddings ])


gate = VectorCandidatesGate(vc= VC, embedding_model= embeddings_model)



ars = []
crs = []

for i in range(100):

    a = time.time()

    _, needs_retrieval = gate(squad['question'][i])

    if needs_retrieval:
        retrieved_context = retriever.similarity_search(query=squad['question'][i], k=1)
        _, ar = answer_relevancy(retrieved_context, squad['question'][i])
        cr = context_relevancy(retrieved_context, squad['question'][i])

    else:
        pseudo_context = pseudo_context_generate(squad['question'][i])

        _, ar = answer_relevancy(pseudo_context, squad['question'][i])
        _, cr = context_relevancy(pseudo_context, squad['question'][i])

    ars.append(ar)
    crs.append(cr)
    b = time.time()
    print(f'Question {i} processed in {b - a} seconds')
    print(f'CR : {cr} --- AR : {ar}')
ars, crs = np.array(ars), np.array(crs)

print(f'ARs mean : {np.mean(ars)}')
print(f'CRs mean : {np.mean(crs)}')



