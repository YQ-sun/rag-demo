from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

bi_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def rerank_responses(user_query: str, candidate_responses: List[str], method:str='embedding') -> List[Tuple[str, float]]:
    """
    Docstring for rerank_responses
    
    :param user_query: Description
    :type user_query: str
    :param candidate_responses: Description
    :type candidate_responses: List[str]
    :param method: Description
    :type method: str
    :return: Description
    :rtype: List[Tuple[str, float]]
    """
    if not candidate_responses:
        return []
    
    if method == 'embedding':
        query_embedding = bi_model.encode(user_query, convert_to_tensor=True)
        response_embeddings = bi_model.encode(candidate_responses, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, response_embeddings)[0]
        
        ranked_responses = sorted(
            zip(candidate_responses, cosine_scores.tolist()), 
            key=lambda x: x[1], 
            reverse=True
        )
        return ranked_responses
    
    return ValueError(f"Unknown reranking method: {method}")