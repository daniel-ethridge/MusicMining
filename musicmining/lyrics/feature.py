"""Extract features from lyrics (or any text data)"""

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


def LDA(count_matrix: pd.DataFrame, n_components: int = 10, *, return_lda_object: bool = False, **kwargs):
    """
    Run Latent Dirichlet Allocation on text

    Parameters
    ----------
    count_matrix : pd.DataFrame
        A count matrix created using sklearn.feature_extraction.text.CountVectorizer
    n_components : int, optional, default 10
        The number of LDA topics
    return_lda_object : bool, optional, default False
        When True, the raw lda object will be included in the dictionary output.

    Returns
    -------
    dict
        Output of LDA. Dictionary keys are 'topic_relevance_words' and 'topic_relevance_documents'. if
        `return_lda_object` is True, it can be retrieved with the key 'lda_object'.
    """
    lda = LatentDirichletAllocation(n_components, **kwargs)
    out_arr = lda.fit_transform(count_matrix)

    return_dict = {"topic_relevance_words": lda.components_, "topic_relevance_documents": out_arr}

    if return_lda_object:
        return_dict["lda_object"] = lda

    return return_dict
