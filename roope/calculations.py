import pickle
from pathlib import Path
import string
import nltk
from nltk.corpus import reuters, nps_chat, stopwords, wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

"""
Reuters:
    List categories        : reuters.categories()
    Get doc IDs in category: reuters.fileids(category)
    Get text               : reuters.raw(fileid)
    Get tokens             : reuters.words(fileid)
"""
"""
Nps_chat:
    # Using chatrooms and their corresponding id as categories
    List categories        : nps_chat.fileids()
    # Assuming each post as doc, each post is list of tokens/words
    Get docs in category   : nps_chat.posts(fileid)
    Get text               : " ".join(nps_chat.posts(fileid)[0])
    Get tokens             : nps_chat.posts(fileid)[0]
    Get posts with metadata: nps_chat.xml_posts(fileid)
"""

def pickle_read(filename: str | Path):
    """
    Read files written in pickle binary format.
    """
    try:
        path = Path(filename).expanduser().resolve()
        with open(path, "rb") as obj:
            return pickle.load(obj)
    except Exception:
        return None

def pickle_write(filename: str | Path, data) -> None:
    """
    Write data into a file in pickle binary format.
    """
    path = Path(filename).expanduser().resolve()
    # Ensure parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(data, file)

def all_paths_exist(paths: list[str | Path]) -> bool:
    """
    Check if all paths/files exist
    """
    return all(Path(p).expanduser().resolve().exists() for p in paths)

def categorize_reuters() -> dict:
    """
    Make a dictionary of each category in reuters with structure:

    dict[category] = {file_id: tokens}
    """
    category_docs = {}
    categories = reuters.categories()
    for category in categories:
        file_ids = reuters.fileids(categories=category)
        docs = {}
        for fid in file_ids:
            tokens = reuters.words(fid)
            processed_tokens = preprocess_tokens(tokens)
            if processed_tokens:
                docs[fid] = processed_tokens
        category_docs[category] = docs
    return category_docs

def categorize_nps_chat() -> dict:
    """
    Make a dictionary of each category in nps_chat with structure:

    dict[category] = {post_id: tokens}

    For nps_chat assume each chatroom is its own category i.e. category = fileid.
    Also post_id is just generated id, nps_chat doesn't have ids for post.
    """
    category_docs = {}
    categories = nps_chat.fileids()
    for category in categories:
        xml_posts = nps_chat.xml_posts(category)
        docs = {}
        for id, post in enumerate(xml_posts):
            post_id = f"{id}"
            tokens = word_tokenize(post.text)
            processed_tokens = preprocess_tokens(tokens)
            if processed_tokens:
                docs[post_id] = processed_tokens
        category_docs[category] = docs
    return category_docs

def preprocess_tokens(tokens: list[str]) -> list[str]:
    """
    Each token into lowercase and remove tokens that are punctuation or digits
    """
    return [t.lower() for t in tokens if t not in string.punctuation and not t.isdigit()]

def proportions_of_stopwords_by_category(categories: dict) -> dict:
    category_prop_stopword = {}
    for category, docs in categories.items():
        tokens_by_category = [t for tokens in docs.values() for t in tokens]
        amount_stopwords = sum(1 for t in tokens_by_category if t in EN_STOPWORDS)
        category_prop_stopword[category] = amount_stopwords / len(tokens_by_category)
    return category_prop_stopword

def get_wordnet_pos(tag: str):
    """
    Convert Treebank POS tags to WordNet POS tags for lemmatization
    """
    if tag.startswith("J"):  
        return wordnet.ADJ
    elif tag.startswith("V"):  
        return wordnet.VERB
    elif tag.startswith("N"):  
        return wordnet.NOUN
    elif tag.startswith("R"):  
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    Using WordNetLemmatizer and POS tagging, lang = "eng"
    """
    lemmatizer = WordNetLemmatizer()
    tokens_pos = pos_tag(tokens, lang="eng")
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in tokens_pos
    ]
    return lemmatized_tokens

def get_sorted_vocabulary(docs: list[list[str]]) -> list[str]:
    vocabulary = {t for doc in docs for t in doc}
    return sorted(vocabulary)

def tf(term: str, doc: list[str]):
    """
    tf, but assume doc is non-empty
    """
    return sum(1 for dterm in doc if dterm == term) / len(doc)

def idf(term: str, docs: list[list[str]]) -> float:
    """
    idf, but if the term was not found in any of the documents then idf = 0.0.

    If idf represents discriminotary power of a term then if that term was not found in any
    documents then it would make sense that the term doesn't discriminate.
    """
    df = sum(1 for doc in docs if term in set(doc))
    if df == 0:
        return 0.0
    return np.log10(len(docs) / df)

def tf_idf_matrix(docs: list[list[str]], vocabulary: list[str]):
    matrix = np.zeros((len(docs), len(vocabulary)), dtype=np.float64)
    idfs = [] # idf is same for each column
    for term in vocabulary:
        idfs.append(idf(term, docs))
    
    for ri, doc in enumerate(docs):
        for ci, term in enumerate(vocabulary):
            matrix[ri, ci] = tf(term, doc) * idfs[ci]
    
    return matrix

def get_vocabulary_tf_idf_matrices_by_category(categories: dict) -> tuple[dict, dict]:
    """
    Compute TF-IDF matrix for each category.
    Rows are documents and columns are terms in vocabulary.
    Documents are sorted by doc_id (file_id or post_id) in alphabetical order.
    Vocabulary is created from lemmatized tokens of each category and is sorted in alphabetical order.
    
    Return:
        (category_matrix, category_vocabulary)
    """
    category_matrix = {}
    category_vocabulary = {}
    for category, docs in categories.items():
        # Sort using doc_id
        sorted_documents = [doc for doc_id, doc in sorted(docs.items(), key=lambda item: item[0])]
        # Lemmatize each document
        for i, doc in enumerate(sorted_documents):
            sorted_documents[i] = lemmatize_tokens(doc)
        category_vocabulary[category] = get_sorted_vocabulary(sorted_documents)
        print(f"Sanity: {category}")
        category_matrix[category] = tf_idf_matrix(sorted_documents, category_vocabulary[category])
    
    return (category_matrix, category_vocabulary)

def get_tf_idf_matrices_by_category(categories: dict, vocabulary: list[str]) -> dict:
    category_matrix = {}
    for category, docs in categories.items():
        # Sort using doc_id
        sorted_documents = [doc for doc_id, doc in sorted(docs.items(), key=lambda item: item[0])]
        print(f"Sanity: {category}")
        category_matrix[category] = tf_idf_matrix(sorted_documents, vocabulary)

    return category_matrix

def step_0(reuters_path: str, nps_chat_path: str):
    """
    Create relevant data structures for reuters and nps_chat and save them into pkl files
    
    dict[category] = {doc_id: tokens, ...}
    """
    pickle_write(reuters_path, categorize_reuters())
    pickle_write(nps_chat_path, categorize_nps_chat())

def step_1(
        reuters_categories: dict,
        nps_categories: dict,
        reuters_result_path: str,
        nps_result_path: str
    ):
    """
    Calculate proportions of NTLK English stopwords for each category for reuters and nps_chat.

    dict[category] = (amount of stopwords in category) / (amount of tokens in category)
    """
    pickle_write(
        reuters_result_path,
        proportions_of_stopwords_by_category(reuters_categories)
    )
    pickle_write(
        nps_result_path,
        proportions_of_stopwords_by_category(nps_categories)
    )

def step_2(
        reuters_categories: dict,
        nps_categories: dict,
        reuters_matrices_path: str,
        nps_matrices_path: str,
    ):
    """
    Compute TF-IDF matrix for each category.
    Rows are documents and columns are terms in vocabulary.
    Documents are sorted by doc_id (file_id or post_id) in alphabetical order.
    Vocabulary is created from lemmatized tokens of each category and is sorted in alphabetical order.

    Compute TF-IDF matrix using NLTK English stopwords as vocabulary for each category.
    Then for those stopwords compute the avg, std, max, and min TF-IDF values over all categories

    These things are done for both Reuters and nps_chat
    """
    matrix_file_name = "category_matrix.pkl"
    vocabulary_file_name = "category_vocabulary.pkl"
    (category_matrix, category_vocabulary) = get_vocabulary_tf_idf_matrices_by_category(reuters_categories)
    # Save results for reuters
    pickle_write(reuters_matrices_path + matrix_file_name, category_matrix)
    pickle_write(reuters_matrices_path + vocabulary_file_name, category_vocabulary)
    category_matrix = {}
    category_vocabulary = {}
    (category_matrix, category_vocabulary) = get_vocabulary_tf_idf_matrices_by_category(nps_categories)
    # Save results for nps_chat
    pickle_write(nps_matrices_path + matrix_file_name, category_matrix)
    pickle_write(nps_matrices_path + vocabulary_file_name, category_vocabulary)
    category_matrix = {}
    category_vocabulary = {}
    
    # Computing TF-IDF matrices, but using NLTK English stopwords as vocabulary instead.
    # Also avg, std, max, and min for TF-IDF for stopword over categories
    matrix_file_name = "stopword_category_matrix.pkl"
    stopword_vocabulary = get_sorted_vocabulary([EN_STOPWORDS])
    stopword_info = {}

    # Reuters
    category_matrix = get_tf_idf_matrices_by_category(reuters_categories, stopword_vocabulary)
    # Save matrices by category for stopwords
    pickle_write(reuters_matrices_path + matrix_file_name, category_matrix)

    # Stack all documents into one big matrix
    vstack_matrices = np.vstack([matrix for matrix in category_matrix.values()])
    category_matrix = {}
    # Columnwise i.e. stopword wise
    avgs = np.mean(vstack_matrices, axis=0)
    stds = np.std(vstack_matrices, axis=0)
    maxs = np.max(vstack_matrices, axis=0)
    mins = np.min(vstack_matrices, axis=0)
    for col, word in enumerate(stopword_vocabulary):
        stopword_info[word] = [avgs[col], stds[col], maxs[col], mins[col]]
    pickle_write(reuters_matrices_path + "stopword_info.pkl", stopword_info)
    print("Step 2, Reuters has been computed")
    stopword_info = {}

    # nps_chat
    category_matrix = get_tf_idf_matrices_by_category(nps_categories, stopword_vocabulary)
    # Save matrices by category for stopwords
    pickle_write(nps_matrices_path + matrix_file_name, category_matrix)
    
    vstack_matrices = np.vstack([matrix for matrix in category_matrix.values()])
    category_matrix = {}
    avgs = np.mean(vstack_matrices, axis=0)
    stds = np.std(vstack_matrices, axis=0)
    maxs = np.max(vstack_matrices, axis=0)
    mins = np.min(vstack_matrices, axis=0)
    for col, word in enumerate(stopword_vocabulary):
        stopword_info[word] = [avgs[col], stds[col], maxs[col], mins[col]]
    pickle_write(nps_matrices_path + "stopword_info.pkl", stopword_info)
    print("Step 2, nps has been computed")




EN_STOPWORDS = None
def main():
    """
    Save computation results to pickle (.pkl) files
    """
    global EN_STOPWORDS
    # Reuters-21578, Distribution 1.0
    nltk.download("reuters", quiet=True)    # dataset
    nltk.download("nps_chat", quiet=True)   # dataset
    nltk.download("stopwords", quiet=True)  # dataset
    nltk.download("wordnet", quiet=True)    # needed for pos tagging and lemmatization
    nltk.download("punkt_tab", quiet=True)  # needed for nltk word_tokenize()
    EN_STOPWORDS = set(stopwords.words("english")) # Has 198 stopwords

    # Save and create relevant data structures for reuters and nps_chat
    reuters_path = "./reuters.pkl"
    nps_chat_path = "./nps_chat.pkl"
    if not all_paths_exist([reuters_path, nps_chat_path]):
        print("Step 0, create relevant data structures\n")
        step_0(reuters_path, nps_chat_path)

    # Rest of the code will work with those structures
    reuters_categories: dict = pickle_read(reuters_path)
    nps_categories: dict = pickle_read(nps_chat_path)

    # Step 1
    reuters_data_path = "./data/reuters/"
    nps_data_path = "./data/nps/"
    reuters_prop_stopwords_path = reuters_data_path + "reuters_prop_stopwords.pkl"
    nps_prop_stopwords_path = nps_data_path + "nps_prop_stopwords.pkl"
    if not all_paths_exist([reuters_prop_stopwords_path, nps_prop_stopwords_path]):
        print("Step 1, proportions of each NLTK English stopwords in each category\n")
        step_1(
            reuters_categories,
            nps_categories,
            reuters_prop_stopwords_path,
            nps_prop_stopwords_path
        )

    # Step 2
    reuters_matrices_path = reuters_data_path + "matrices/"
    nps_matrices_path = nps_data_path + "matrices/"
    if not all_paths_exist([reuters_matrices_path, nps_matrices_path]):
        print("Step 2, bunch of TF-IDF stuff\n")
        step_2(
            reuters_categories, 
            nps_categories,
            reuters_matrices_path,
            nps_matrices_path
        )


    print("All calculations have been done or relevant files already exist")




if __name__ == "__main__":
    main()