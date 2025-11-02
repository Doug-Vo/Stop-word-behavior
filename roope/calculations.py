import pickle
from pydoc import doc
import re
from pathlib import Path
import string
from collections import defaultdict
from turtle import down, title
import nltk
from nltk.corpus import reuters, nps_chat, stopwords, wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from gensim import downloader
#from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


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
        if len(docs) > 0:
            category_docs[category] = docs
    return category_docs

def categorize_nps_chat() -> dict:
    """
    Make a dictionary of each category in nps_chat with structure:

    dict[category] = {post_id: tokens}

    For nps_chat assume post's class is category
    Also post_id is just generated id, nps_chat doesn"t have ids for post.
    """
    
    category_docs = defaultdict(dict)
    posts = nps_chat.xml_posts()
    for id, post in enumerate(posts):
        category = post.get("class")
        post_id = f"{id}"
        tokens = word_tokenize(post.text)
        # Remove any toke containing "user" followed by digits
        tokens = [t for t in tokens if not re.search(r"(?i)user\d+", t)]
        processed_tokens = preprocess_tokens(tokens)
        if processed_tokens:
            category_docs[category][post_id] = processed_tokens


    """
    old way
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
    """
    return category_docs

def preprocess_tokens(tokens: list[str]) -> list[str]:
    """
    Each token into lowercase and remove tokens that are punctuation or digits
    """
    processed_tokens = []
    remove_chars = string.punctuation
    translator = str.maketrans("", "", remove_chars)
    for t in tokens:
        # remove punctuations
        t = t.translate(translator)
        # into lowercase and strip whitespace
        t = t.lower().strip()
        # Keep non-empty and not digit tokens
        if t and not t.isdigit():
            processed_tokens.append(t)
        
    return processed_tokens

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
    documents then it would make sense that the term doesn"t discriminate.
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

def get_stopword_candidates_step3(tf_idf: dict, vocabularies: dict, top_n: int = 100) -> list[tuple[str, float, int]]:
    """
    tf_idf is a dictionary {category: tf_idf_matrix}
    vocabularies is a dictionary {category: vocabulary}, assumes words in vocabulary are in same order
    as they are in tf_idf_matrix columns.

    return [(word, avg_score, number of categories), ...], which is sorted first by amount of documents and then avg tf-idf value
    """
    # Dictionary for each word, sum tf_idf in each category and categories the word appeared in
    word_stats = {}
    for category, matrix in tf_idf.items():
        vocabulary = vocabularies[category]
        word_sum_tf_idf = np.sum(matrix, axis=0)
        for i, word in enumerate(vocabulary):
            if word not in word_stats:
                word_stats[word] = {"scores": [], "categories": set()}
            word_stats[word]["scores"].append(word_sum_tf_idf[i])
            word_stats[word]["categories"].add(category)

    # [(word, avg_score, number_of_categories), ...]
    data = []
    for word, info in word_stats.items():
        avg_score = np.mean(info["scores"])
        ncategories = len(info["categories"])
        data.append((word, avg_score, ncategories))
    # sort the list by number of categories and avg_score in ascending order
    data.sort(key=lambda e: (-e[2], e[1]))
    return data[:top_n]

def get_stopword_candidates_step4(tf_idf: dict, vocabularies: dict, threshold: float = 0.5) -> set[str]:
    """
    tf_idf is a dictionary {category: tf_idf_matrix}
    vocabularies is a dictionary {category: vocabulary}, assumes words in vocabulary are in same order
    as they are in tf_idf_matrix columns.

    return [word, ...] such that each word has score 0.0 for atleast threshold % of categories
    """

    word_counts = {}  # counts of zero TF-IDF appearances per word
    for category, matrix in tf_idf.items():
        vocabulary = vocabularies[category]
        # Compute sums TF-IDF for each word in this category, since sum is 0 for
        # category if that word has appeared in all documents or none of them.
        word_tf_idf_sums = np.sum(matrix, axis=0)
        for i, word in enumerate(vocabulary):
            if word not in word_counts:
                word_counts[word] = 0
            if word_tf_idf_sums[i] <= 0:
                word_counts[word] += 1
    # Select words where zero-TF-IDF appears in >= threshold fraction of categories
    stopwords_set = {word for word, count in word_counts.items() if count / len(tf_idf) >= threshold}

    return stopwords_set


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

def step_3(
        reuter_TF_IDF: dict,
        reuter_vocabularies: dict,
        nps_TF_IDF: dict,
        nps_vocabularies: dict,
        reuter_result_path: str,
        nps_result_path: str
    ):
    """
    Find top 100 stopwords for reuter and nps, by ranking words by their avg tf-idf value
    and amount of categories the word appears in.

    Also calculate proportion of how many of these new stopwords also overlap with NLTK english stopword list. This overlap is not saved
    as it can be easily calculated from saved results, but it is printed.
    """
    reuter_top_100 = get_stopword_candidates_step3(reuter_TF_IDF, reuter_vocabularies, top_n=100)
    nps_top_100 = get_stopword_candidates_step3(nps_TF_IDF, nps_vocabularies, top_n=100)
    
    #{word: (avg_score, ncategories)}
    reuter_dict = {}
    for word, avg_score, ncategories in reuter_top_100:
        reuter_dict[word] = (avg_score, ncategories)
    reuter_stopwords = set(reuter_dict.keys())
    if reuter_stopwords:
        reuter_overlap = len(reuter_stopwords & EN_STOPWORDS) / len(reuter_stopwords)
        print(f"Reuter overlap with NLTK english stopwords: {reuter_overlap}")
        pickle_write(reuter_result_path + "100_stopwords.pkl", reuter_dict)
    else:
        print("Warning! Reuter had 0-stopwords in step 3. No results saved")
    print(sorted(reuter_stopwords))
    

    nps_dict = {}
    for word, avg_score, ncategories in nps_top_100:
        nps_dict[word] = (avg_score, ncategories)
    nps_stopwords = set(nps_dict.keys())
    if nps_stopwords:
        nps_overlap = len(nps_stopwords & EN_STOPWORDS) / len(nps_stopwords)
        print(f"Nps overlap with NLTK english stopwords: {nps_overlap}")
        pickle_write(nps_result_path + "100_stopwords.pkl", nps_dict)
    else:
        print("Warning! Nps had 0-stopwords in step 3. No results saved")
    print(sorted(nps_stopwords))
    
def step_4(
        reuter_TF_IDF: dict,
        reuter_vocabularies: dict,
        nps_TF_IDF: dict,
        nps_vocabularies: dict,
        reuter_result_path: str,
        nps_result_path: str,
        threshold: float = 0.15
    ):
    """
    Find top 100 stopwords for reuter and nps assuming word"s TF-IDF score is 0 at least in threshold of corpus" categories for each corpus

    Also calculate proportion of how many of these new stopwords also overlap with NLTK english stopword list. This overlap is not saved
    as it can be easily calculated from saved results, but it is printed.
    """
    reuter_stopwords = get_stopword_candidates_step4(reuter_TF_IDF, reuter_vocabularies, threshold=threshold)
    nps_stopwords = get_stopword_candidates_step4(nps_TF_IDF, nps_vocabularies, threshold=threshold)
    if reuter_stopwords:
        reuter_overlap = len(reuter_stopwords & EN_STOPWORDS) / len(reuter_stopwords)
        print(f"Reuter overlap with NLTK english stopwords: {reuter_overlap}")
        pickle_write(reuter_result_path + "stopwords.pkl", reuter_stopwords)
    else:
        print("Warning! Reuter had 0-stopwords in step 4. No results saved")
    print(sorted(reuter_stopwords))
    
    if nps_stopwords:
        nps_overlap = len(nps_stopwords & EN_STOPWORDS) / len(nps_stopwords)
        print(f"Nps overlap with NLTK english stopwords: {nps_overlap}")
        pickle_write(nps_result_path + "stopwords.pkl", nps_stopwords)
    else:
        print("Warning! Nps had 0-stopwords in step 4. No results saved")
    print(sorted(nps_stopwords))

def ndocs_containing_term(term: str, categories: dict) -> tuple[int, int]:
    """
    Calculate number of documents cotaining term from categories and number of documents not containing term.
    
    return (n_contain, n_not_contain)
    """
    n_total_docs = 0
    ndocs_contain_term = 0
    for category, docs in categories.items():
        n_total_docs += len(docs)
        df = sum(1 for _, doc in docs.items() if term in set(doc))
        ndocs_contain_term += df
    return (ndocs_contain_term, n_total_docs - ndocs_contain_term)

def step6_discriminant_charasteristic(term: str, categories: dict, not_categories: dict) -> tuple[float, float]:
    """
    Calculate discriminant and charasteristic as described in paper:
    https://pdfs.semanticscholar.org/42d5/8221ee7a0ce4ee8c8ddfe9d1b6b5fb29dd2c.pdf

    return (discriminant, charasteristic)
    """
    tp, fn = ndocs_containing_term(term, categories)
    p = tp + fn
    fp, tn = ndocs_containing_term(term, not_categories)
    n = fp + tn
    # TP+FN=P, TN+FP=N
    sensitivity = tp / p
    specificity = tn / n
    fall_out = fp / n
    discriminant = sensitivity - fall_out
    charasteristic = sensitivity - specificity
    return (discriminant, charasteristic)

def step6_list_discrimnant_charasteristic_by_category(categories: dict):
    """
    dict[category] = {term: (discriminant, charasteristic)}
    """
    category_list = defaultdict(dict)
    lemmatized_categories = defaultdict(dict)
    category_vocabulary = {}
    for category, docs in categories.items():
        # Lemmatize each document
        for doc_id, doc in docs.items():
            lemmatized_categories[category][doc_id] = lemmatize_tokens(doc)
        # Get vocabulary of documents
        category_vocabulary[category] = get_sorted_vocabulary([doc for _, doc in lemmatized_categories[category].items()])
        print(f"Sanity: {category}")

    temp_not_categories = lemmatized_categories.copy()
    for category, docs in lemmatized_categories.items():
        vocabulary = category_vocabulary[category]
        temp_categories = {category: docs}
        # Exclude this category from other categories
        deleted_docs = temp_not_categories[category]
        del temp_not_categories[category]

        for term in vocabulary:
            disc, charast = step6_discriminant_charasteristic(term, temp_categories, temp_not_categories)
            category_list[category][term] = (disc, charast)
        # Add the exluded category back
        temp_not_categories[category] = deleted_docs
        print(f"Sanity: {category}")

    return category_list

def step_6_get_avg_word_score(category_list: dict):
    """
    Calculate avg charasteristic, avg discriminant, and avg score

    word_score[word] = (avg chara, avg disc, avg score, count) 
    """
    ## chara_threshold = 0.0
    # disc_threshold = 0.3 not used with duc vo"s method
    # score = (chara - abs(disc))
    # word_score[word] = (avg_chara, avg_disc, avg_score, count),
    word_score = {}
    for category, info in category_list.items():
        for word, (disc, chara) in info.items():
            if word_score.get(word) == None:
                word_score[word] = (chara, disc, chara - abs(disc), 1)
            else:
                old_chara, old_disc, old_score, old_count = word_score[word]
                new_count = old_count + 1
                avg_chara = (old_chara*old_count + chara)/new_count
                avg_disc = (old_disc*old_count + disc)/new_count
                avg_score = (old_score*old_count + (chara - abs(disc)))/new_count
                word_score[word] = (avg_chara, avg_disc, avg_score, new_count)

    #stopwords = sorted([(word, info) for word, info in word_score.items()], key=lambda e: e[1][2], reverse=True)
    # [(word, info), ...]
    #return stopwords
    return word_score

def step_6(
        reuter_categories: dict,
        nps_categories: dict,
        reuter_result_path: str,
        nps_result_path: str
    ):
    """
    Calculate discriminant and charasteristic matrices for reuter and nps.

    Calculate n most approriate stopswords using discriminant and charateristic
    like described in paper:
    https://pdfs.semanticscholar.org/42d5/8221ee7a0ce4ee8c8ddfe9d1b6b5fb29dd2c.pdf

    """
    # calculate discriminat and charasteristic for each category
    reut_category_list = step6_list_discrimnant_charasteristic_by_category(reuter_categories)
    pickle_write(reuter_result_path + "category_list.pkl", reut_category_list)
    nps_category_list = step6_list_discrimnant_charasteristic_by_category(nps_categories)
    pickle_write(nps_result_path + "category_list.pkl", nps_category_list)

    reut_word_scores = step_6_get_avg_word_score(reut_category_list)
    if reut_word_scores:
        pickle_write(reuter_result_path + "word_scores.pkl", reut_word_scores)
        reut_stopwords = sorted([(word, info) for word, info in reut_word_scores.items()], key=lambda e: e[1][2], reverse=True)
        for n in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            top_n = set([word for (word, _) in reut_stopwords[0:n]])
            print(f"{n}. {len(top_n & EN_STOPWORDS)/len(top_n)}")
    else:
        print("Warning!! Step6 not stopwords for Reuters. Word scores not saved")

    nps_word_scores = step_6_get_avg_word_score(nps_category_list)
    if nps_word_scores:
        pickle_write(nps_result_path + "word_scores.pkl", nps_word_scores)
        nps_stopwords = sorted([(word, info) for word, info in nps_word_scores.items()], key=lambda e: e[1][2], reverse=True)
        for n in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            top_n = set([word for (word, _) in nps_stopwords[0:n]])
            print(f"{n}. {len(top_n & EN_STOPWORDS)/len(top_n)}")
    else:
        print("Warning!! Step6 not stopwords for nps. Word scores not saved")



def step_6_plot(
        word_scores: dict, result_path: str = None,
        title = "Precision at K vs. default stopword set"
    ):
    """
    Plot Precision at K vs default stopword set, save the the figure in png format to result_path
    """
    sstopwords = sorted([(word, info) for word, info in word_scores.items()], key=lambda e: e[1][2], reverse=True)
    k_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    precision_results = []
    for k in k_values:
        top_k_terms = set([word for (word, _) in sstopwords[0:k]])
        precision = len(top_k_terms & EN_STOPWORDS)/len(top_k_terms)
        precision_results.append({"k": k, "precision": precision})
    # create a new DataFrame for plotting
    precision_df = pd.DataFrame(precision_results)
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    sns.lineplot(
        data=precision_df,
        x="k",
        y="precision",
        marker="o",
        markersize=8
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Number of Top Stopwords (K)", fontsize=12)
    plt.ylabel("Precision (Overlap with NLTK english stopword set)", fontsize=12)
    plt.xticks(k_values)
    plt.ylim(0, 1.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    if result_path:
        filename = result_path+"fig_precision.png"
        path = Path(filename).expanduser().resolve()
        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_df(
        df, x_key: str, y_key: str, hue_key: str, 
        title: str="Title", legend_title="Legend Title", 
        xlab: str = None, ylab: str = None,
        result_path: str = None
    ):
    """
    Plot 2D representation from df, where x-axis has x_key, and y-axis has y_key
    """
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=x_key, 
        y=y_key, 
        data=df, 
        hue=hue_key,
        palette="viridis", 
        s=100
    )
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title=legend_title, loc="upper right")
    if result_path:
        filename = result_path+"2D_presentation.png"
        path = Path(filename).expanduser().resolve()
        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def word_score2df(word_score: dict):
    """
    Convert word_score into df and give it columns: "word", "phi", "abs delta", "stopword score", "count"
    """
    # word_score[word] = (avg_chara, avg_disc, avg_score, count)
    results = []
    for word, info in word_score.items():
        avg_chara = info[0]
        avg_disc = info[1]
        avg_score = info[2]
        count = info[3]
        results.append(
            {
                "word": word,
                "phi": avg_chara,
                "abs delta": avg_disc,
                "stopword score": avg_score,
                "count": count
            }
        )
    # create a new DataFrame
    df = pd.DataFrame(results)
    return df

def step_7(
        reuters_word_score: dict, 
        nps_word_scores: dict,
        reuter_result_path: str,
        nps_result_path: str
        ):
    #dict[word] = (avg_chara, avg_disc, avg_score, count)
    reuters_df = word_score2df(reuters_word_score)
    reuters_df = reuters_df.sort_values(by="stopword score", ascending=False)
    reuters_top50 = reuters_df.head(50).copy()
    plot_df(
        reuters_top50,
        x_key="phi",
        y_key="abs delta",
        hue_key="abs delta",
        xlab = "phi",
        ylab = "abs delta",
        title="Top 50 Stopwords (Reuters)",
        legend_title="abs delta",
        result_path=reuter_result_path
    )

    nps_df = word_score2df(nps_word_scores)
    nps_df = nps_df.sort_values(by="stopword score", ascending=False)
    nps_top50 = reuters_df.head(50).copy()
    plot_df(
        nps_top50,
        x_key="phi",
        y_key="abs delta",
        xlab = "phi",
        ylab = "abs delta",
        hue_key="abs delta",
        title="Top 50 Stopwords (Nps chat)",
        legend_title="abs delta",
        result_path=nps_result_path
    )

def plot_clusters(
        df, x_key: str, y_key: str, cluster_key: str, 
        title: str="Title", legend_title="Cluster", 
        xlab: str = None, ylab: str = None,
        result_path: str = None
    ):
    # plot
    plt.figure(figsize=(10, 8))
    for cluster_id in sorted(df[cluster_key].unique()):
        subset = df[df[cluster_key] == cluster_id]
        plt.scatter(
            subset[x_key],
            subset[y_key],
            s=100,
            label=f"Cluster {cluster_id}",
            alpha=0.8,
            edgecolors="w"
        )

    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title=legend_title, loc="upper right")
    if result_path:
        filename = result_path+"2D_cluster.png"
        path = Path(filename).expanduser().resolve()
        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def step_8(
        reuters_word_score: dict, 
        nps_word_scores: dict,
        reuter_result_path: str,
        nps_result_path: str
    ):
    TOP_N_CLUSTER = 30
    NUM_CLUSTERS = 3  
    reuters_df = word_score2df(reuters_word_score)
    reuters_df = reuters_df.sort_values(by="stopword score", ascending=False)
    reuters_top30_cluster = reuters_df.head(TOP_N_CLUSTER).copy()
    # Features for clustering
    X = reuters_top30_cluster[["abs delta", "phi"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Apply K-Means
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    reuters_top30_cluster["cluster"] = kmeans.fit_predict(X_scaled)
    plot_clusters(
        reuters_top30_cluster,
        x_key="phi",
        y_key="abs delta",
        cluster_key="cluster",
        title=f"K-Means Clustering of Top {TOP_N_CLUSTER} Stopwords (k={NUM_CLUSTERS}) (Reuters)",
        xlab="Absolute Charasteristic Capability",
        ylab="Discriminant Capability",
        legend_title="Cluster",
        result_path=reuter_result_path
    )

    nps_df = word_score2df(nps_word_scores)
    nps_df = nps_df.sort_values(by="stopword score", ascending=False)
    nps_top30_cluster = nps_df.head(TOP_N_CLUSTER).copy()
    # Features for clustering
    X = nps_top30_cluster[["abs delta", "phi"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Apply K-Means
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    nps_top30_cluster["cluster"] = kmeans.fit_predict(X_scaled)
    plot_clusters(
        nps_top30_cluster,
        x_key="phi",
        y_key="abs delta",
        cluster_key="cluster",
        title=f"K-Means Clustering of Top {TOP_N_CLUSTER} Stopwords (k={NUM_CLUSTERS}) (Nps chat)",
        xlab="Absolute Charasteristic Capability",
        ylab="Discriminant Capability",
        legend_title="Cluster",
        result_path=nps_result_path
    )

def step9_vocabulary_discrimnant_charasteristic(categories: dict, vocabulary: set[str]):
    """
    category_list[category] = {term: (disc, chara)}
    """
    #dict[category] = {term: (discriminant, charasteristic)}
    category_list = defaultdict(dict)
    temp_not_categories = categories.copy()
    for category, docs in categories.items():
        temp_categories = {category: docs}
        # Exclude this category from other categories
        deleted_docs = temp_not_categories[category]
        del temp_not_categories[category]
        for term in vocabulary:
            disc, chara = step6_discriminant_charasteristic(term, temp_categories, temp_not_categories)
            category_list[category][term] = (disc, chara)
        temp_not_categories[category] = deleted_docs
        print(f"Sanity: {category}")
    
    return category_list



def step_9(
        reuters_categories: dict,
        nps_categories: dict,
        reuters_result_path: str,
        nps_result_path: str
    ):
    reuters_category_list = step9_vocabulary_discrimnant_charasteristic(reuters_categories, EN_STOPWORDS)
    terms_reuters = step_6_get_avg_word_score(reuters_category_list)
    pickle_write(reuters_result_path+"category_list_vocabulary.pkl", reuters_category_list)
    pickle_write(reuters_result_path+"armano_vocabulary.pkl", terms_reuters)
    
    nps_category_list = step9_vocabulary_discrimnant_charasteristic(nps_categories, EN_STOPWORDS)
    terms_nps = step_6_get_avg_word_score(nps_category_list)
    pickle_write(nps_result_path+"category_list_vocabulary.pkl", nps_category_list)
    pickle_write(nps_result_path+"armano_vocabulary.pkl", terms_nps)

    NUM_CLUSTERS = 3  
    # Plot reuters and nltk stopwords
    df = word_score2df(terms_reuters)
    df = df.sort_values(by="stopword score", ascending=False)
    # Features for clustering
    X = df[["abs delta", "phi"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Apply K-Means
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    plot_clusters(
        df=df,
        x_key="phi",
        y_key="abs delta",
        cluster_key="cluster",
        title=f"K-Means Clustering of NLTK English Stopwords (k={NUM_CLUSTERS}) (Reuters)",
        xlab="Phi (φ) Score - Characteristic Capability",
        ylab="Absolute Delta (|δ|) Score - Discriminant Capability",
        legend_title="Cluster",
        result_path=reuters_result_path
    )
    # Plot nmps and nltk stopwords
    df = word_score2df(terms_nps)
    df = df.sort_values(by="stopword score", ascending=False)
    # Features for clustering
    X = df[["abs delta", "phi"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Apply K-Means
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    plot_clusters(
        df=df,
        x_key="phi",
        y_key="abs delta",
        cluster_key="cluster",
        title=f"K-Means Clustering of NLTK English Stopwords (k={NUM_CLUSTERS}) (Nps chat)",
        xlab="Phi (φ) Score - Characteristic Capability",
        ylab="Absolute Delta (|δ|) Score - Discriminant Capability",
        legend_title="Cluster",
        result_path=nps_result_path
    )

def get_cluster_summary(top_terms, model, model_name:str,  nclusters:int):
    X_w2v = []
    valid_terms = []
    print("Extracting model embeddings for terms...")
    for term in top_terms:
        if term in model:       
            X_w2v.append(model[term])
            valid_terms.append(term)
        else:
            print(f"Term not found in {model_name}: {term}")
    X_w2v = np.array(X_w2v)
    print(f"Found embeddings for {len(valid_terms)} / {len(top_terms)} words.")
    print(valid_terms)
    if len(valid_terms) == 0:
        raise ValueError("No embeddings found. Cannot proceed.")
    
    print(f"Running K-Means clustering on embeddings (k={nclusters})...")
    kmeans = KMeans(n_clusters=nclusters, random_state=12, n_init=10)
    clusters = kmeans.fit_predict(X_w2v)
    print("Clustering completed.")
    print("Running t-SNE for visualization...")
    perplexity_val_top30 = min(5, len(valid_terms) - 1)
    if perplexity_val_top30 < 1:
        perplexity_val_top30 = 1

    tsne = TSNE(
        n_components=2,
        random_state=1,
        perplexity=perplexity_val_top30,
        init="pca"
    )
    X_tsne = tsne.fit_transform(X_w2v)
    print("t-SNE completed.")
    # Combine results
    cluster_summary = pd.DataFrame({
        "word": valid_terms,
        "tsne_1": X_tsne[:, 0],
        "tsne_2": X_tsne[:, 1],
        "cluster": clusters
    })
    
    return cluster_summary

def step_10(
        reuters_word_score: dict,
        nps_word_score: dict,
        model_path: str,
        model_name: str,
        reuters_result_path: str,
        nps_result_path: str,
        nltk_result_path: str
    ):
    """
    Use pretrained Word2Vec model, K-means, t-SNE to visualize 30 stopwords form step 6
    """
    top_n=30
    nclusters=3
    reuters_df = word_score2df(reuters_word_score)
    reuters_df = reuters_df.sort_values(by="stopword score", ascending=False)
    top30_cluster = reuters_df.head(top_n).copy()
    top_terms = top30_cluster["word"].tolist()
    model = pickle_read(model_path)
    #w2v_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    reuters_cluster_summary = get_cluster_summary(
        top_terms,
        model,
        nclusters=nclusters,
    )
    pickle_write(reuters_result_path+f"{model_name}_cluster_summary.pkl", reuters_cluster_summary)
    plot_df(
        reuters_cluster_summary,
        x_key="tsne_1",
        y_key="tsne_2",
        hue_key="cluster",
        title=f"{model_name} Clusters of Top {len(reuters_cluster_summary)} Armano Stopwords (k={nclusters}) (Reuters)",
        xlab="t-SNE Dimension 1",
        ylab="t-SNE Dimension 2",
        legend_title="Cluster",
        result_path=f"{reuters_result_path}/{model_name}/"
    )

    nps_df = word_score2df(nps_word_score)
    nps_df = nps_df.sort_values(by="stopword score", ascending=False)
    top30_cluster = nps_df.head(top_n).copy()
    top_terms = top30_cluster["word"].tolist()
    nps_cluster_summary = get_cluster_summary(
        top_terms,
        model,
        nclusters=nclusters,
    )
    pickle_write(nps_result_path+f"{model_name}_cluster_summary.pkl", nps_cluster_summary)
    plot_df(
        nps_cluster_summary,
        x_key="tsne_1",
        y_key="tsne_2",
        hue_key="cluster",
        title=f"{model_name} Clusters of Top {len(nps_cluster_summary)} Armano Stopwords (k={nclusters}) (Nps chat)",
        xlab="t-SNE Dimension 1",
        ylab="t-SNE Dimension 2",
        legend_title="Cluster",
        result_path=f"{nps_result_path}/{model_name}/"
    )

    nltk_cluster_summary = get_cluster_summary(
        list(EN_STOPWORDS),
        model,
        nclusters=nclusters
    )
    pickle_write(nltk_result_path+f"{model_name}_cluster_summary.pkl", nltk_cluster_summary)
    plot_df(
        nltk_cluster_summary,
        x_key="tsne_1",
        y_key="tsne_2",
        hue_key="cluster",
        title=f"{model_name} Clusters of NLTK English Stopword list (k={nclusters})",
        xlab="t-SNE Dimension 1",
        ylab="t-SNE Dimension 2",
        legend_title="Cluster",
        result_path=f"{nltk_result_path}/{model_name}/"
    )





def category_list2df(category_list: dict):
    df = pd.DataFrame(
        [
            [term, category, disc, chara]
            for category, terms in category_list.items()
            for term, (disc, chara) in terms.items()
        ],
        columns=["word", "label", "delta", "phi"]
    )
    return df

def plotRombus(df: pd.DataFrame):
    plt.figure(figsize=(10, 10))
    sns.set_style("whitegrid")
    rhombus_x = [0, 1, 0, -1, 0]
    rhombus_y = [1, 0, -1, 0, 1]
    plt.plot(rhombus_x, rhombus_y, color="lightgray", linestyle="--", zorder=1, label="Theoretical Boundary")
    sns.scatterplot(
        data=df,
        x="phi",
        y="delta",
        hue="label",  # Color points by their target category
        alpha=0.7,
        s=50,  # Marker size
        zorder=2
    )
    plt.title(r"Term Distribution in $\phi-\delta$ Space", fontsize=16)
    plt.xlabel(r"Characteristic Capability ($\phi$)", fontsize=12)
    plt.ylabel(r"Discriminant Capability ($\delta$)", fontsize=12)
    # Add center lines
    plt.axhline(0, color="black", linewidth=0.5, zorder=1)
    plt.axvline(0, color="black", linewidth=0.5, zorder=1)
    # Ensure the plot is square
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(title="Category (C)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()



EN_STOPWORDS = None
def main():
    """
    Save computation results to pickle (.pkl) files and figures in .png files
    """
    global EN_STOPWORDS
    print("Downloading necessary files, if not loaded")
    # Reuters-21578, Distribution 1.0
    nltk.download("reuters", quiet=True)    # dataset
    nltk.download("nps_chat", quiet=True)   # dataset
    nltk.download("stopwords", quiet=True)  # dataset
    nltk.download("wordnet", quiet=True)    # needed for pos tagging and lemmatization
    nltk.download("punkt_tab", quiet=True)  # needed for nltk word_tokenize()
    EN_STOPWORDS = set(stopwords.words("english")) # Has 198 stopwords
    w2v_model_path = "./w2v_model.pkl"
    fasText_model_path = "./fasText_model_path.pkl"
    glove_model_path = "./glove.pkl"
    if not all_paths_exist([w2v_model_path]):
        w2v_model = downloader.load("word2vec-google-news-300")
        pickle_write(w2v_model_path, w2v_model)
        del w2v_model
    if not all_paths_exist([fasText_model_path]):
        ft_model = downloader.load("fasttext-wiki-news-subwords-300")
        pickle_write(fasText_model_path, ft_model)
        del ft_model
    if not all_paths_exist([glove_model_path]):
        glove_model = downloader.load("glove-twitter-100")
        pickle_write(glove_model_path, glove_model)
        del glove_model

    # Save and create relevant data structures for reuters and nps_chat
    reuters_path = "./reuters.pkl"
    nps_chat_path = "./nps_chat.pkl"
    if not all_paths_exist([reuters_path, nps_chat_path]):
        print("\nStep 0, create relevant data structures")
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
        print("Step 1, proportions of each NLTK English stopwords in each category")
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
        print("\nStep 2, bunch of TF-IDF stuff")
        step_2(
            reuters_categories, 
            nps_categories,
            reuters_matrices_path,
            nps_matrices_path
        )

    # Step 3
    reuters_stopwords_path3 = reuters_data_path + "step3_100_stopwords/"
    nps_stopwords_path3 = nps_data_path + "step3_100_stopwords/"
    if not all_paths_exist([reuters_stopwords_path3, nps_stopwords_path3]):
        print("\nStep3, calculating top 100 stopwords by considering words that achieve lowest score across categories")
        step_3(
            pickle_read(reuters_matrices_path + "category_matrix.pkl"),
            pickle_read(reuters_matrices_path + "category_vocabulary.pkl"),
            pickle_read(nps_matrices_path + "category_matrix.pkl"),
            pickle_read(nps_matrices_path + "category_vocabulary.pkl"),
            reuters_stopwords_path3,
            nps_stopwords_path3
        )

    # Step 4
    reuters_stopwords_path4 = reuters_data_path + "step4_stopwords/"
    nps_stopwords_path4 = nps_data_path + "step4_stopwords/"
    if not all_paths_exist([reuters_stopwords_path4, nps_stopwords_path4]):
        print("\nStep4, calculating stopwords that have TF-IDF score of 0.0 in at least 20% of categories in each corpus")
        step_4(
            pickle_read(reuters_matrices_path + "category_matrix.pkl"),
            pickle_read(reuters_matrices_path + "category_vocabulary.pkl"),
            pickle_read(nps_matrices_path + "category_matrix.pkl"),
            pickle_read(nps_matrices_path + "category_vocabulary.pkl"),
            reuters_stopwords_path4,
            nps_stopwords_path4,
            threshold=0.20
        )

    # Step 5, not implemented, foreign language corpus (Duc vo)

    # Step 6 Really expensive to compute, well my code is inefficient python...
    reuters_stopwords_path6 = reuters_data_path + "step6_stopwords/"
    nps_stopwords_path6 = nps_data_path + "step6_stopwords/"
    if not all_paths_exist([reuters_stopwords_path6, nps_stopwords_path6]):
        print("\nStep6, Armano et al paper's method for identifying stopwords")
        step_6(
            reuters_categories,
            nps_categories,
            reuters_stopwords_path6,
            nps_stopwords_path6
        )
        word_score_path = reuters_stopwords_path6 + "word_scores.pkl"
        if all_paths_exist([word_score_path]):
            word_scores = pickle_read(word_score_path)
            step_6_plot(
                word_scores, reuters_stopwords_path6,
                title="Precision at K vs default stopword set (Reuters)"
            )
        else:
            print(f"{word_score_path} doesn't exist, not plotting precision vs default stopword set")
        word_score_path = nps_stopwords_path6 + "word_scores.pkl"
        if all_paths_exist([word_score_path]):
            word_scores = pickle_read(word_score_path)
            step_6_plot(
                word_scores, nps_stopwords_path6,
                title="Precision at K vs default stopword set (Nps chat)"
            )
        else:
            print(f"{word_score_path} doesn't exist, not plotting precision vs default stopword set")

    # Step 7 graphics
    reuter_word_score_path = reuters_stopwords_path6 + "word_scores.pkl"
    reuters_stopwords_path7 = reuters_data_path + "step7_stopwords/"
    nps_word_score_path = nps_stopwords_path6 + "word_scores.pkl"
    nps_stopwords_path7 = nps_data_path + "step7_stopwords/"
    if all_paths_exist([reuter_word_score_path, nps_word_score_path]) and not all_paths_exist([reuters_stopwords_path7, nps_stopwords_path7]):
        print("\nStep7, 2D representaion of the 50 most approriate stopwords from step 6")
        reuters_word_scores = pickle_read(reuters_stopwords_path6 + "word_scores.pkl")
        nps_word_scores = pickle_read(nps_stopwords_path6 + "word_scores.pkl")
        step_7(
            reuters_word_scores,
            nps_word_scores,
            reuters_stopwords_path7,
            nps_stopwords_path7
        )

    # Step 8
    reuters_stopwords_path8 = reuters_data_path + "step8_stopwords/"
    nps_stopwords_path8 = nps_data_path + "step8_stopwords/"
    if all_paths_exist([reuter_word_score_path, nps_word_score_path]) and not all_paths_exist([reuters_stopwords_path8, nps_stopwords_path8]):
        print("\nStep8, k-means algorithm with 3 clusters")
        reuters_word_score = pickle_read(reuter_word_score_path)
        nps_word_score = pickle_read(nps_word_score_path)
        step_8(
            reuters_word_score,
            nps_word_score,
            reuters_stopwords_path8,
            nps_stopwords_path8
        )

    # Step 9 Repeat step 8 with Englisht NLTK stopword list
    reuters_stopwords_path9 = reuters_data_path + "step9_stopwords/"
    nps_stopwords_path9 = nps_data_path + "step9_stopwords/"
    if not all_paths_exist([reuters_stopwords_path9, nps_stopwords_path9]):
        print("\nStep9, k-means algorithm with 3 clusters for NLTK english stopwords")
        step_9(
            reuters_categories,
            nps_categories,
            reuters_stopwords_path9,
            nps_stopwords_path9
        )

    # Step 10 use k-means algorithm and use Word2Vec representaion of 30 stopwords from step 6.
    reuters_stopwords_path10 = reuters_data_path + "step10_stopwords/"
    nps_stopwords_path10 = nps_data_path + "step10_stopwords/"
    nltk_stopwords_path10 = "./data/nltk/step10_stopwords/"
    if all_paths_exist([w2v_model_path, reuter_word_score_path, nps_word_score_path]) and not all_paths_exist([reuters_stopwords_path10, nps_stopwords_path10, nltk_stopwords_path10]):
        print("\nStep10, Word2Vec, k-means, 2D presentation for stopwords from step 6 and NLTK englist stopwords")
        reuters_word_score = pickle_read(reuter_word_score_path)
        nps_word_score = pickle_read(nps_word_score_path)
        step_10(
            reuters_word_score,
            nps_word_score,
            w2v_model_path,
            "Word2Vec",
            reuters_stopwords_path10,
            nps_stopwords_path10,
            nltk_stopwords_path10
        )

    # Step 11 Repeat 10) using FasText and Glove embeddings.
    reuters_stopwords_path11 = reuters_data_path + "step11_stopwords/"
    nps_stopwords_path11 = nps_data_path + "step11_stopwords/"
    nltk_stopwords_path11 = "./data/nltk/step11_stopwords/"
    if all_paths_exist([fasText_model_path, glove_model_path]) and not all_paths_exist([reuters_stopwords_path11, nps_stopwords_path11, nltk_stopwords_path11]):
        print("\nStep11, same as step 10, but using FasText and Glove embeddings")
        reuters_word_score = pickle_read(reuter_word_score_path)
        nps_word_score = pickle_read(nps_word_score_path)
        step_10(
            reuters_word_score,
            nps_word_score,
            fasText_model_path,
            "FasText",
            reuters_stopwords_path11,
            nps_stopwords_path11,
            nltk_stopwords_path11
        )
        step_10(
            reuters_word_score,
            nps_word_score,
            glove_model_path,
            "Glove",
            reuters_stopwords_path11,
            nps_stopwords_path11,
            nltk_stopwords_path11
        )
    print("All calculations that could have been done or relevant files already exist")



if __name__ == "__main__":
    main()