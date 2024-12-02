import csv
import string
from collections import defaultdict
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi

class fullTextSearch:
    __documents = {} # {id : detail}
    __doc_term_freq = {} # {id : dict_freq of words}
    __term_id = {} # {term : id}
    __id_term = {} # {id : term}
    __num_doc = 0

    __inverted_index = defaultdict(dict) # {__term_id : {doc_id: frequency}}

    def __preprocess_text(self, text) -> list:
        '''
            @Params: string - text
            Returns: list of words after preprocess
        '''
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        text = text.lower()

        tokens = ViTokenizer.tokenize(text).split()

        return tokens


    def __preprocess_document(self) -> None:
        '''
            Split each detail of transaction to word and compute frequency of it
        '''
        all_terms = set() # * Set of unique words in whole __documents

        for doc_id, data in self.__documents.items():
            tokens = self.__preprocess_text(data['detail'])
            term_freqs = defaultdict(int)

            # * Calculate frequency of words in each transactions   
            for token in tokens:
                term_freqs[token] += 1
                all_terms.add(token)

            self.__doc_term_freq[doc_id] = term_freqs

        # * Map string to int for faster compute
        for idx, term in enumerate(all_terms):
            self.__term_id[term] = idx
            self.__id_term[idx] = term

        # * Convert string key to int key 
        for doc_id, term_freqs in self.__doc_term_freq.items():
            new_term_freqs = {}
            
            for term, freq in term_freqs.items():
                tid = self.__term_id[term]
                new_term_freqs[tid] = freq

            self.__doc_term_freq[doc_id] = new_term_freqs


    def __build_inverted_index(self):
        '''
            Build inverted index for transactions
            __inverted_index {__term_id : {doc_id: frequency}}
        '''
        for doc_id, term_freqs in self.__doc_term_freq.items():
            for t_id, freq in term_freqs.items():
                self.__inverted_index[t_id][doc_id] = freq;


    def __build_bm25(self):
        self.__corpus = []
        for doc_id in sorted(self.__documents.keys()):
            terms = [self.__id_term[tid] for tid in self.__doc_term_freq[doc_id].keys()]
            self.__corpus.append(terms)
        
        self.bm25 = BM25Okapi(self.__corpus)


    def __init__(self, filename) -> None:
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                description = row 
                self.__documents[self.__num_doc] = description
                self.__num_doc += 1
        self.__preprocess_document()
        self.__build_inverted_index()
        self.__build_bm25()


    def search(self, query, num = 5) -> list:
        '''
            @ Params: query - string: query need to find
            Return 5 document most relevent
        '''
        query_terms = self.__preprocess_text(query)
        scores = self.bm25.get_scores(query_terms)
        
        ranked_docs = sorted([(doc_id, score) for doc_id, score in enumerate(scores)], key=lambda x: x[1], reverse=True)[:num]

        return ranked_docs
    

    def get_transaction_info(self, doc_id):
        if doc_id in self.__documents:
            return self.__documents[doc_id]
        else:
            return None