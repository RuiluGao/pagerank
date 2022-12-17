#!/usr/bin/python3

'''
This file calculates pagerank vectors for small-scale webgraphs.
'''

import math
import torch
import gzip
import csv

import logging
import gensim.downloader
vectors = gensim.downloader.load('glove-wiki-gigaword-300')


class WebGraph():

    def __init__(self, filename, max_nnz=None, filter_ratio=None):

        self.url_dict = {}
        indices = []

        from collections import defaultdict
        target_counts = defaultdict(lambda: 0)

        # loop through filename to extract the indices
        logging.debug('computing indices')
        with gzip.open(filename,newline='',mode='rt') as f:
            for i,row in enumerate(csv.DictReader(f)):
                if max_nnz is not None and i>max_nnz:
                    break
                import re
                regex = re.compile(r'.*((/$)|(/.*/)).*')
                if regex.match(row['source']) or regex.match(row['target']):
                    continue
                source = self._url_to_index(row['source'])
                target = self._url_to_index(row['target'])
                target_counts[target] += 1
                indices.append([source,target])

        # remove urls with too many in-links
        if filter_ratio is not None:
            new_indices = []
            for source,target in indices:
                if target_counts[target] < filter_ratio*len(self.url_dict):
                    new_indices.append([source,target])
            indices = new_indices

        # compute the values
        logging.debug('computing values')
        values = []
        last_source = indices[0][0]
        last_i = 0
        for i,(source,target) in enumerate(indices+[(None,None)]):
            if source==last_source:
                pass
            else:
                total_links = i-last_i
                values.extend([1/total_links]*total_links)
                last_source = source
                last_i = i

        # generate the sparse matrix
        i = torch.LongTensor(indices).t()
        v = torch.FloatTensor(values)
        n = len(self.url_dict)
        self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
        self.index_dict = {v: k for k, v in self.url_dict.items()}
    

    def _url_to_index(self, url):
        if url not in self.url_dict:
            self.url_dict[url] = len(self.url_dict)
        return self.url_dict[url]


    def _index_to_url(self, index):
        return self.index_dict[index]


    def make_personalization_vector(self, query=None):
        n = self.P.shape[0]

        if query is None:
            v = torch.ones(n)

        else:
            v = torch.zeros(n)
            for url,i in self.url_dict.items():
                if url_satisfies_query(url, query):
                    v[i] = 1
        
        v_sum = torch.sum(v)
        assert(v_sum>0)
        v /= v_sum

        return v


    def power_method(self, v=None, x0=None, alpha=0.85, max_iterations=1000, epsilon=1e-6):
        with torch.no_grad():
            n = self.P.shape[0]

            # compute the a vector
            nondangling_nodes = torch.sparse.sum(self.P,1).indices()
            a = torch.ones([n,1])
            a[nondangling_nodes] = 0

            # create input variables if none given
            if v is None:
                v = torch.Tensor([1/n]*n)
                v = torch.unsqueeze(v,1)
            v /= torch.norm(v)

            if x0 is None:
                x0 = torch.Tensor([1/(math.sqrt(n))]*n)
                x0 = torch.unsqueeze(x0,1)
            x0 /= torch.norm(x0)

            # main loop
            xprev = x0
            x = xprev.detach().clone()
            for i in range(max_iterations):
                xprev = x.detach().clone()
                q = (alpha*x.t()@a + (1-alpha)) * v.t()
                x = torch.sparse.addmm(
                        q.t(),
                        self.P.t(),
                        x,
                        beta=1,
                        alpha=alpha
                        )
                x /= torch.norm(x, p=1)
                accuracy = torch.norm(x-xprev)
                logging.debug('i='+str(i)+' accuracy='+str(accuracy)+' sum(x)='+str( torch.sum(x)))
                if accuracy < epsilon:
                    break

            return x.squeeze()


    def search(self, pi, query='', max_results=10, p=35):
        '''
        This function prints the top ranked urls that match the input query.

        NOTE:
        For the Task 1, there is no need to modify this code.
        For the Task 2, then you would have to modify this code.
        '''
        n = self.P.shape[0]
        vals,indices = torch.topk(pi,n)

        urls = [self._index_to_url(index.item()) for index in indices]
        pagerank = [val.item() for val in vals]

        scores = []

        similar_words = []
        for term in query.split():
            if term[0] != '-':
                similar_words += get_similar_words(term, add_score=True)

        if query == '':
            scores = pagerank

        else:
            for i, url in enumerate(urls):
                score = 0
                for word_vector in similar_words:
                    word = word_vector[0]
                    word_similarity = word_vector[1]
                    new_n = url.count(word)
                    score += new_n*(word_similarity**p)

                ranking = pagerank[i] * score
                scores.append(ranking)

        url_score = list(zip(urls, scores))
        url_score.sort(key=lambda x: x[1], reverse=True)
        
        matches = 0
        for i in range(n):
            if matches >= max_results:
                break
            url = url_score[i][0]
            if url_satisfies_query(url,query):
                ranking = url_score[i][1]
                logging.info(f'rank={matches} ranking={ranking:0.4e} url={url}')
                matches += 1

def get_similar_words(term, n=5, add_score=False):
    similar_words_v = vectors.most_similar(term)[:n]

    similar_words = []
    if not add_score:
        for similar_word_v in similar_words_v:
            similar_words.append(similar_word_v[0])
    else:
        return similar_words_v

    return similar_words


def url_satisfies_query(url, query):
    '''
    This functions supports a moderately sophisticated syntax for searching urls for a query string.
    The function returns True if any word in the query string is present in the url.
    But, if a word is preceded by the negation sign `-`,
    then the function returns False if that word is present in the url,
    even if it would otherwise return True.

    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'coronavirus covid')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'coronavirus')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid -speech')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid -corona')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '-speech')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '-corona')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '')
    True
    '''
    satisfies = False
    terms = query.split()

    num_terms=0
    for term in terms:
        if term[0] != '-':
            num_terms+=1
            similar_terms = get_similar_words(term)
            if term in url:
                satisfies = True
            for similar_term in similar_terms:
                if similar_term in url:
                    satisfies = True
    if num_terms==0:
        satisfies=True

    for term in terms:
        if term[0] == '-':
            if term[1:] in url:
                return False
    return satisfies

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--personalization_vector_query', default='')
    parser.add_argument('--search_query', default='')
    parser.add_argument('--filter_ratio', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=0.85)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--max_results', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    g = WebGraph(args.data, filter_ratio=args.filter_ratio)
    v = g.make_personalization_vector(args.personalization_vector_query)
    pi = g.power_method(v, alpha=args.alpha, max_iterations=args.max_iterations, epsilon=args.epsilon)
    g.search(pi, query=args.search_query, max_results=args.max_results)
