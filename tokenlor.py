#!/usr/bin/python
'''Usage:
./tokenizor.py tweets.txt num_words all_words_output bow_output
'''

import sys
from collections import defaultdict
from operator import itemgetter
import nltk
from nltk import word_tokenize


def read_tweets_to_bow(filename):
    word_counts = defaultdict(int)
    user_word_counts = defaultdict(lambda : defaultdict(int))
    lc = 1
    with open(filename) as f:
        for line in f:
            if lc % 10000 == 0:
                print( "%d lines processed" % lc)
            lc+=1
            items = line.strip().split('\t')
            uid = items[0]
            tweets = items[2].lower()
            for word in word_tokenize(tweets):
                if word[0].isalpha():
                    word_counts[word] = word_counts[word]+1
                    user_word_counts[uid][word] = user_word_counts[uid][word]+1
    return word_counts, user_word_counts


def output_bow(word_dictionary, user_word_counts, filename):
    users = sorted(user_word_counts.keys())
    word2idx = dict([(word_dictionary[i], i) for i in range(len(word_dictionary))])
    ofile = open(filename, 'w')
    for u in users:
        u_word_count = user_word_counts[u]
        idx_count = [(word2idx[w], c) for w,c in u_word_count.iteritems() if w in word2idx]
        sorted_counts = sorted(idx_count, key=itemgetter(0))

        if len(sorted_counts)>0:
            ofile.write("%s\t%s\n" % (u, ' '.join(["%d:%d" % (x[0],x[1]) for x in sorted_counts])))


def output_dictionary(word_dictionary, filename):
    with open(filename, 'w') as f:
        for w in word_dictionary:
            f.write("%s\n" % w)


def main():
    word_counts, user_word_counts = read_tweets_to_bow(sys.argv[1])
    word_counts = sorted([(k,v) for k,v in word_counts.iteritems()], key=itemgetter(1), reverse=True)

    num_top_word = int(sys.argv[2])
    if num_top_word>0 and num_top_word < len(word_counts):
        word_dictionary = [word_counts[i][0] for i in range(num_top_word)]
    else:
        word_dictionary = [wc[0] for wc in word_counts]

    print( "Output results...")
    output_dictionary(word_dictionary, sys.argv[3])
    output_bow(word_dictionary, user_word_counts, sys.argv[4])






if __name__ == "__main__":
    main()





