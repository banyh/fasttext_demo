# -*- coding: utf-8 -*-
from os.path import join
from django.shortcuts import render
from django.http import JsonResponse
from gensim.models.word2vec import Word2Vec

f1 = Word2Vec.load('/home/banyhong/fastText/twwiki1_50/twwiki.gensim')
f2 = Word2Vec.load('/home/banyhong/fastText/twwiki2_50/twwiki.gensim')
f3 = Word2Vec.load('/home/banyhong/fastText/twwiki3_50/twwiki.gensim')
w2v = Word2Vec.load('/media/wordvec/twwiki_stan_dict')


def most_sim(model, word):
    not_found = []
    for w in word:
        if w not in model:
            not_found.append(w)
    if not_found:
        s = u'缺: ' + u' '.join(not_found)
        return [(s, 0.0)] * 10
    return model.most_similar(word)


sim1 = lambda word: most_sim(w2v, word)
sim2 = lambda word: most_sim(f1, word)
sim3 = lambda word: most_sim(f2, word)
sim4 = lambda word: most_sim(f3, word)


def similarity(request):
    word = request.POST['word'].split()
    ret = []
    for (w1, s1), (w2, s2), (w3, s3), (w4, s4) in zip(sim1(word), sim2(word), sim3(word), sim4(word)):
        ret.append((w1, s1, w2, s2, w3, s3, w4, s4))
    return JsonResponse({'data': ret})
