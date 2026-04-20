import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    QWERTY_NEIGHBORS = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'srfce', 'e': 'wsdr',
        'f': 'drtgc', 'g': 'ftyh', 'h': 'gyuj', 'i': 'ujko', 'j': 'huik',
        'k': 'jiol', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
        'z': 'asx'
    }

    def add_typo(word):
        chars = list(word.lower())
        idx = random.randint(0, len(chars) - 1)
        c = chars[idx]
        if c in QWERTY_NEIGHBORS:
            chars[idx] = random.choice(QWERTY_NEIGHBORS[c])
        return ''.join(chars)

    tokens = word_tokenize(example["text"])
    new_tokens = []

    for word in tokens:
        # synonym replacement
        if random.random() < 0.15 and word.isalpha() and len(word) > 4:
            synsets = wordnet.synsets(word)
            if synsets:
                lemmas = synsets[0].lemmas()
                candidates = [
                    l.name() for l in lemmas
                    if l.name().lower() != word.lower()
                    and "_" not in l.name()
                    and l.name().isalpha()
                ]
                if candidates:
                    new_tokens.append(random.choice(candidates))
                    continue
        # typo injection
        if random.random() < 0.50 and word.isalpha() and len(word) > 3:
            new_tokens.append(add_typo(word))
            continue
        new_tokens.append(word)

    example["text"] = TreebankWordDetokenizer().detokenize(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example