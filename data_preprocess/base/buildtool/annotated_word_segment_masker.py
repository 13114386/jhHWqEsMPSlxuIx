from __future__ import unicode_literals, print_function, division
import json

class DocIterator():
    def __init__(
        self,
        article_fieldname="article",
        summary_fieldname="highlights",
        skip_line_path=None
    ):
        self.article_fieldname = article_fieldname
        self.summary_fieldname = summary_fieldname
        if skip_line_path is None:
            self.skip_indexes = []
        else:
            self.skip_indexes = self.load_skip_indexes(skip_line_path)
        self.count = 0

    def load_skip_indexes(self, skip_line_path):
        with open(skip_line_path, "r") as fp:
            lines = fp.read()
            skip_line_indexes = lines.splitlines()
        skip_line_indexes = [int(i) for i in skip_line_indexes]
        skip_line_indexes.sort()
        return skip_line_indexes

    def __call__(self, fp):
        for line in fp:
            if len(self.skip_indexes) == 0 or self.count < self.skip_indexes[0]:
                self.count += 1
                data = json.loads(line)
                yield (data[self.article_fieldname], data[self.summary_fieldname])
            elif self.count == self.skip_indexes[0]:
                del self.skip_indexes[0]
                self.count += 1

def word_iterator(doc):
    words = doc.split(" ")
    for word in words:
        yield word


class AnnotatedWordSegmentMasker():
    def __call__(self, sentences, doc, logger):
        '''
            Stanford CoreNLP uses Penn Treebank word segmentation tokenization by default.
            We mask the word (by subword tokens) in order to map them to the BART tokenizer's 
            word segmentation tokenization where the preceding whitespace of a leading subword
            is used as part of the subword.

            sentences: annotated tokens in the form of sentence list.
            There are four cases:
            1. token and word exactly match.
            2. token partially match the word due to:
               a. the annotater may omit some leading characters like "+" sign.
               b. the annotater may omit some trailing characters like "+" sign.
               c. the annotater may some trailing characters at the end of the sentence
                  as tokens of next sentence.
        '''
        word_iter = word_iterator(doc)
        masks = []
        leafover = []
        for index, sentence in enumerate(sentences):
            mask = []
            wpos = 0
            word = None
            word_list = []
            tokens = sentence[0].split(" ")
            for token in tokens:
                while True:
                    if len(leafover) > 0:
                        word = leafover[0]
                        leafover = []
                    if word is None:
                        word = next(word_iter)
                        word_list.append(word)
                    # Try to match the token part of the word.
                    while wpos < len(word) and \
                        len(token) <= len(word[wpos:]):
                        if token == word[wpos:wpos+len(token)]:
                            mask += [1] if wpos == 0 else [0]
                            wpos += len(token)
                            token = None # Signal token being consumed.
                            break
                        else:
                            # Annotator may omit some leading special tokens like "+" in the originial doc.
                            # So, forward one position and try again.
                            wpos += 1

                    if token is None:
                        # Whatever left in the word to be tried by next token of the sentence.
                        # Or, the token of next sentence if the token is the last token of the sentence.
                        if wpos == len(word):
                            # word is consumed too.
                            wpos = 0
                            word = None
                        break
                    else:
                        # If whatever left in the word fails to consume the token,
                        # they are must be omitted stuff by annotator.
                        # Ignore them.
                        wpos = 0
                        word = None
            if wpos > 0 and wpos < len(word):
                # If there are some leftover in the word while
                # all annotated tokens of the sentence are consumed,
                # the annotator must consider the them a token in the next sentence.
                # So, keep it for next sentence.
                leafover.append(word[wpos:])
            #logger.info(f"word list: {word_list}")
            #logger.info(f"annotated token list: {tokens}")
            if len(mask) != len(tokens):
                raise ValueError("Annotated tokens fail to reconstruct original words.")
            masks.append(mask)
        return masks
