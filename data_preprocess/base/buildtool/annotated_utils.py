from __future__ import unicode_literals, print_function, division


def AnnotateSentenceIteratorByField(annotation, fieldname, first_sentence_only=False):
    sentences = annotation["sentences"]
    if first_sentence_only:
        ranges = [0, 1]
    else:
        ranges = [0, len(sentences)]
    for i, sentence in zip(range(*ranges), sentences):
        yield sentence[fieldname]


def AnnotateSentenceIteratorByFields(annotation, fieldnames, first_sentence_only=False):
    sentences = annotation["sentences"]
    if first_sentence_only:
        ranges = [0, 1]
    else:
        ranges = [0, len(sentences)]
    for i, sentence in zip(range(*ranges), sentences):
        annotated = {fieldname: sentence[fieldname] for fieldname in fieldnames}
        yield annotated


def AnnotateCorefsIterator(annotation, first_sentence_only=False):
    corefs = annotation["corefs"]
    for k, mentions in corefs.items():
        if first_sentence_only:
            mentions = [m for m in mentions if m["sentNum"] == 1]
        yield mentions


def cumulative(values):
    length = len(values)
    values = [sum(values[0:x:1]) for x in range(0, length+1)]
    return values
