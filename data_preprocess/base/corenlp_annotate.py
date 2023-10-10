from __future__ import unicode_literals, print_function, division
import sys, os, argparse, json, codecs
from PyCoreNLP.PyCoreNLP import PyCoreNLP
from file_utils import ChunkSaver

def loadFromJson(filename):
    with codecs.open(filename,'r',encoding = 'utf-8') as fp:
        data = json.load(fp, strict = False)
    return data

def saveToJson(filename, data):
    with codecs.open(filename,'w',encoding = 'utf-8') as fp:
        json.dump(data, fp, indent=4)


class NLParsingProcess():
    def __init__(self):
        super().__init__()

    def __call__(self, args, annotator):
        source_filepath = os.path.join(args.dataset_root,
                                       args.source_folder,
                                       f'{args.split_type}{args.file_ext}')
        output_dir = os.path.join(args.dataset_root, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
        error_logpath = os.path.join(output_dir,
                                    args.error_log.format(split_type=args.split_type))
        annotation_filepaths = {colname: os.path.join(output_dir,
                                            f'{args.split_type}.{colname}{args.file_ext}') \
                                                for colname in args.column_names}
        annotations = {colname: [] for colname in args.column_names}
        with ChunkSaver(annotation_filepaths, args.chunk_size, convert_json=False) as saver, \
            open(source_filepath, "r", encoding="utf-8") as src_fp, \
            open(error_logpath, "w") as err_fp:

            # Acquire annotation from parsing service.
            for index, line in enumerate(src_fp):
                try:
                    line = json.loads(line)
                    # Annotate both before adding to list to sync
                    encoded0 = self.annotate(line[args.column_names[0]], annotator, mode=args.corenlp_mode)
                    encoded1 = self.annotate(line[args.column_names[1]], annotator, mode=args.corenlp_mode)
                    annotations[args.column_names[0]].append(encoded0)
                    annotations[args.column_names[1]].append(encoded1)
                    if saver(features=annotations, index=index, last_save=False):
                        [annotations[k].clear() for k, v in annotations.items()]
                except (json.decoder.JSONDecodeError, UnicodeEncodeError) as ex:
                    err_fp.write(f"{index}:->{str(ex)}\n")
                    err_fp.flush()
                    print(ex, file=sys.stderr)
                except Exception as ex:
                    err_fp.write(f"{index}:->{str(ex)}\n")
                    err_fp.flush()
                    print(ex, file=sys.stderr)
                except:
                    err_fp.write(f"{index}:->{str(ex)}\n")
                    err_fp.flush()
                    print("Unknown exception\n", file=sys.stderr)

            try:
                # If any remaining
                if saver(features=annotations, index=index, last_save=True):
                    [annotations[k].clear() for k, v in annotations.items()]
            except Exception as ex:
                err_fp.write(f"{index}:->{str(ex)}\n")
                err_fp.flush()
                print(ex, file=sys.stderr)
            except:
                err_fp.write(f"{index}:->Unknown exception\n")
                err_fp.flush()
                print("Unknown exception\n", file=sys.stderr)

    def annotate(self, src, annotator, mode):
        # First column's annotation
        text = src.strip()
        annotated = annotator.annotate(text, mode=mode)
        encoded = json.dumps(annotated, ensure_ascii=False)#.encode('utf8').decode()
        return encoded


class DummpyAnnotator():
    def __init__(self, bugit=False):
        self.bugit = bugit

    def annotate(self, data):
        if not self.bugit:
            return {"sentences": [{"index": 0, "parse": "(ROOT\n  (S\n    (NP (NNP Sigma) (NNP Alpha) (NNP Epsilon))\n    (VP (VBZ is)\n      (VP (VBG being)\n        (VP (VBD tossed)\n          (PRT (RP out))\n          (PP (IN by)\n            (NP\n              (NP (DT the) (NNP University))\n              (PP (IN of)\n                (NP (NNP Oklahoma))))))))\n    (. .)))", "basicDependencies": [{"dep": "ROOT", "governor": 0, "governorGloss": "ROOT", "dependent": 6, "dependentGloss": "tossed"}, {"dep": "compound", "governor": 3, "governorGloss": "Epsilon", "dependent": 1, "dependentGloss": "Sigma"}, {"dep": "compound", "governor": 3, "governorGloss": "Epsilon", "dependent": 2, "dependentGloss": "Alpha"}, {"dep": "nsubjpass", "governor": 6, "governorGloss": "tossed", "dependent": 3, "dependentGloss": "Epsilon"}, {"dep": "aux", "governor": 6, "governorGloss": "tossed", "dependent": 4, "dependentGloss": "is"}, {"dep": "auxpass", "governor": 6, "governorGloss": "tossed", "dependent": 5, "dependentGloss": "being"}, {"dep": "compound:prt", "governor": 6, "governorGloss": "tossed", "dependent": 7, "dependentGloss": "out"}, {"dep": "case", "governor": 10, "governorGloss": "University", "dependent": 8, "dependentGloss": "by"}, {"dep": "det", "governor": 10, "governorGloss": "University", "dependent": 9, "dependentGloss": "the"}, {"dep": "nmod", "governor": 6, "governorGloss": "tossed", "dependent": 10, "dependentGloss": "University"}, {"dep": "case", "governor": 12, "governorGloss": "Oklahoma", "dependent": 11, "dependentGloss": "of"}, {"dep": "nmod", "governor": 10, "governorGloss": "University", "dependent": 12, "dependentGloss": "Oklahoma"}, {"dep": "punct", "governor": 6, "governorGloss": "tossed", "dependent": 13, "dependentGloss": "."}], "enhancedDependencies": [{"dep": "ROOT", "governor": 0, "governorGloss": "ROOT", "dependent": 6, "dependentGloss": "tossed"}, {"dep": "compound", "governor": 3, "governorGloss": "Epsilon", "dependent": 1, "dependentGloss": "Sigma"}, {"dep": "compound", "governor": 3, "governorGloss": "Epsilon", "dependent": 2, "dependentGloss": "Alpha"}, {"dep": "nsubjpass", "governor": 6, "governorGloss": "tossed", "dependent": 3, "dependentGloss": "Epsilon"}, {"dep": "aux", "governor": 6, "governorGloss": "tossed", "dependent": 4, "dependentGloss": "is"}, {"dep": "auxpass", "governor": 6, "governorGloss": "tossed", "dependent": 5, "dependentGloss": "being"}, {"dep": "compound:prt", "governor": 6, "governorGloss": "tossed", "dependent": 7, "dependentGloss": "out"}, {"dep": "case", "governor": 10, "governorGloss": "University", "dependent": 8, "dependentGloss": "by"}, {"dep": "det", "governor": 10, "governorGloss": "University", "dependent": 9, "dependentGloss": "the"}, {"dep": "nmod:agent", "governor": 6, "governorGloss": "tossed", "dependent": 10, "dependentGloss": "University"}, {"dep": "case", "governor": 12, "governorGloss": "Oklahoma", "dependent": 11, "dependentGloss": "of"}, {"dep": "nmod:of", "governor": 10, "governorGloss": "University", "dependent": 12, "dependentGloss": "Oklahoma"}, {"dep": "punct", "governor": 6, "governorGloss": "tossed", "dependent": 13, "dependentGloss": "."}], "enhancedPlusPlusDependencies": [{"dep": "ROOT", "governor": 0, "governorGloss": "ROOT", "dependent": 6, "dependentGloss": "tossed"}, {"dep": "compound", "governor": 3, "governorGloss": "Epsilon", "dependent": 1, "dependentGloss": "Sigma"}, {"dep": "compound", "governor": 3, "governorGloss": "Epsilon", "dependent": 2, "dependentGloss": "Alpha"}, {"dep": "nsubjpass", "governor": 6, "governorGloss": "tossed", "dependent": 3, "dependentGloss": "Epsilon"}, {"dep": "aux", "governor": 6, "governorGloss": "tossed", "dependent": 4, "dependentGloss": "is"}, {"dep": "auxpass", "governor": 6, "governorGloss": "tossed", "dependent": 5, "dependentGloss": "being"}, {"dep": "compound:prt", "governor": 6, "governorGloss": "tossed", "dependent": 7, "dependentGloss": "out"}, {"dep": "case", "governor": 10, "governorGloss": "University", "dependent": 8, "dependentGloss": "by"}, {"dep": "det", "governor": 10, "governorGloss": "University", "dependent": 9, "dependentGloss": "the"}, {"dep": "nmod:agent", "governor": 6, "governorGloss": "tossed", "dependent": 10, "dependentGloss": "University"}, {"dep": "case", "governor": 12, "governorGloss": "Oklahoma", "dependent": 11, "dependentGloss": "of"}, {"dep": "nmod:of", "governor": 10, "governorGloss": "University", "dependent": 12, "dependentGloss": "Oklahoma"}, {"dep": "punct", "governor": 6, "governorGloss": "tossed", "dependent": 13, "dependentGloss": "."}], "tokens": [{"index": 1, "word": "Sigma", "originalText": "Sigma", "lemma": "Sigma", "characterOffsetBegin": 0, "characterOffsetEnd": 5, "pos": "NNP", "before": "", "after": " "}, {"index": 2, "word": "Alpha", "originalText": "Alpha", "lemma": "Alpha", "characterOffsetBegin": 6, "characterOffsetEnd": 11, "pos": "NNP", "before": " ", "after": " "}, {"index": 3, "word": "Epsilon", "originalText": "Epsilon", "lemma": "Epsilon", "characterOffsetBegin": 12, "characterOffsetEnd": 19, "pos": "NNP", "before": " ", "after": " "}, {"index": 4, "word": "is", "originalText": "is", "lemma": "be", "characterOffsetBegin": 20, "characterOffsetEnd": 22, "pos": "VBZ", "before": " ", "after": " "}, {"index": 5, "word": "being", "originalText": "being", "lemma": "be", "characterOffsetBegin": 23, "characterOffsetEnd": 28, "pos": "VBG", "before": " ", "after": " "}, {"index": 6, "word": "tossed", "originalText": "tossed", "lemma": "toss", "characterOffsetBegin": 29, "characterOffsetEnd": 35, "pos": "VBD", "before": " ", "after": " "}, {"index": 7, "word": "out", "originalText": "out", "lemma": "out", "characterOffsetBegin": 36, "characterOffsetEnd": 39, "pos": "RP", "before": " ", "after": " "}, {"index": 8, "word": "by", "originalText": "by", "lemma": "by", "characterOffsetBegin": 40, "characterOffsetEnd": 42, "pos": "IN", "before": " ", "after": " "}, {"index": 9, "word": "the", "originalText": "the", "lemma": "the", "characterOffsetBegin": 43, "characterOffsetEnd": 46, "pos": "DT", "before": " ", "after": " "}, {"index": 10, "word": "University", "originalText": "University", "lemma": "University", "characterOffsetBegin": 47, "characterOffsetEnd": 57, "pos": "NNP", "before": " ", "after": " "}, {"index": 11, "word": "of", "originalText": "of", "lemma": "of", "characterOffsetBegin": 58, "characterOffsetEnd": 60, "pos": "IN", "before": " ", "after": " "}, {"index": 12, "word": "Oklahoma", "originalText": "Oklahoma", "lemma": "Oklahoma", "characterOffsetBegin": 61, "characterOffsetEnd": 69, "pos": "NNP", "before": " ", "after": " "}, {"index": 13, "word": ".", "originalText": ".", "lemma": ".", "characterOffsetBegin": 70, "characterOffsetEnd": 71, "pos": ".", "before": " ", "after": "\n"}]}, {"index": 1, "parse": "(ROOT\n  (S\n    (NP (PRP It))\n    (VP (VBZ 's)\n      (ADVP (RB also))\n      (VP (VBN run)\n        (NP\n          (NP (NN afoul))\n          (PP (IN of)\n            (NP (NNS officials))))\n        (PP (IN at)\n          (NP (NNP Yale) (, ,) (NNP Stanford)\n            (CC and)\n            (NNP Johns) (NNP Hopkins)))\n        (PP (IN in)\n          (NP (JJ recent) (NNS months)))))\n    (. .)))", "basicDependencies": [{"dep": "ROOT", "governor": 0, "governorGloss": "ROOT", "dependent": 4, "dependentGloss": "run"}, {"dep": "nsubjpass", "governor": 4, "governorGloss": "run", "dependent": 1, "dependentGloss": "It"}, {"dep": "auxpass", "governor": 4, "governorGloss": "run", "dependent": 2, "dependentGloss": "'s"}, {"dep": "advmod", "governor": 4, "governorGloss": "run", "dependent": 3, "dependentGloss": "also"}, {"dep": "dobj", "governor": 4, "governorGloss": "run", "dependent": 5, "dependentGloss": "afoul"}, {"dep": "case", "governor": 7, "governorGloss": "officials", "dependent": 6, "dependentGloss": "of"}, {"dep": "nmod", "governor": 5, "governorGloss": "afoul", "dependent": 7, "dependentGloss": "officials"}, {"dep": "case", "governor": 14, "governorGloss": "Hopkins", "dependent": 8, "dependentGloss": "at"}, {"dep": "compound", "governor": 14, "governorGloss": "Hopkins", "dependent": 9, "dependentGloss": "Yale"}, {"dep": "punct", "governor": 9, "governorGloss": "Yale", "dependent": 10, "dependentGloss": ","}, {"dep": "conj", "governor": 9, "governorGloss": "Yale", "dependent": 11, "dependentGloss": "Stanford"}, {"dep": "cc", "governor": 9, "governorGloss": "Yale", "dependent": 12, "dependentGloss": "and"}, {"dep": "conj", "governor": 9, "governorGloss": "Yale", "dependent": 13, "dependentGloss": "Johns"}, {"dep": "nmod", "governor": 4, "governorGloss": "run", "dependent": 14, "dependentGloss": "Hopkins"}, {"dep": "case", "governor": 17, "governorGloss": "months", "dependent": 15, "dependentGloss": "in"}, {"dep": "amod", "governor": 17, "governorGloss": "months", "dependent": 16, "dependentGloss": "recent"}, {"dep": "nmod", "governor": 4, "governorGloss": "run", "dependent": 17, "dependentGloss": "months"}, {"dep": "punct", "governor": 4, "governorGloss": "run", "dependent": 18, "dependentGloss": "."}], "enhancedDependencies": [{"dep": "ROOT", "governor": 0, "governorGloss": "ROOT", "dependent": 4, "dependentGloss": "run"}, {"dep": "nsubjpass", "governor": 4, "governorGloss": "run", "dependent": 1, "dependentGloss": "It"}, {"dep": "auxpass", "governor": 4, "governorGloss": "run", "dependent": 2, "dependentGloss": "'s"}, {"dep": "advmod", "governor": 4, "governorGloss": "run", "dependent": 3, "dependentGloss": "also"}, {"dep": "dobj", "governor": 4, "governorGloss": "run", "dependent": 5, "dependentGloss": "afoul"}, {"dep": "case", "governor": 7, "governorGloss": "officials", "dependent": 6, "dependentGloss": "of"}, {"dep": "nmod:of", "governor": 5, "governorGloss": "afoul", "dependent": 7, "dependentGloss": "officials"}, {"dep": "case", "governor": 14, "governorGloss": "Hopkins", "dependent": 8, "dependentGloss": "at"}, {"dep": "compound", "governor": 14, "governorGloss": "Hopkins", "dependent": 9, "dependentGloss": "Yale"}, {"dep": "punct", "governor": 9, "governorGloss": "Yale", "dependent": 10, "dependentGloss": ","}, {"dep": "conj:and", "governor": 9, "governorGloss": "Yale", "dependent": 11, "dependentGloss": "Stanford"}, {"dep": "compound", "governor": 14, "governorGloss": "Hopkins", "dependent": 11, "dependentGloss": "Stanford"}, {"dep": "cc", "governor": 9, "governorGloss": "Yale", "dependent": 12, "dependentGloss": "and"}, {"dep": "conj:and", "governor": 9, "governorGloss": "Yale", "dependent": 13, "dependentGloss": "Johns"}, {"dep": "compound", "governor": 14, "governorGloss": "Hopkins", "dependent": 13, "dependentGloss": "Johns"}, {"dep": "nmod:at", "governor": 4, "governorGloss": "run", "dependent": 14, "dependentGloss": "Hopkins"}, {"dep": "case", "governor": 17, "governorGloss": "months", "dependent": 15, "dependentGloss": "in"}, {"dep": "amod", "governor": 17, "governorGloss": "months", "dependent": 16, "dependentGloss": "recent"}, {"dep": "nmod:in", "governor": 4, "governorGloss": "run", "dependent": 17, "dependentGloss": "months"}, {"dep": "punct", "governor": 4, "governorGloss": "run", "dependent": 18, "dependentGloss": "."}], "enhancedPlusPlusDependencies": [{"dep": "ROOT", "governor": 0, "governorGloss": "ROOT", "dependent": 4, "dependentGloss": "run"}, {"dep": "nsubjpass", "governor": 4, "governorGloss": "run", "dependent": 1, "dependentGloss": "It"}, {"dep": "auxpass", "governor": 4, "governorGloss": "run", "dependent": 2, "dependentGloss": "'s"}, {"dep": "advmod", "governor": 4, "governorGloss": "run", "dependent": 3, "dependentGloss": "also"}, {"dep": "dobj", "governor": 4, "governorGloss": "run", "dependent": 5, "dependentGloss": "afoul"}, {"dep": "case", "governor": 7, "governorGloss": "officials", "dependent": 6, "dependentGloss": "of"}, {"dep": "nmod:of", "governor": 5, "governorGloss": "afoul", "dependent": 7, "dependentGloss": "officials"}, {"dep": "case", "governor": 14, "governorGloss": "Hopkins", "dependent": 8, "dependentGloss": "at"}, {"dep": "compound", "governor": 14, "governorGloss": "Hopkins", "dependent": 9, "dependentGloss": "Yale"}, {"dep": "punct", "governor": 9, "governorGloss": "Yale", "dependent": 10, "dependentGloss": ","}, {"dep": "conj:and", "governor": 9, "governorGloss": "Yale", "dependent": 11, "dependentGloss": "Stanford"}, {"dep": "compound", "governor": 14, "governorGloss": "Hopkins", "dependent": 11, "dependentGloss": "Stanford"}, {"dep": "cc", "governor": 9, "governorGloss": "Yale", "dependent": 12, "dependentGloss": "and"}, {"dep": "conj:and", "governor": 9, "governorGloss": "Yale", "dependent": 13, "dependentGloss": "Johns"}, {"dep": "compound", "governor": 14, "governorGloss": "Hopkins", "dependent": 13, "dependentGloss": "Johns"}, {"dep": "nmod:at", "governor": 4, "governorGloss": "run", "dependent": 14, "dependentGloss": "Hopkins"}, {"dep": "case", "governor": 17, "governorGloss": "months", "dependent": 15, "dependentGloss": "in"}, {"dep": "amod", "governor": 17, "governorGloss": "months", "dependent": 16, "dependentGloss": "recent"}, {"dep": "nmod:in", "governor": 4, "governorGloss": "run", "dependent": 17, "dependentGloss": "months"}, {"dep": "punct", "governor": 4, "governorGloss": "run", "dependent": 18, "dependentGloss": "."}], "tokens": [{"index": 1, "word": "It", "originalText": "It", "lemma": "it", "characterOffsetBegin": 72, "characterOffsetEnd": 74, "pos": "PRP", "before": "\n", "after": ""}, {"index": 2, "word": "'s", "originalText": "'s", "lemma": "be", "characterOffsetBegin": 74, "characterOffsetEnd": 76, "pos": "VBZ", "before": "", "after": " "}, {"index": 3, "word": "also", "originalText": "also", "lemma": "also", "characterOffsetBegin": 77, "characterOffsetEnd": 81, "pos": "RB", "before": " ", "after": " "}, {"index": 4, "word": "run", "originalText": "run", "lemma": "run", "characterOffsetBegin": 82, "characterOffsetEnd": 85, "pos": "VBN", "before": " ", "after": " "}, {"index": 5, "word": "afoul", "originalText": "afoul", "lemma": "afoul", "characterOffsetBegin": 86, "characterOffsetEnd": 91, "pos": "NN", "before": " ", "after": " "}, {"index": 6, "word": "of", "originalText": "of", "lemma": "of", "characterOffsetBegin": 92, "characterOffsetEnd": 94, "pos": "IN", "before": " ", "after": " "}, {"index": 7, "word": "officials", "originalText": "officials", "lemma": "official", "characterOffsetBegin": 95, "characterOffsetEnd": 104, "pos": "NNS", "before": " ", "after": " "}, {"index": 8, "word": "at", "originalText": "at", "lemma": "at", "characterOffsetBegin": 105, "characterOffsetEnd": 107, "pos": "IN", "before": " ", "after": " "}, {"index": 9, "word": "Yale", "originalText": "Yale", "lemma": "Yale", "characterOffsetBegin": 108, "characterOffsetEnd": 112, "pos": "NNP", "before": " ", "after": ""}, {"index": 10, "word": ",", "originalText": ",", "lemma": ",", "characterOffsetBegin": 112, "characterOffsetEnd": 113, "pos": ",", "before": "", "after": " "}, {"index": 11, "word": "Stanford", "originalText": "Stanford", "lemma": "Stanford", "characterOffsetBegin": 114, "characterOffsetEnd": 122, "pos": "NNP", "before": " ", "after": " "}, {"index": 12, "word": "and", "originalText": "and", "lemma": "and", "characterOffsetBegin": 123, "characterOffsetEnd": 126, "pos": "CC", "before": " ", "after": " "}, {"index": 13, "word": "Johns", "originalText": "Johns", "lemma": "Johns", "characterOffsetBegin": 127, "characterOffsetEnd": 132, "pos": "NNP", "before": " ", "after": " "}, {"index": 14, "word": "Hopkins", "originalText": "Hopkins", "lemma": "Hopkins", "characterOffsetBegin": 133, "characterOffsetEnd": 140, "pos": "NNP", "before": " ", "after": " "}, {"index": 15, "word": "in", "originalText": "in", "lemma": "in", "characterOffsetBegin": 141, "characterOffsetEnd": 143, "pos": "IN", "before": " ", "after": " "}, {"index": 16, "word": "recent", "originalText": "recent", "lemma": "recent", "characterOffsetBegin": 144, "characterOffsetEnd": 150, "pos": "JJ", "before": " ", "after": " "}, {"index": 17, "word": "months", "originalText": "months", "lemma": "month", "characterOffsetBegin": 151, "characterOffsetEnd": 157, "pos": "NNS", "before": " ", "after": " "}, {"index": 18, "word": ".", "originalText": ".", "lemma": ".", "characterOffsetBegin": 158, "characterOffsetEnd": 159, "pos": ".", "before": " ", "after": ""}]}]}
        else:
            return {[1]}


def parsing_phase(args):
    if args.port:
        url = f'http://localhost:{args.port}'

    # annotator = DummpyAnnotator(bugit=False) # For debugging
    annotator = PyCoreNLP(url)
    nlp_process = NLParsingProcess()
    nlp_process(args, annotator)
    print(f"Processed: {args.split_type}")

def parse_args():
    parser = argparse.ArgumentParser(description="NLP parsing task by Stanford CoreNLP")
    parser.add_argument('--dataset_root', type=str, default="./",
                        help='The root directory in which all data are to load and save.')
    parser.add_argument('--source_folder', type=str, default="downloaded",
                        help='The folder in which raw text data files reside.')
    parser.add_argument('--output_folder', type=str, default="corenlp.parse",
                        help='The folder to which parsing annotation data is saved.')
    parser.add_argument('--split_type', type=str, default="train",
                        help='The data split type, one of [train, valid, test].')
    parser.add_argument('--corenlp_mode', type=str, default=None,
                        help='Stanford CoreNLP tasking mode.')
    parser.add_argument('--file_ext', type=str, default=".json",
                        help='The raw source text data file name pattern.')
    parser.add_argument('--column_names', type=str, default=["article", "highlights"])
    parser.add_argument('--error_log', type=str, default="error.{split_type}.log")
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Save annotation data by the chunk size.')
    parser.add_argument('--port', type=str, default=None,
                        help='The communication port number of Stanford CoreNLP sever.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    parsing_phase(args)

if __name__ == '__main__':
    main()
    print("Done")
