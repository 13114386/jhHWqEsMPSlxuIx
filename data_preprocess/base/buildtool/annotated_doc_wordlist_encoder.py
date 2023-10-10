from __future__ import unicode_literals, print_function, division
from buildtool.annotated_utils import cumulative

debugging=False

class NestedDataTokenizer():
    def __init__(
        self,
        tokenizer,
        leading_space_word,
        pad_to_max_length,
        ignore_pad_token_for_loss,
        max_length=None,
        truncation=False,
        transform=None
    ):
        self.tokenizer = tokenizer
        self.leading_space_word = leading_space_word
        self.pad_to_max_length = pad_to_max_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.max_target_length = None
        self.max_length = max_length
        self.truncation = truncation
        self.transform = transform

    def __call__(self, data, data_mask, sent_sizes, logger):
        new_data = []
        for item, mask, sizes in zip(data, data_mask, sent_sizes):
            if isinstance(item, list):
                d = self(item, mask, sizes, logger) # Recursive
                new_data.append(d)
            else:
                output = self.tokenize(item, mask, sizes)
                new_data.append(output)
        return new_data

    def tokenize_word_list(self, data, data_mask, sent_sizes):
        '''
            There may be a unexhausted list of conditions by which byte pair segments a word.
            So, we can't reliably rely on return_offsets_mapping of a tokenizer on a sentence.
            We instead tokenize word by word to ensure a proper subword offset mask.
        '''
        mask = []
        tokens = []
        if self.transform:
            data = self.transform(data)

        if not isinstance(data, list):
            word_list = data.split(" ")
        else:
            word_list = data

        cumul_sizes = cumulative(sent_sizes)
        tokenized_sent_sizes = []
        sent_ptr = 0
        # for index, (word, word_mask) in enumerate(zip(word_list, data_mask)):
        for index, word in enumerate(word_list):
            if self.leading_space_word and 0 < index:# and word_mask:
                # BART tokenizer treat leading space as part of a word in a sentence.
                # So, we mimic the effect.
                word = " " + word
            output = self.tokenizer(word,
                                    max_length=self.max_target_length,
                                    padding=False,
                                    truncation=False)

            input_ids = output["input_ids"]
            bos, eos = input_ids[0], input_ids[-1]
            input_ids = input_ids[1:-1]

            n_added = len(input_ids)
            if self.truncation and (self.max_length is not None and self.max_length > 0):
                n_added = min(n_added, self.max_length - (len(tokens) + 2))
                # n_added = max(0, n_added)
                assert n_added >= 0, "n_added should not be less than zero."
            if n_added > 0:
                tokens += input_ids[:n_added]
                mask += [1] + [0]*(n_added-1)

            # Reach the length limit.
            if n_added < len(input_ids):
                length = len(tokens) - sum(tokenized_sent_sizes)
                if length > 0:
                    tokenized_sent_sizes.append(length)
                break

            # When reaching the end of the word level sentence,
            # record the length of token level sentence.
            if index == cumul_sizes[sent_ptr+1]-1:
                # subtract the lengths of previous sentences
                length = len(tokens) - sum(tokenized_sent_sizes)
                tokenized_sent_sizes.append(length)
                sent_ptr += 1

        tokens = [bos] + tokens + [eos]
        mask = [1] + mask + [1]
        # Take into account bos and eos
        tokenized_sent_sizes[0]+=1
        tokenized_sent_sizes[-1]+=1
        assert len(tokens) == sum(tokenized_sent_sizes)

        attention_mask = [1]*len(tokens)

        output = {"input_ids": tokens,
                  "attention_mask": attention_mask,
                  "token_mask": mask,
                  "tokenized_sent_sizes": tokenized_sent_sizes}
        return output

    def tokenize(self, data, data_mask, sent_sizes):
        output = self.tokenize_word_list(data, data_mask, sent_sizes)

        if debugging:
            test_output = self.tokenizer(data,
                                        max_length=self.max_target_length,
                                        padding=self.pad_to_max_length,
                                        truncation=False,
                                        return_offsets_mapping=True)
            print("sentence: ", test_output)
            print("wordlist: ", output)
            if not (test_output["input_ids"] == output["input_ids"]):
                raise ValueError("tokenizing word list failed to match tokenizing sentence.")

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.pad_to_max_length == "max_length" and self.ignore_pad_token_for_loss:
            output["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in data] for data in output["input_ids"]
            ]

        return output

class AnnotatedDocEncoder():
    def __init__(
        self,
        tokenizer,
        leading_space_word,
        pad_to_max_length,
        ignore_pad_token_for_loss,
        max_length=None,
        truncation=False,
    ):
        self.nd_tokenizer = NestedDataTokenizer(tokenizer,
                                                leading_space_word,
                                                pad_to_max_length,
                                                ignore_pad_token_for_loss,
                                                max_length=max_length,
                                                truncation=truncation,
                                            )

    def __call__(self, doc, doc_mask, sent_sizes, logger):
        output = self.nd_tokenizer(doc, doc_mask, sent_sizes, logger)
        return output
