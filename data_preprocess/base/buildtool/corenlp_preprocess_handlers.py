from __future__ import unicode_literals, print_function, division
'''
    Refactor code from https://github.com/KaiQiangSong/struct_infused_summ
'''
import regex

debugging=False

def stringify_ranges(ranges):
    if isinstance(ranges, list):
        return "_".join([str(i) for i in ranges])
    elif isinstance(ranges, str):
        return regex.sub(r"(\d+\)-(\d+)", r"\1_\2", ranges)
    else:
        return ranges


class DependencyParsingProcess():
    def __call__(self, data, missing_token=""):
        tree = self.extract(data, missing_token)
        return self.extract_bfs(tree)

    def extract(self, data, missing_token):
        number = len(data)
        inarc = [""] * number
        edge = [[]] * number
        token = [""] * number
        ROOT = -1

        for item in data:
            id = item["dependent"] - 1
            father = item["governor"] - 1

            inarc[id] = item["dep"]
            token[id] = item.get("dependentGloss", missing_token)
            if debugging and token[id] == missing_token:
                print("========The item misses dependentGloss========")
                print(item)

            if (inarc[id] != "ROOT"):
                if edge[father] != []:
                    edge[father].append(id)
                else:
                    edge[father] = [id]
            else:
                ROOT = id

        tree = {"number": number,
                "inarc": inarc,
                "edge": edge,
                "token": token,
                "ROOT": ROOT}
        return tree

    def extract_bfs(self, tree):
        queue = [tree["ROOT"]]
        depth = [0] * tree["number"]
        parent = [-1] * tree["number"]

        while (len(queue) > 0):
            current = queue.pop(0)
            for node in tree["edge"][current]:
                depth[node] = depth[current] + 1
                parent[node] = current
                queue.append(node)

        outarc = [len(tree["edge"][i]) for i in range(tree["number"])]        
        return tree["inarc"], outarc, depth, parent, tree["ROOT"]


class ConstituencyParsingProcess():
    def __call__(self, parse_str):
        constituency = self.extract(parse_str)
        return self.extract_bfs(constituency)

    def extract(self, parse_str):
        number = 0
        stack = []
        name = []
        edge = []
        tmp = ""

        for ch in parse_str:
            if ch == '(':
                # New node
                if (len(stack) > 0) and (name[stack[-1]] == ""):
                    name[stack[-1]] = tmp.strip()
                    tmp = ""

                stack.append(number)
                number +=1

                name.append("")
                edge.append([])
                if len(stack) > 1:
                    edge[stack[-2]].append(stack[-1])

            elif ch == ')':
                if tmp != "":
                    name[stack[-1]] = tmp.strip()
                    tmp = ""
                done = stack.pop()
                if edge[done] == []:
                    strs = name[done].split(" ")
                    name[done] = strs[0]
                    name.append(strs[1])
                    edge.append([])
                    edge[done].append(number)
                    number += 1
            else:
                tmp += ch

        data = {"number":number,
                "name":name,
                "edge":edge}
        return data

    def extract_bfs(self, data):
        queue = [0]
        depth = [0] * data["number"]
        pos = [""] * data["number"]
        # parent = [-1] * data["number"]

        while (len(queue) > 0):
            current = queue.pop(0)
            for node in data["edge"][current]:
                depth[node] = depth[current] + 1
                if data["edge"][node] != []:
                    queue.append(node)
                else:
                    pos[node] = data["name"][current]
        pos = [pos[i] for i in range(data["number"]) if data["edge"][i] == []]
        depth = [depth[i] for i in range(data["number"]) if data["edge"][i] == []]
        return pos, depth


class CorefParsingProcess():
    def __call__(self, mentions, include_entity_text=False):
        '''
            Collect coreferences sentence by sentence.
            Note that 
                1. sentence numbers are one-based w.r.t. the article doc.
                2. coreference indices are one-based w.r.t. their sentence.
            Convert coreference indices to zero-based for machine learning.
            Leave sentence numbers unchanged here.
        '''
        corefs = {
            "coref_head_index": [],
            "coref_index": [],
            "repr_mask": [],
            "entity_mask": [],
            "entity_span": [],
            "sent_num": [], # one based rather than zero based.
            "type": [],
            "number": [],
            "animacy": [],
            "gender": [],
        }
        if include_entity_text:
            corefs["entity_text"] = ""

        for m in mentions:
            if m["isRepresentativeMention"] and include_entity_text:
                corefs["entity_text"] = m["text"]
            # Get coref's indices w.r.t. the sentence.
            coref_index = list(range(m['startIndex']-1, m['endIndex']-1))
            coref_head_index = [m['headIndex']-1]
            # Create coref mask to identify antecedent from mentions.
            bit = [1] if m["isRepresentativeMention"] else [0]
            repr_mask = bit * len(coref_index)
            # Create word mask to identify between multi-word antecedent and mentions.
            entity_mask = [1] + [0]*(len(coref_index)-1)
            entity_span = [len(entity_mask)]
            if bit[0] == 1:
                corefs["coref_head_index"] = coref_head_index + corefs["coref_head_index"]
                corefs["coref_index"] = coref_index + corefs["coref_index"]
                corefs["repr_mask"] = repr_mask + corefs["repr_mask"]
                corefs["entity_mask"] = entity_mask + corefs["entity_mask"]
                corefs["entity_span"] = entity_span + corefs["entity_span"]
                corefs["sent_num"] = [m['sentNum']] + corefs["sent_num"]
                corefs["type"] = [m["type"]] + corefs["type"]
                corefs["number"] = [m["number"]] + corefs["number"]
                corefs["animacy"] = [m["animacy"]] + corefs["animacy"]
                corefs["gender"] = [m["gender"]] + corefs["gender"]
            else:
                corefs["coref_head_index"] += coref_head_index
                corefs["coref_index"] += coref_index
                corefs["repr_mask"] += repr_mask
                corefs["entity_mask"] += entity_mask
                corefs["entity_span"] += entity_span
                corefs["sent_num"] += [m['sentNum']]
                corefs["type"] += [m["type"]]
                corefs["number"] += [m["number"]]
                corefs["animacy"] += [m["animacy"]]
                corefs["gender"] += [m["gender"]]
        return corefs


class NerParsingProcess():
    def __init__(self):
        self.menfields = [
            "ner",
            "text",
            "tokenBegin", "tokenEnd",
            "docTokenBegin", "docTokenEnd"
        ]

    def __call__(self, sentence):
        # Check if any annotated word has actually more than one words"
        # Specify spliting by whitespace to avoid spliting on special chars,
        # e.g. 0845\xa06010128 which may be a phone number.
        # But be aware of a side-effect that contiguous whitespaces are counted as words.
        # # whats = [token["word"].split(" ") for token in sentence["tokens"]]
        every_word_counts = [len(token["word"].split(" ")) for token in sentence["tokens"]]
        total_word_counts = sum(every_word_counts)
        sentence_length = len(sentence["tokens"])
        if sentence_length != total_word_counts:
            raise ValueError("Found word containing subwords.")

        mentioned = []
        for mentions in sentence["entitymentions"]:
            men = {k: mentions[k] for k in self.menfields}
            mentioned.append(men)

        return {"sentenceindex": sentence["index"],
                "sentencelength": sentence_length,
                "entitymentions": mentioned}