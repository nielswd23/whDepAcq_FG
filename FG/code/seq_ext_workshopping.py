from nltk.tree import ParentedTree
from nltk.corpus import BracketParseCorpusReader
import re 


corpus_root = "../CHILDESTreebank"

# # brown eve 
# file_BE = "brown_eve_animacy_theta.parsed"
# corpus = BracketParseCorpusReader(corpus_root, file_BE)
# parsed_sents = corpus.parsed_sents() 

# ## find bad trees
# heights = []
# for tree in parsed_sents:
#     heights.append(tree.height())


# valian
file_val = "valian_animacy_theta.parsed"
corpus = BracketParseCorpusReader(corpus_root, file_val)
parsed_sents = corpus.parsed_sents() 


# ## find bad trees
# heights = []
# for tree in parsed_sents:
#     heights.append(tree.height())




### pulling out unique nodes 
unique_nodes = []
for tree in parsed_sents:
    for subtree in tree.subtrees():
        node = subtree.label()
        if node not in unique_nodes:
            unique_nodes.append(node)

unique_nodes.sort()

# struggling to deal with NP-<ANIM>-< still. will have to find the tree in python




# new plan with this extraction. Going to compile all of the 
# wh trace trees. going to fix any of the "bad tree detected" nonsense.
# going to pull out wh traces.
# then I'll clean the trees to have a standard labeling.
# then this will give me a nice corpus of child directed wh-dep trees.
# then, with this collected, i'll pull out syntactic paths and have a 
# dictionary of all the conversions that I want to do from old formatting 
# nodes to new ones 