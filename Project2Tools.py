import pandas as pd
# panadas provides an R-like DataFrame object, ideal for working with tabular data
import numpy as np
# numpy provides access to many mathematical operations, 
# and can perform vectorised operations on pandas dataseries objects (columns)
import string
# string provides us with a list of ascii lowercase letters (the english alphabet)

"""
CALCULATING BINARY ENTROPIES:

In order to see how efficient the huffman codes are, we need to find the theoretical minimum AWL; the binary entropy
"""

symbol_entropy_contribution = lambda prob: prob*np.log2(prob) if not np.isclose(prob,0,atol=1e-12) else 0
# the last part just handles the case that the probability is near-zero; we say 0*log(0) is 0

def calculate_binary_entropy(df,lang):
    """
    Inupt: df - cleaned dataframe, lang - language name (matching a datafrae column)
    Output: binary entropy of the specified language
    Notes: Pure Function; operations only performed on a copy of the dataframe, leaving the original unchanged
    """
    probs = df[:][[lang]]
    probs['entropy'] = df[lang].apply(symbol_entropy_contribution)
    return -probs['entropy'].sum()

"""
Having created a function to calculate the binary entropy of a language, we build up a dictionary
of the binary entropies for each language in our dataframe; 
"""

binary_entropies = lambda df: {lang:calculate_binary_entropy(df,lang) for lang in df.columns[1:]}

"""
CALCULATING HUFFMAN CODES:
"""

def priority_order(leaf_nodes):
    """
    Input: leaf_nodes - a dict containing symbols / symbol combinations and their corresponding probabilities
    Output: a list of symbols / symbol combinations, sorted by probability
    """
    return sorted(leaf_nodes, key=lambda x: leaf_nodes[x])

def recursive_huffman(leaf_nodes, d=2):
    """
    Input: leaf_nodes - a dict containing symbols / symbol combinations and their corresponding probabilities,
           d - the number of symbols available for encoding (default is 2 - binary encoding)
    Output: code - a dict with symbols / symbol combinations and their corresponding Huffman Codes
    Notes: 
            Base case - when only d elements in leaf_nodes; taken to be the root; assign them first d 'usable_symbols'
            (usable symbols being digits 0-9 then upper case letters - therefore max allowed d is 36)
            
            Method:
            > assign lfn a copy of the current leaf_nodes
            > find the d nodes with lowest probability (say A,B,...Final)
            > combine their symbols to create new node (say AB...Final) with asscociated probability of 
              prob(A) + prob(B) + ... + prob(Final)
            > delete A,B,...,Final
            > (recursion step) call recursive_huffman() with the newly updated leaf_nodes
            > once the base case has been reached, and results propagated back to this level,
              code_so_far is the longest HC assigned so far. 
            > assign the codes for the d least probable nodes to be this longest code so far, appended with first d 'usable_symbols'
    """
    usable_symbols = [str(i) for i in range(10)] + list(string.ascii_uppercase)
    assert 0<d<=36
    
    if len(leaf_nodes) == d:
        return dict(zip(leaf_nodes.keys(), usable_symbols[:d]))
    
    lfn = leaf_nodes.copy()
    least_prob_node_pair = priority_order(lfn)[:d]
    
    new_name = ''.join([least_prob_node_pair[i] for i in range(d)])
    lfn[new_name] = np.sum([lfn[least_prob_node_pair[i]] for i in range(d)])
    
    for i in range(d):
        del lfn[least_prob_node_pair[i]]
    
    code = recursive_huffman(lfn, d)
    d_least_prob_codes = ''.join([least_prob_node_pair[i] for i in range(d)])
    code_so_far = code[d_least_prob_codes]
    
    for i in range(d):
        code[least_prob_node_pair[i]] = code_so_far + usable_symbols[i]
    
    return code

def Huffman_encode_language(df, lang, d=2):
    """
    Input: df - the dataframe containing the probabilities of letter occurances for given language(s)
           lang - the language to generate huffman codes for (should be column of df)
           d - order of encoding (default is 2 - binary encoding)
    Output: codes - a dict of the Huffman Codes for each letter
            AWL - the average word length for the encoded letter, weighted by probability of occurance
    Notes:
           For some values of d, at the final recursion there are fewer than d nodes - this results in
           sub-optimal encoding as there exist fewer than d one-symbol encoded letters/letter-combinations.
           To avoid this, we add a few 'letters' to the alphabet (_, __, ...), all with probabilites of 0
           in order to ensure we will end up with d nodes at the final recursion.
    """
    leaf_nodes = pd.Series(df[lang].values,index=df['Letter']).to_dict()
    
    len_alphabet_needed_for_d_roots = d + (d-1)*int(np.ceil((26-d)/(d-1))) if d<26 else 26
    extra = len_alphabet_needed_for_d_roots - 26
    for i in range(extra):
        leaf_nodes['_'*(i+1)] = 0
        
    encoding = recursive_huffman(leaf_nodes, d)
    codes = {i:encoding[i] for i in string.ascii_lowercase}
    AWL = np.sum(len(codes[i])*leaf_nodes[i] for i in string.ascii_lowercase)
    return codes, AWL

def trivial_block_code():
    """
    Returns a binary block encoding for the lowercase ascii characters
    """
    return {val:bin(key)[2:].zfill(5) for key, val in enumerate(string.ascii_lowercase)}

def AWL(df, codes, lang):
    """
    Input: df - DataFrame contatining letters and corresponding probabilities,
           codes - a dict of Huffman Codes for lowercase ascii characters
           lang - language to use (must be column of df)
    Output: Average Word Length of the codes, weighted by probability
    """
    probs = pd.Series(df[lang].values,index=df['Letter']).to_dict()
    return np.sum(len(codes[i])*probs[i] for i in string.ascii_lowercase)

def generate_letter_frequencies_ad_hoc(text_sample_file, language):
    """
    Input: text_sample_file - the cleaned text sample to use to generate letter frequencies
           language - the name of the language used in the text sample
    Output: Dataframe containing 2 columns; Letter and <language> (<language> will contain
            probabilities for each letter)
    """
    counts = {i:0 for i in string.ascii_lowercase}
    sample_size = 0
    
    with open(text_sample_file, 'r') as f:
        sample = f.read().replace('\n','')
        sample_size = len(sample)
        for c in sample:
            counts[c]+=1
    
    frequencies = {i: counts[i]/sample_size for i in counts}
    data = {'Letter':list(frequencies.keys()), language:list(frequencies.values())}
    df = pd.DataFrame.from_dict(data)
    return df

