import torch 
import pandas as pd 
import os 
from main import data_to_tensor


## helper functions to test a lstm output on the set of test stimuli
def check(tensor, emb): 
    test_emb = emb(tensor)
    test_output, _ = model(test_emb)
    probabilities = torch.nn.functional.softmax(test_output, dim=-1)
    logprobs = torch.nn.functional.log_softmax(test_output, dim=-1)

    return logprobs

# check(torch.LongTensor([1,2,3,0,0,0,0,0]), embedding)
# check(torch.LongTensor([1,2,3,4,2,3,0,0]), embedding)


def format_sequences(file_path, phrase_dict): # taking our flattened sequences and converting them into our lex and nolex format while adding WUGs where we have unseen lexical items
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            l = []
            for word in line.split():
                if word.split('_')[1] not in phrase_dict: # only need the lexical phrase_dict because there should not be any unseen phrase labels
                    l.append((word.split('_')[0], "WUG"))
                else:
                    l.append((word.split('_')[0], word.split('_')[1]))
            lines.append(l)

    seqs_nolex = []
    seqs_lex = []
    for seq in lines:
        seqs_nolex.append([tuple[0] for tuple in seq])
        seqs_lex.append([item for tup in seq for item in tup])

    return seqs_nolex, seqs_lex

def test_score(test_item, item_out, item_mask, emb, phrase_dict):
    model_output = check(test_item, emb)

    out_seq = item_out[item_mask]

    logprobs = []
    predicted_seq = []
    desired_seq = []
    phrase_dict['END'] = 1 # this helps the readability of the output sequences
    for i,n in enumerate(out_seq):
        logprobs.append(model_output[i][n])
        
        n_pred = torch.argmax(model_output[i])
        phrase_pred = next((k for k, v in phrase_dict.items() if v == n_pred), # looks for the value in the phrase dict to find the phrase label  
                           None)
        predicted_seq.append(phrase_pred)

        phrase_desired = next((k for k, v in phrase_dict.items() if v == n), # looks for the value in the phrase dict to find the phrase label  
                           None)
        desired_seq.append(phrase_desired)

    seq_logprob = torch.sum(torch.tensor(logprobs)) 
    len_fac_score = seq_logprob/len(logprobs)
    
    return len_fac_score.item(), seq_logprob.item(), desired_seq, predicted_seq


def create_test_df(path, filename, phrase_dict_lex, phrase_dict, mod_run, PhraseOnly=True): # PhraseOnly = True for no lex condition
    test_nolex, test_lex = format_sequences(path + filename, phrase_dict_lex)
    
    if PhraseOnly:
        test_in, test_out, test_mask = data_to_tensor(test_nolex, phrase_dict)

        d = []
        for i,item in enumerate(test_in):
            len_fac, logprob, seq, pred_seq = test_score(item, test_out[i], 
                                                        test_mask[i], embedding, 
                                                        phrase_dict)
            d.append({'len_fac':len_fac, 'logprob':logprob, 
                        'seq': seq, 'pred_seq': pred_seq, 
                        'test': filename[:-4], 'orig_stim':test_lex[i], 
                        'mod_run': mod_run})
    else: 
        test_in, test_out, test_mask = data_to_tensor(test_lex, phrase_dict_lex)

        d = []
        for i,item in enumerate(test_in):
            len_fac, logprob, seq, pred_seq = test_score(item, test_out[i], 
                                                        test_mask[i], embedding, 
                                                        phrase_dict_lex)
            d.append({'len_fac':len_fac, 'logprob':logprob, 
                        'seq': seq, 'pred_seq': pred_seq, 
                        'test': filename[:-4], 'orig_stim':test_lex[i],
                        'mod_run': mod_run})
        
    return d


## testing on a folder of saved models
test_d = []
for filename in os.listdir("./saved_models/Lex"):
    if filename == ".DS_Store":  # Skip .DS_Store
        continue

    model_run = filename[-5]

    ### load the model (and relevant components) ###
    print("./saved_models/Lex/" + filename)
    checkpoint = torch.load("./saved_models/Lex/" + filename, weights_only=False)
    model = torch.nn.LSTM(**checkpoint['lstm_args'])

    model.load_state_dict(checkpoint['model_state_dict'])

    embedding = checkpoint['embedding']

    nolex_phrase_dict = checkpoint['nolex_phrase_dict']
    lex_phrase_dict = checkpoint['lex_phrase_dict']



    ### compiling results ###

    # DeVilliers
    d_path = "./testing_stimuli/flattened/DeVilliers/"
    test_d.extend(create_test_df(d_path, "LongDistance.txt", 
                                lex_phrase_dict, nolex_phrase_dict, 
                                model_run, PhraseOnly=False))
    test_d.extend(create_test_df(d_path, "ShortDistance.txt", 
                                lex_phrase_dict, nolex_phrase_dict, 
                                model_run, PhraseOnly=False))

    # Sprouse
    s_path = "./testing_stimuli/flattened/Sprouse/"

    for filename in os.listdir(s_path):
        test_d.extend(create_test_df(s_path, filename, lex_phrase_dict, 
                                    nolex_phrase_dict, 
                                    model_run, PhraseOnly=False))
        
    # Liu
    l_path = "./testing_stimuli/flattened/Liu/"

    for filename in os.listdir(l_path):
        test_d.extend(create_test_df(l_path, filename, lex_phrase_dict, 
                                    nolex_phrase_dict,
                                    model_run, PhraseOnly=False))



test_df = pd.DataFrame(test_d)
test_df.to_csv('lex_models_test_results.csv', index=False)

