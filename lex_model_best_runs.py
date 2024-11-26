import torch
from main import build_phrase_dict, data_to_tensor, train_model

### input ###
file_path = 'flat_paths.txt'

with open(file_path, 'r') as file:
    lines = [[(word.split('_')[0], word.split('_')[1]) for word in line.split()] for line in file]

seqs_nolex = []
seqs_lex = []
for seq in lines:
    seqs_nolex.append([tuple[0] for tuple in seq])
    seqs_lex.append([item for tup in seq for item in tup])



# training and test split
index = round(len(seqs_nolex)*0.8)


phrase_dict = build_phrase_dict(seqs_nolex) # builds a mapping from phrases to integers
phrase_dict_lex = build_phrase_dict(seqs_lex) # dict with lexical items
input, output, mask = data_to_tensor(seqs_lex, phrase_dict_lex) # convert to tensors for training



# training 
lr = 0.0001
epochs = 500
batch_size = 300
embedding_dim = 300
lstm_args = {'input_size': embedding_dim, 'hidden_size': len(phrase_dict) + 2,
            'batch_first': True} # hidden_size will be length of phrase_dict + 2 (the number of unique words with the unseen WUG plus 0 padding symbol and 1 start)


# 10 runs of the model
for n in range(10):
    embedding, model, result = train_model(
        input=input,  # Your input tensor
        output=output,  # Your output tensor
        mask=mask,  # Mask tensor
        print_every=1, 
        lr=lr, 
        epochs=epochs, 
        embedding_dim=embedding_dim, 
        batch_size=batch_size, 
        eval_index=index,
        **lstm_args
    )

    ### save the model ###
    torch.save({'model_state_dict':model.state_dict(), 
                'embedding':embedding,
                'lstm_args':lstm_args,
                'nolex_phrase_dict': phrase_dict,
                'lex_phrase_dict': phrase_dict_lex
                }, f"BestLexLSTM{n}.tar")