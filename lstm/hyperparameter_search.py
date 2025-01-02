from itertools import product
import pandas as pd 
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
index = round(len(lines)*0.8)

phrase_dict = build_phrase_dict(seqs_lex) # builds a mapping from phrases to integers
input, output, mask = data_to_tensor(seqs_lex, phrase_dict) # convert to tensors for training



### training models ###
hyperparameters = {
    'embedding_dim': [10, 20, 100, 300],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'batch_size': [50, 300, 700],
    'epochs': [500]
}

# Generate all combinations of hyperparameters
param_combinations = list(product(*hyperparameters.values()))

# train on different hyperparameter values 
results = []  # To store results of each run

# Loop over each combination of hyperparameters
for params in param_combinations:
    # Map each value to the corresponding hyperparameter key
    param_dict = dict(zip(hyperparameters.keys(), params))
    
    # Unpack the parameters to call train_model
    embedding_dim = param_dict['embedding_dim']
    lr = param_dict['learning_rate']
    batch_size = param_dict['batch_size']
    epochs = param_dict['epochs']
    
    # Print the current combination (for tracking)
    print(f"Testing combination: {param_dict}")

    lstm_args = {'input_size': embedding_dim, 'hidden_size': len(phrase_dict) + 2, 'batch_first': True}
    # hidden_size will be length of phrase_dict + 2 (the number of unique words including unseen WUG plus start and padding symbol)

    # Run train_model and capture the final embedding and model
    embedding, model, result = train_model(
        input=input,  # Your input tensor
        output=output,  # Your output tensor
        mask=mask,  # Mask tensor
        print_every=10, 
        lr=lr, 
        epochs=epochs, 
        embedding_dim=embedding_dim, 
        batch_size=batch_size, 
        eval_index=index,
        **lstm_args
    )

    # append this run's results 
    results.extend(result)

print(len(results))
print(results[0])
df = pd.DataFrame(results)
# df.to_csv('nolex_test_results.csv', index=False)