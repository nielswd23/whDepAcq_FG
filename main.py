import torch
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import pandas as pd 
#from collections import


def build_phrase_dict(data, phrase_only=True):
    # expects 2-dimensional data
    current_int = 2 # we reserve 1 for the start symbol and 0 for padding
    phrase_dict = {}
    if phrase_only:
        for line in data:
            for phrase in line:
                if phrase[0] not in phrase_dict:
                    phrase_dict[phrase[0]] = current_int
                    current_int += 1
    else:
        for line in data:
            for phrase in line:
                if phrase not in phrase_dict:
                    phrase_dict[phrase] = current_int
                    current_int += 1

    return phrase_dict


def data_to_tensor(data, int_dict, phrase_only=True):
    if phrase_only:
        input_data_as_int = []
        output_data_as_int = []
        for line in data:
            input_line_as_int = [1] # let 1 be our start symbol
            output_line_as_int = []
            for phrase in line:
                input_line_as_int.append(int_dict[phrase[0]])
                output_line_as_int.append(int_dict[phrase[0]])
            output_line_as_int.append(1) # also use 1 as end symbol
            input_data_as_int.append(torch.LongTensor(input_line_as_int))
            output_data_as_int.append(torch.LongTensor(output_line_as_int))
    else:
        input_data_as_int = []
        output_data_as_int = []
        for line in data:
            input_line_as_int = [1]  # let 1 be our start symbol
            output_line_as_int = []
            for phrase in line:
                input_line_as_int.append(int_dict[phrase])
                output_line_as_int.append(int_dict[phrase])
            output_line_as_int.append(1)
            input_data_as_int.append(torch.LongTensor(input_line_as_int))
            output_data_as_int.append(torch.LongTensor(output_line_as_int))

    input_as_tensor = torch.LongTensor(torch.nn.utils.rnn.pad_sequence(input_data_as_int)).T # convert to tensor and make batch first
    output_as_tensor = torch.LongTensor(torch.nn.utils.rnn.pad_sequence(output_data_as_int)).T
    mask = input_as_tensor != 0 # mask to remove padded entries
    return input_as_tensor, output_as_tensor, mask


def train_model(input, output, mask, print_every, lr, epochs, embedding_dim, batch_size, eval_index, **kwargs):
    # Model components
    embedding = torch.nn.Embedding(input.max() + 1, embedding_dim)
    model = torch.nn.LSTM(**kwargs)
    
    # Optimizer
    params = list(embedding.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr)
    
    # Loss function
    loss_func = torch.nn.CrossEntropyLoss()
    
    # Dataset and DataLoader for training data
    dataset = TensorDataset(input[:eval_index], output[:eval_index], mask[:eval_index])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Validation (held-out) data setup
    held_out_input = input[eval_index:]
    held_out_target = output[eval_index:]
    held_out_mask = mask[eval_index:]
    

    # tracking training loss and held out loss
    d = []
    # l_train_loss = []
    # l_held_out_loss = []
    for epoch in range(epochs):
        total_train_loss = 0  # Track training loss for each epoch
        
        # Training phase
        model.train()
        for batch_input, batch_output, batch_mask in data_loader:
            # Zero gradients for each batch
            optimizer.zero_grad()
            
            # Embed and pass through LSTM
            embeds = embedding(batch_input)  # Shape: [batch_size, seq_length, embedding_dim]
            out_hat, _ = model(embeds)  # Shape: [batch_size, seq_length, hidden_size]
            
            # Apply mask and calculate training loss
            loss = loss_func(out_hat[batch_mask], batch_output[batch_mask])
            total_train_loss += loss.item()
            
            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
        
        # Validation phase (held-out loss calculation)
        model.eval()
        with torch.no_grad():
            held_out_embeds = embedding(held_out_input)
            held_out_out, _ = model(held_out_embeds)
            held_out_loss = loss_func(held_out_out[held_out_mask], held_out_target[held_out_mask])
        
        # Print average training loss and held-out loss for the epoch
        avg_train_loss = total_train_loss / len(data_loader)
        if (epoch + 1) % print_every == 0:
            print(f"Epoch: {epoch + 1}, Training Loss: {avg_train_loss}, Held-Out Loss: {held_out_loss.item()}")
        
        # l_train_loss.append(avg_train_loss)
        # l_held_out_loss.append(held_out_loss.item())
        d.append({"learning_rate":lr, "epoch":epoch, 
                  "embedding_dim":embedding_dim, "batch_size":batch_size,
                  "training_loss":avg_train_loss, 
                  "held_out_loss":held_out_loss.item()})
        ## TODO create a pandas df and append each row with parameters values and the loss item 

    return embedding, model, d


### input ###
file_path = 'flat_paths.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()
    lines = [line.split() for line in lines][:-1]
    lines = [[(word.split('_')[0], word.split('_')[1]) for word in line] for line in lines]


# training and test split
index = round(len(lines)*0.8)

# phrase_dict = build_phrase_dict(lines, phrase_only=False) # builds a mapping from phrases to integers
# input, output, mask = data_to_tensor(lines, phrase_dict, phrase_only=False) # convert to tensors for training

phrase_dict = build_phrase_dict(lines) # builds a mapping from phrases to integers
input, output, mask = data_to_tensor(lines, phrase_dict) # convert to tensors for training



### training models ###
# Define hyperparameters and their possible values
# hyperparameters = {
#     'embedding_dim': [3, 7, 10, 20],
#     'learning_rate': [0.001, 0.01, 0.1],
#     'batch_size': [2, 4, 8, 16, 50, 150],
#     'epochs': [10, 50, 100, 300]
# }
hyperparameters = {
    'embedding_dim': [3, 7, 20],
    'learning_rate': [0.001, 0.1],
    'batch_size': [50, 300],
    'epochs': [10]
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

    lstm_args = {'input_size': embedding_dim, 'hidden_size': 7, 'batch_first': True}

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
df.to_csv('test_results.csv', index=False)

print(df)

# # Display or analyze results
# with open("hyperparam_test.txt", 'w') as file:
#     for item in results:
#         file.write(f"{result}\n")










# #### test
# embedding_dim = 7
# lr = 0.01
# batch_size = 100
# epochs = 10

# lstm_args = {'input_size': embedding_dim, 'hidden_size': 7, 'batch_first': True}

# # Run train_model and capture the final embedding and model
# embedding, model, result = train_model(
#     input=input,  # Your input tensor
#     output=output,  # Your output tensor
#     mask=mask,  # Mask tensor
#     print_every=10, 
#     lr=lr, 
#     epochs=epochs, 
#     embedding_dim=embedding_dim, 
#     batch_size=batch_size, 
#     eval_index=index,
#     **lstm_args
# )






#  # testing on held out data
# embeds = embedding(input[index:])

# model.eval()
# with torch.no_grad():
#     test_out, _ = model(embeds)

# test_target = output[index:]
# test_mask = mask[index:]

# loss_fct = torch.nn.CrossEntropyLoss()
# held_out_loss = loss_fct(test_out[test_mask], test_target[test_mask])

# print(held_out_loss)




# embedding_dim = 7
# # currently hidden_size has to be 7. 344 for Label-lex path. It is the number of unique items (so len of phrase dict) + 2 for our 0 and 1
# lstm_args = {'input_size': embedding_dim, 'hidden_size': 7, 'batch_first': True}

# embedding, model = train_model(input[:index], output[:index], mask[:index], 10, .01, 100, embedding_dim, 10, **lstm_args) 


# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())




# ### save the model ###
# torch.save({'model_state_dict':model.state_dict(), 
#             'embedding':embedding,
#             'lstm_args':lstm_args
#             }, "phraseLSTM.tar")










# ### Testing ###
# embeds = embedding(input[index:])

# model.eval()
# with torch.no_grad():
#     test_out, _ = model(embeds)

# test_target = output[index:]
# test_mask = mask[index:]

# targ_w_mask = test_target[test_mask]
# model_out_w_mask = test_out[test_mask]

# # with open('tmp.txt', 'w') as file:
# #     file.write(f"test_out:\n{test_out[:10]}\n\n")
# #     file.write(f"test_target:\n{test_target[:10]}\n\n")
# #     file.write(f"test_mask:\n{test_mask[:10]}\n\n")
# #     file.write(f"Target with mask:\n{targ_w_mask[:10]}\n\n")
# #     file.write(f"Model out with mask:\n{model_out_w_mask[:10]}\n\n")

# loss_fct = torch.nn.CrossEntropyLoss()
# held_out_loss = loss_fct(model_out_w_mask, targ_w_mask)
# # why am i getting this error?? RuntimeError: Boolean value of Tensor with more than one value is ambiguous

# # held_out_loss = torch.nn.CrossEntropyLoss(test_out, test_target)

# print(held_out_loss.item())







# #### Testing how to pass input through the trained model ####
# # mapping the input to the embedding used to train the model
# embeds = embedding(input[:6])

# print(input[:6])
# # print(embeds)

# model.eval()
# with torch.no_grad():
#     test_out, _ = model(embeds)
# # print(test_out)

# probabilities = torch.nn.functional.softmax(test_out, dim=-1)
# predicted_digits = torch.argmax(probabilities, dim=-1)
# # print(predicted_digits)

# with open('test_out.txt', 'w') as file:
#     file.write(f"Input:\n{input[:6]}\n\n")
#     file.write(f"Output:\n{output[:6]}\n\n")
#     file.write(f"Mask:\n{mask[:6]}\n\n")
#     file.write(f"Input embeds:\n{embeds}\n\n")
#     file.write(f"Probabilities:\n{probabilities}\n\n")
#     file.write(f"Predicted seqs:\n{predicted_digits}")

