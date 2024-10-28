import torch
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
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

def train_model(input, output, mask, print_every, lr, epochs, embedding_dim, batch_size, **kwargs):
    # model
    embedding = torch.nn.Embedding(input.max() + 1, embedding_dim)
    model = torch.nn.LSTM(**kwargs)
    
    # optimizer
    params = list(embedding.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr)
    
    # loss function
    loss_func = torch.nn.CrossEntropyLoss()
    
    # Dataset and DataLoader
    dataset = TensorDataset(input, output, mask)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0  # Track loss for each epoch
        
        for batch_input, batch_output, batch_mask in data_loader:
            # Zero gradients for each batch
            optimizer.zero_grad()
            
            # Embed and pass through LSTM
            embeds = embedding(batch_input)  # Shape: [batch_size, seq_length, embedding_dim]
            out_hat, _ = model(embeds)  # Shape: [batch_size, seq_length, hidden_size]
            
            # Apply mask and calculate loss
            loss = loss_func(out_hat[batch_mask], batch_output[batch_mask])
            total_loss += loss.item()
            
            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
        
        # Print average loss for the epoch
        if (epoch + 1) % print_every == 0:
            print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(data_loader)}")

    return embedding, model

    # # model
    # embedding = torch.nn.Embedding(input.max()+1, embedding_dim)
    # model = torch.nn.LSTM(**kwargs)
    # # optimizer
    # params = list(embedding.parameters()) + list(model.parameters())
    # optimizer = torch.optim.Adam(params, lr)
    # # loss function
    # loss_func = torch.nn.CrossEntropyLoss()

    # for i in range(epochs):
    #     optimizer.zero_grad()
    #     embeds = embedding(input)
    #     out_hat, _ = model(embeds) # batches x seq x hidden_sz = number of possible outputs
    #     #breakpoint()
    #     loss = loss_func(out_hat[mask], output[mask])

    #     #if (i+1 % print_every) == 0:
    #     print("Epoch: {}: Loss: {}".format(i+1, loss.item()))

    #     loss.backward()
    #     optimizer.step()

    # return (embedding, model)


file_path = 'flat_paths.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()
    lines = [line.split() for line in lines][:-1]
    lines = [[(word.split('_')[0], word.split('_')[1]) for word in line] for line in lines]


# phrase_dict = build_phrase_dict(lines, phrase_only=False) # builds a mapping from phrases to integers
# input, output, mask = data_to_tensor(lines, phrase_dict, phrase_only=False) # convert to tensors for training

phrase_dict = build_phrase_dict(lines) # builds a mapping from phrases to integers
input, output, mask = data_to_tensor(lines, phrase_dict) # convert to tensors for training


## calculating the 80/20 train test cutoff
index = round(len(input) * 0.8)


#breakpoint()

embedding_dim = 7
# currently hidden_size has to be 7. 344 for Label-lex path. It is the number of unique items (so len of phrase dict) + 2 for our 0 and 1
lstm_args = {'input_size': embedding_dim, 'hidden_size': 7, 'batch_first': True}

embedding, model = train_model(input[:index], output[:index], mask[:index], 10, .01, 100, embedding_dim, 10, **lstm_args) 


print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

### save the model ###
torch.save({'model_state_dict':model.state_dict(), 
            'embedding':embedding,
            'lstm_args':lstm_args
            }, "phraseLSTM.tar")




### Testing ###
embeds = embedding(input[index:])

model.eval()
with torch.no_grad():
    test_out, _ = model(embeds)

test_target = output[index:]
test_mask = mask[index:]

targ_w_mask = test_target[test_mask]
model_out_w_mask = test_out[test_mask]

# with open('tmp.txt', 'w') as file:
#     file.write(f"test_out:\n{test_out[:10]}\n\n")
#     file.write(f"test_target:\n{test_target[:10]}\n\n")
#     file.write(f"test_mask:\n{test_mask[:10]}\n\n")
#     file.write(f"Target with mask:\n{targ_w_mask[:10]}\n\n")
#     file.write(f"Model out with mask:\n{model_out_w_mask[:10]}\n\n")

loss_fct = torch.nn.CrossEntropyLoss()
held_out_loss = loss_fct(model_out_w_mask, targ_w_mask)
# why am i getting this error?? RuntimeError: Boolean value of Tensor with more than one value is ambiguous

# held_out_loss = torch.nn.CrossEntropyLoss(test_out, test_target)

print(held_out_loss.item())







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

