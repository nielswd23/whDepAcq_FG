import torch
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import pandas as pd 


def build_phrase_dict(data):
    # expects 2-dimensional data
    current_int = 2 # we reserve 1 for the start symbol and 0 for padding
    phrase_dict = {}
    for line in data:
        for phrase in line:
            if phrase not in phrase_dict:
                phrase_dict[phrase] = current_int
                current_int += 1

    phrase_dict['WUG'] = current_int # unseen character during testing

    return phrase_dict

def data_to_tensor(data, int_dict):
    input_data_as_int = []
    output_data_as_int = []
    for line in data:
        input_line_as_int = [1] # let 1 be our start symbol
        output_line_as_int = []
        for phrase in line:
            input_line_as_int.append(int_dict[phrase])
            output_line_as_int.append(int_dict[phrase])
        output_line_as_int.append(1) # also use 1 as end symbol
        input_data_as_int.append(torch.LongTensor(input_line_as_int))
        output_data_as_int.append(torch.LongTensor(output_line_as_int))

    input_as_tensor = torch.LongTensor(torch.nn.utils.rnn.pad_sequence(input_data_as_int)).T # convert to tensor and make batch first
    output_as_tensor = torch.LongTensor(torch.nn.utils.rnn.pad_sequence(output_data_as_int)).T
    mask = input_as_tensor != 0 # mask to remove padded entries
    return input_as_tensor, output_as_tensor, mask


def train_model(input, output, mask, print_every, lr, epochs, embedding_dim, batch_size, eval_index, **kwargs):
    # Model components
    embedding = torch.nn.Embedding(input.max() + 2, embedding_dim) # +2 because we are indexing from 0 with the input and an additional unseen WUG character for testing
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

    model.eval()
    with torch.no_grad():
        held_out_embeds = embedding(held_out_input)
        held_out_out, _ = model(held_out_embeds)
        held_out_loss = loss_func(held_out_out[held_out_mask], held_out_target[held_out_mask])

    d.append({"learning_rate":lr, "epoch":0, 
                "embedding_dim":embedding_dim, "batch_size":batch_size,
                "training_loss":"NA", 
                "held_out_loss":held_out_loss.item()})
    # measuring held_out loss before training to see the starting place


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
        d.append({"learning_rate":lr, "epoch":epoch+1, 
                  "embedding_dim":embedding_dim, "batch_size":batch_size,
                  "training_loss":avg_train_loss, 
                  "held_out_loss":held_out_loss.item()})

    return embedding, model, d



