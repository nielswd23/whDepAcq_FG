library(ggplot2)
library(dplyr)

# Set working directory
setwd("~/Desktop/FG_project/whDepAcq_FG/lstm/")

# Read in the data
data <- read.csv("test_results.csv") 

# Convert hyperparameters to factors for easy grouping in plots
data$embedding_dim <- as.factor(data$embedding_dim)
data$learning_rate <- as.factor(data$learning_rate)
data$batch_size <- as.factor(data$batch_size)

# add run_id column
data <- data %>%
  group_by(learning_rate, batch_size, embedding_dim) %>%
  mutate(run_id = as.factor(cur_group_id())) %>%
  ungroup()



### Plot held-out loss curves ###
## all hyper parameters together ##
ggplot(data, aes(x = epoch, y = held_out_loss, color = interaction(learning_rate, embedding_dim, batch_size))) +
  geom_line() +
  labs(title = "Held-Out Loss Curves by Hyperparameter Settings",
       x = "Epoch",
       y = "Held-Out Loss",
       color = "Hyperparameter Settings") +
  theme_classic() +
  theme(legend.position = "bottom") +
  scale_color_discrete(name = "Settings",
                       labels = unique(paste("Learning Rate:", data$learning_rate,
                                             "| Embedding:", data$embedding_dim,
                                             " | Batch size:", data$batch_size)))

ggplot(data, aes(x = epoch, y = held_out_loss, group = run_id)) +
  geom_line() +
  labs(title = "Held-Out Loss Curves by Hyperparameter Settings",
       x = "Epoch",
       y = "Held-Out Loss") +
  theme_classic() +
  theme(legend.position = "bottom") +
  facet_wrap(vars(learning_rate, batch_size, embedding_dim), labeller = label_both)





## filtering out one dimension ##
ggplot(data, aes(x = epoch, y = held_out_loss, group = run_id, color = learning_rate)) +
  geom_line() +
  labs(title = "Held-Out Loss Curves by Hyperparameter Settings",
       x = "Epoch",
       y = "Held-Out Loss") +
  theme_classic() +
  theme(legend.position = "bottom") 

ggplot(data, aes(x = epoch, y = held_out_loss, group = run_id, color = batch_size)) +
  geom_line() +
  labs(title = "Held-Out Loss Curves by Hyperparameter Settings",
       x = "Epoch",
       y = "Held-Out Loss") +
  theme_classic() +
  theme(legend.position = "bottom") 

ggplot(data, aes(x = epoch, y = held_out_loss, group = run_id, color = embedding_dim)) +
  geom_line() +
  labs(title = "Held-Out Loss Curves by Hyperparameter Settings",
       x = "Epoch",
       y = "Held-Out Loss") +
  theme_classic() +
  theme(legend.position = "bottom") 







# facet lr
ggplot(data, aes(x = epoch, y = held_out_loss, color = interaction(embedding_dim, batch_size))) +
  geom_line() +
  labs(title = "Held-Out Loss Curves by Hyperparameter Settings",
       x = "Epoch",
       y = "Held-Out Loss",
       color = "Hyperparameter Settings") +
  facet_wrap(~learning_rate) +
  theme_classic() +
  theme(legend.position = "bottom") +
  scale_color_discrete(name = "Settings",
                       labels = unique(paste("Embedding:", data$embedding_dim,
                                             "| Batch:", data$batch_size)))

# facet embedding
ggplot(data, aes(x = epoch, y = held_out_loss, color = interaction(learning_rate, batch_size))) +
  geom_line() +
  labs(title = "Held-Out Loss Curves by Hyperparameter Settings",
       x = "Epoch",
       y = "Held-Out Loss",
       color = "Hyperparameter Settings") +
  facet_wrap(~embedding_dim) +
  theme_classic() +
  theme(legend.position = "bottom") +
  scale_color_discrete(name = "Settings",
                       labels = unique(paste("Learning Rate:", data$learning_rate,
                                             "| Batch:", data$batch_size)))

# facet batch size 
ggplot(data, aes(x = epoch, y = held_out_loss, color = interaction(learning_rate, embedding_dim))) +
  geom_line() +
  labs(title = "Held-Out Loss Curves by Hyperparameter Settings",
       x = "Epoch",
       y = "Held-Out Loss",
       color = "Hyperparameter Settings") +
  facet_wrap(~batch_size) +
  theme_classic() +
  theme(legend.position = "bottom") +
  scale_color_discrete(name = "Settings",
                       labels = unique(paste("Learning Rate:", data$learning_rate,
                                             "| Embedding:", data$embedding_dim)))
