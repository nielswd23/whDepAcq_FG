library(ggplot2)
library(dplyr)

# Set working directory
setwd("~/Desktop/FG_project/whDepAcq_FG/lstm/")

### Read in the data ###
## no lex LSTM data
data <- read.csv("nolex_test_results.csv")

# ## lex LSTM data
# data1 <- read.csv("lex_test_results.csv") 
# data2 <- read.csv("lex_test_results2.csv") 
# data_100 <- read.csv("lex_test_results3.csv") 
# data_500 <- read.csv("lex_test_results_500.csv") 
# data <- bind_rows(data1, data2, data_100, data_500)


# Convert hyperparameters to factors for easy grouping in plots
data$embedding_dim <- as.factor(data$embedding_dim)
data$learning_rate <- as.factor(data$learning_rate)
data$batch_size <- as.factor(data$batch_size)

# add run_id column
data <- data %>%
  group_by(learning_rate, batch_size, embedding_dim) %>%
  mutate(run_id = as.factor(cur_group_id())) %>%
  ungroup()

data_clean <- na.omit(data)

min_loss_df <- data %>%
  group_by(run_id, learning_rate, batch_size, embedding_dim) %>%
  summarize(min_loss = min(held_out_loss),
            epochs = n(), .groups = "drop")

filter(min_loss_df, min_loss == min(min_loss))


# ## filtering nolex runs by the number of epochs
# # first group with 10 epochs
# data <- filter(data, row_number() < 397)
# # second group with 100 epochs
# data <- filter(data, row_number() > 396 & row_number() < 1609)
# # third group with 500 epochs
# data <- filter(data, row_number() > 1608)


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



# looking at only one lr
data %>% filter(learning_rate==0.0001) %>%
  ggplot(., aes(x = epoch, y = held_out_loss, group = run_id, color = embedding_dim)) +
    geom_line() +
    labs(title = "Held-Out Loss Curves by Hyperparameter Settings",
         x = "Epoch",
         y = "Held-Out Loss") +
    theme_classic() 
    # scale_color_discrete(name = "Settings",
    #                      labels = unique(paste("Embedding:", data$embedding_dim,
    #                                            " | Batch size:", data$batch_size)))



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
