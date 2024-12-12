from transformers import RobertaForSequenceClassification
from transformers import AdamW
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
import os
from d2l import torch as d2l

def read_data(data_dir, is_train):
    # Read the IMDb review dataset text sequences and labels.
    data = []
    labels = []
    
    dir = 'train' if is_train else 'test'
    
    for label in ['pos', 'neg']:
        # Directory path for each postive and negative
        dir_path = os.path.join(data_dir, dir, label)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                review = f.read()
                data.append(review)
                # Assign label = 1 for positive and 0 for negative
                labels.append(1 if label == 'pos' else 0)

    return data, labels

def load_data_imdb(batch_size, num_steps=500):
    
    data_dir = "./data/aclImdb"

    # if os.path.exists(data_dir):
    #     print("Dataset already exist, Skipping download")
    # else:
    #     d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
    #                            '01ada507287d82875905620988597833ad4e0903')
    #     data_dir = d2l.download_extract('aclImdb', 'aclImdb')

    train_data = read_data(data_dir, is_train=True)
    test_data = read_data(data_dir, is_train=False)

    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')

    vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

    train_features = torch.tensor([d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    
    train_labels = torch.tensor(train_data[1], dtype=torch.float32)
    test_labels = torch.tensor(test_data[1], dtype=torch.float32)

    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_iter, test_iter, vocab


def main():
  # Parameters
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_name = "roberta-base" # Can cahnge to other models of roberta
  max_len = 512 
  batch_size = 4
  epochs = 4
  learning_rate = 1e-6

  train_iter, test_iter, vocab = load_data_imdb(batch_size)

  # Load RoBERTa model with classification head (2 classes for sentiment)
  model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
  model.to(device)

  optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
  criterion = CrossEntropyLoss()

  for epoch in range(0, epochs):
      model.train()
      total_loss = 0.0
      total_num = 0
      correct = 0
      for x_train, y_train in train_iter:
          x_train, y_train = x_train.to(device), y_train.to(device)
          optimizer.zero_grad()
          out = model(x_train)
          loss = criterion(out, y_train.long())
          loss.backward()
          optimizer.step()

          total_loss += loss

          _, predicted = torch.max(out.data, 1)
          total_num += y_train.size(0)
          correct += (predicted == y_train).sum().item()

if __name__ == "__main__":
   main()