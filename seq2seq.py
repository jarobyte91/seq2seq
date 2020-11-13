import torch
import torch.nn as nn
from tqdm.notebook import tqdm

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, train_loader, dev_loader, epochs, lr, weight_decay, path = "Seq2Seq.pt"):
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = weight_decay)
        losses = []
        train_accuracies = []
        dev_accuracies = []
        non_pad_train_accuracies = []
        non_pad_dev_accuracies = []
        print("Training process:")
        
        print("epoch | train loss | train accuracy | dev accuracy | non-pad train accuracy | non-pad dev accuracy")

        for e in tqdm(range(epochs + 1)):
            for batch in iter(train_loader):
                x, y = batch
                # To use the semantics of the axis of the RNN correctly
                x = x.T
                y = y.T
                net_outputs = self(x, y[:-1, :])
                loss = criterion(net_outputs.transpose(1, 2), y[1:, :])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())

            if e % (epochs // 10) == 0:
                train_accuracy, non_pad_train_accuracy = self.compute_accuracy(train_loader)
                dev_accuracy, non_pad_dev_accuracy = self.compute_accuracy(dev_loader)
                train_accuracies.append(train_accuracy)
                dev_accuracies.append(dev_accuracy)
                print(f"{e:>5}{loss.item():^14.3f}{train_accuracy:^18.2f}{dev_accuracy:^14.2f}{non_pad_train_accuracy:^22.2f}{non_pad_dev_accuracy:^26.2f}")
    
        self.eval()
        torch.save(self.state_dict(), path)
        return losses, train_accuracies, dev_accuracies
    
    def compute_accuracy(self, loader):
        self.eval()
        hits = []
        sizes = []
        non_pad_hits = []
        non_pad_sizes = []
        with torch.no_grad():
            for batch in iter(loader):
                x, y = batch
                # To use the axis of the RNN correctly
                x = x.T
                y = y.T
                net_outputs = self(x, y[:-1, :])
                batch_hits = (net_outputs.argmax(-1) == y[1:, :])

                # To count the non-pad characters in the batch, y instead of y[:, 1:] is 
                # incorrect because the net is not supposed to give back the whole y.
                batch_non_pads = (y[1:, :] != 0)
                batch_non_pad_hits = batch_hits * batch_non_pads

                hits.append(batch_hits.sum().item())
                sizes.append(batch_hits.flatten().shape[0])
                non_pad_hits.append(batch_non_pad_hits.sum().item())
                non_pad_sizes.append(batch_non_pads.sum().item())
                
        self.train()
        
        # Useful for debugging
        
        # print(f"hits {sum(hits):,}")
        # print(f"sizes {sum(sizes):,}")
        # print(f"non_pad_hits {sum(non_pad_hits):,}")
        # print(f"non_pad_sizes {sum(non_pad_sizes):,}")

        # print()
        
        return 100 * sum(hits) / sum(sizes), 100 * sum(non_pad_hits) / sum(non_pad_sizes)
    
    def forward(self, X, Y):
        context = self.encoder(X)
        Y, Y_last = self.decoder(Y, context)
        Y = self.output_layer(Y)
        return Y
    
    def inference(self, X, predictions = 100):
        self.eval()
        context = self.encoder(X)
        output = []
        device = next(self.parameters()).device
        initial_token = torch.ones(1, X.shape[1]).long().to(device)
        output.append((initial_token, None))
        
        for i in range(predictions):
            last_token, last_hidden = output[-1]
            new_token, new_last_hidden = self.decoder(last_token, context, hx = last_hidden)
            new_token = self.output_layer(new_token).argmax(-1)
            output.append((new_token, new_last_hidden))
        output = torch.stack([token for token, hidden in output]).squeeze()    
        return output

    
class Seq2SeqEmbeddings(Seq2Seq):
    def __init__(self, encoder_vocab, decoder_vocab, 
                 encoder_embedding_dim, decoder_embedding_dim, 
                 encoder_hidden_dim, encoder_layers, 
                 decoder_hidden_dim, decoder_layers,
                 dropout):
        super().__init__()
        self.encoder_embeddings = nn.Embedding(len(encoder_vocab), encoder_embedding_dim)
        self.decoder_embeddings = nn.Embedding(len(decoder_vocab), decoder_embedding_dim) 
        
        self.encoder_rnn = nn.LSTM(encoder_embedding_dim, encoder_hidden_dim, encoder_layers, batch_first = True, 
                                   dropout = dropout, bidirectional = True)
        
        self.decoder_rnn = nn.LSTM(decoder_embedding_dim, decoder_hidden_dim, decoder_layers, batch_first = True, 
                                   dropout = dropout)
        
        self.encoder2decoder = nn.Linear(2 * 2 * encoder_layers * encoder_hidden_dim, 
                                         decoder_layers * decoder_hidden_dim)
        
        self.output_layer = nn.Linear(decoder_hidden_dim, len(decoder_vocab))
        self.dropout = nn.Dropout(dropout)
        print(f"This net has {sum([t.flatten().shape[0] for t in self.parameters()]):,} parameters.")

    def encoder(self, X):
        X = self.encoder_embeddings(X)
        X = X.relu()
        X = self.dropout(X)
        X, X_last = self.encoder_rnn(X)
        context = torch.cat([X_last[0].reshape((X.shape[0], -1)), 
                             X_last[1].reshape((X.shape[0], -1))], 
                            axis = 1)
        context = context.relu()
        context = self.dropout(context)
        context = self.encoder2decoder(context).reshape((self.decoder_rnn.num_layers, X.shape[0], -1))
        context = context.relu()
        context = self.dropout(context)
        return context
    
    def decoder(self, Y, context):
        Y = self.decoder_embeddings(Y)
        Y = Y.relu()
        Y = self.dropout(Y)
        Y, Y_last = self.decoder_rnn(Y, (context, context))
        Y = self.output_layer(Y)
        return Y, Y_last
    

class Seq2SeqEmbeddingsConcatFullTeacherForcing(Seq2Seq):
    def __init__(self, encoder_vocab, decoder_vocab, 
                 encoder_embedding_dim, decoder_embedding_dim, 
                 encoder_hidden_dim, encoder_layers, 
                 decoder_hidden_dim, decoder_layers,
                 dropout):
        super().__init__()
        self.encoder_embeddings = nn.Embedding(len(encoder_vocab), encoder_embedding_dim)
        self.decoder_embeddings = nn.Embedding(len(decoder_vocab), decoder_embedding_dim) 
        
        self.encoder_rnn = nn.GRU(encoder_embedding_dim, encoder_hidden_dim, encoder_layers, 
                                  dropout = dropout, bidirectional = True)
        
        self.decoder_rnn = nn.GRU(2 * encoder_hidden_dim * encoder_layers + decoder_embedding_dim, 
                                  decoder_hidden_dim, decoder_layers, 
                                  dropout = dropout)
        
        self.output_layer = nn.Linear(decoder_hidden_dim, len(decoder_vocab))
        self.dropout = nn.Dropout(dropout)
        print(f"This net has {sum([t.flatten().shape[0] for t in self.parameters()]):,} parameters.")

    def encoder(self, X):
        X = self.encoder_embeddings(X)
        X = X.relu()
        X = self.dropout(X)
        X, X_last = self.encoder_rnn(X)
        X_last = torch.cat((X_last[0], X_last[1]), axis = 1)
        return X_last
    
    def decoder(self, Y, context, hx = None):
        Y = self.decoder_embeddings(Y)
        context = context.unsqueeze(0).expand(Y.shape[0], Y.shape[1], context.shape[-1])
        Y = torch.cat((Y, context), axis = -1)
        Y = Y.relu()
        Y = self.dropout(Y)
        Y, Y_last = self.decoder_rnn(Y, hx)
        return Y, Y_last