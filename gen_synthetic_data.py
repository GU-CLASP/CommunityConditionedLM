import numpy as np
import random
import scipy.stats as stats
import itertools
import click

START = '<S>'
STOP = '</S>'

class MarkovChain:
    def __init__(self, transition_matrix, vocab):
        self.vocab = dict((s,i) for i,s in enumerate([START] + vocab))
        self.states = np.array(vocab + [STOP])
        assert (len(vocab) + 1, len(vocab) + 1) == transition_matrix.shape
        self.t_var = {s: transition_matrix[i] for i,s in enumerate([START] + vocab)}
        self.state = START
    def __iter__(self):
        self.state = START
        return self
    def __next__(self):
        self.state = np.random.choice(self.states, p=self.t_var[self.state])
        if self.state == STOP:
            raise StopIteration()
        return self.state

@click.command()
@click.argument('n_communities', type=int)
@click.argument('samples_per_community', type=int)
@click.option('--max-seq-len', default=100)
@click.option('--vocab-size', default=10)
@click.option('--alpha-start', default=0.01)
@click.option('--alpha-end', default=1.5)
def cli(n_communities, samples_per_community, max_seq_len, vocab_size, alpha_start, alpha_end):

    vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:vocab_size]
    t_dim = len(vocab) + 1

    for i, alpha in enumerate(np.linspace(alpha_start, alpha_end, n_communities)): # various Dirichlet hyperparameters 
        comm = f"COMM{i:02d}"
        print(f"Generating {comm}")
        t_mat = np.array([np.random.dirichlet([alpha] * t_dim) for _ in range(t_dim)])
        mc = MarkovChain(t_mat, vocab)
        with open('data/synthetic/' + comm + '.txt', 'w') as f:
            for i in range(samples_per_community):
                sample = ' '.join(itertools.islice(mc, max_seq_len))
                if sample:
                    f.write(sample + '\n')

if __name__ == '__main__':
    cli()


