"""A common message-passing backbone and cross-channel structural readout."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ControlledClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, classes, config):
        super().__init__()
        hidden = config['hidden_dim']
        self.input = nn.Linear(node_dim, hidden)
        self.messages = nn.ModuleList([
            nn.Linear(hidden + edge_dim, hidden)
            for _ in range(config['layers'])
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(config['layers'])
        ])
        self.epsilon = nn.Parameter(torch.zeros(config['layers']))
        self.structural = nn.Sequential(
            nn.Linear(27, hidden), nn.ReLU(), nn.Dropout(config['dropout']),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(),
            nn.Dropout(config['dropout']), nn.Linear(hidden, classes),
        )
        self.checkpoint_chunks = config['checkpoint_structural_chunks']

    def forward(self, x, edge_index, edge_attr, structural_chunks, total_triples):
        # Native graph attributes pass through the message-passing backbone.
        node_states = torch.relu(self.input(x))
        source, target = edge_index
        for level, (message, update) in enumerate(zip(self.messages, self.updates)):
            message_values = torch.relu(message(torch.cat((node_states[source], edge_attr), 1)))
            aggregated = torch.zeros_like(node_states).index_add(0, target, message_values)
            node_states = update((1 + self.epsilon[level]) * node_states + aggregated)

        # Mix each complete structural row before averaging over triples.
        structural_sum = None
        for block in structural_chunks:
            block_values = (
                checkpoint(self.structural, block, use_reentrant=False)
                if self.training and self.checkpoint_chunks
                else self.structural(block)
            )
            block_sum = block_values.sum(0)
            structural_sum = block_sum if structural_sum is None else structural_sum + block_sum
        if structural_sum is None or total_triples <= 0:
            raise ValueError('Nonempty full ordered-triple features are required')
        structural_sum = structural_sum / total_triples
        return self.output(torch.cat((node_states.mean(0), node_states.sum(0), structural_sum), 0))
