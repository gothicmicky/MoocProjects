#!/usr/bin/env python
class Neuron:
    def __init__(self, inbound_neurons=[]):
        # Neuron from which this Neuron receives values
        self.inbound_neurons = inbound_neurons
        # Neuron to which this Neuron passes values
        self.outbound_neurons = []
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)


