#!/usr/bin/env python

import argparse
import json
import torch
import torch.nn as nn



def compute_gradients(model, inputs, target_class):
    """
    Compute gradients of the output with respect to the inputs for a specific target class.
    """
    model.eval()
    inputs.requires_grad = True

    outputs = model(inputs)
    loss = outputs[0, target_class]
    loss.backward()

    gradients = inputs.grad
    return gradients





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="")
    args = parser.parse_args()




