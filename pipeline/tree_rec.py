import os
import torch
import argparse
from tree_io import write_tree

ARTEFACTS = os.path.join(os.path.dirname(__file__), 'artefacts')

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', help='Checkpoint stem (no path or extension)')
args = parser.parse_args()

checkpoint_path = os.path.join(ARTEFACTS, args.checkpoint + '.pth.best')
chkpnt = torch.load(checkpoint_path, map_location='cpu')
embeddings = chkpnt['embeddings']
objects = chkpnt['objects']

print(f"Nodes: {len(objects)}")

N = len(objects)
norms = embeddings.norm(dim=1)

def angular_dist(u, v):
    return 1 - torch.dot(u, v) / (u.norm() * v.norm())

parent_of = {}
for i, node in enumerate(objects):
    if node == 'root':
        continue
    candidates = [j for j in range(N) if norms[j] < norms[i]]
    if not candidates:
        parent_of[node] = 'root'
        continue
    parent_of[node] = objects[min(candidates, key=lambda j: angular_dist(embeddings[i], embeddings[j]).item())]

recovered_tree = {node: [] for node in objects}
for node, parent in parent_of.items():
    recovered_tree[parent].append(node)

print(f"Root children: {recovered_tree['root']}")

write_tree(recovered_tree, args.checkpoint + '_recovered')