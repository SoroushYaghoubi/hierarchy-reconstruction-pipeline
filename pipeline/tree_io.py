# pipeline/tree_io.py
import os
import csv
import json

ARTEFACTS = os.path.join(os.path.dirname(__file__), 'artefacts')

def read_tree(stem):
    path = os.path.join(ARTEFACTS, stem + '.json')
    with open(path, 'r') as f:
        return json.load(f)

def write_tree(tree, stem):
    path = os.path.join(ARTEFACTS, stem + '.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(tree, f, indent=2)

def read_closure(stem):
    path = os.path.join(ARTEFACTS, stem + '.csv')
    edges = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            edges.append((row[0], row[1]))
    return edges

def write_closure(edges, stem):
    path = os.path.join(ARTEFACTS, stem + '.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id1', 'id2', 'weight'])
        for child, ancestor in edges:
            writer.writerow([child, ancestor, 1])