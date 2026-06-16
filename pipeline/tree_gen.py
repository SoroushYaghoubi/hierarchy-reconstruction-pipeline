import random 
import argparse
from tree_io import write_closure, compute_closure

def generate_tree(depth, branch_fact, stop_prob=0.0, chain_prob=0.0):
    tree = {}
    counter = 0

    def add_node(parent, current_depth):
        nonlocal counter
        node = 'root' if counter == 0 else f'node_{counter}'
        counter += 1
        tree[node] = []
        if parent is not None:
            tree[parent].append(node)
        if current_depth < depth:
            # early stop — node becomes a leaf
            if random.random() < stop_prob:
                return
            # chain — node gets exactly one child regardless of branch_fact
            if random.random() < chain_prob:
                add_node(node, current_depth + 1)
                return
            for _ in range(branch_fact):
                add_node(node, current_depth + 1)

    add_node(None, 0)
    return tree

def pretty_print_tree(tree, node='root', p='', last=True):
    print(p + ('└── ' if last else '├── ') + node)
    for i, c in enumerate(tree[node]):
        pretty_print_tree(tree, c, p + ('    ' if last else '│   '), i == len(tree[node]) - 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, required=True)
    parser.add_argument('-depth', type=int, required=True)
    parser.add_argument('-branch_fact', type=int, required=True)
    parser.add_argument('-stop_prob', type=float, default=0.0)
    parser.add_argument('-chain_prob', type=float, default=0.0)
    parser.add_argument('-seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    tree = generate_tree(args.depth, args.branch_fact, args.stop_prob, args.chain_prob)
    pretty_print_tree(tree)
    closure = compute_closure(tree)
    write_closure(closure, args.name)
    print(f'Nodes: {len(tree)}, Closure edges: {len(closure)}')