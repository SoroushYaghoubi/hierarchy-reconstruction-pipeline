import random 
import argparse
from tree_io import write_tree, write_closure

def generate_tree(depth, branch_fact, depth_var=0, branch_var=0):
    tree = {}
    counter = 0

    def add_node(parent, current_depth, local_max_depth):
        nonlocal counter
        node = 'root' if counter == 0 else f'node_{counter}'
        counter += 1
        tree[node] = []
        if parent is not None:
            tree[parent].append(node)
        if current_depth < local_max_depth:
            n_children = max(1, branch_fact + random.randint(-branch_var, branch_var))
            for _ in range(n_children):
                child_max_depth = depth + random.randint(-depth_var, depth_var)
                add_node(node, current_depth + 1, child_max_depth)

    add_node(None, 0, depth)
    return tree

def pretty_print_tree(tree, node='root', p='', last=True):
    print(p + ('└── ' if last else '├── ') + node)
    for i, c in enumerate(tree[node]):
        pretty_print_tree(tree, c, p + ('    ' if last else '│   '), i == len(tree[node]) - 1)

def compute_closure(tree):
    parent_map = {}
    for parent, children in tree.items():
        for child in children:
            parent_map[child] = parent

    edges = []
    for node in tree:
        current = node
        while current in parent_map:
            current = parent_map[current]
            edges.append((node, current))

    return edges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, required=True)
    parser.add_argument('-depth', type=int, required=True)
    parser.add_argument('-branch_fact', type=int, required=True)
    parser.add_argument('-depth_var', type=int, default=0)
    parser.add_argument('-branch_var', type=int, default=0)
    args = parser.parse_args()

    tree = generate_tree(args.depth, args.branch_fact, args.depth_var, args.branch_var)
    pretty_print_tree(tree)
    closure = compute_closure(tree)
    write_closure(closure, args.name + '_original')
    write_tree(tree, args.name + '_original')
    print(f'Nodes: {len(tree)}, Closure edges: {len(closure)}')