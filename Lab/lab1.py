import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from powerlaw import Fit
import powerlaw

'''
Disclaimer: I acknowledge the use of Generative AI for explaining and helping me understand the 
documentation of how to use NetworkX.
'''

# Generate a random graph per instruction
def generate_random_graph(num_nodes=50, max_edges=4):
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
    
    degree_limit = {node: 0 for node in G.nodes()}  # Track total degree per node

    for node in G.nodes():
        num_edges = random.randint(0 if min(4, max_edges - degree_limit[node]) == 0 else 1, min(4, max_edges - degree_limit[node]))
        available_targets = [n for n in G.nodes() if n != node and degree_limit[n] < max_edges]
        if available_targets:
            targets = random.sample(available_targets, min(num_edges, len(available_targets)))
            for target in targets:
                if not G.has_edge(target, node):
                    G.add_edge(node, target)
                    degree_limit[node] += 1
                    degree_limit[target] += 1
    return G

'''
Visualize the graph (I still don't know how to store the locations of the nodes after removing nodes)
'''
def visualize_graph(G):
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.show()

    for node in G.nodes():
        print(f"Node {node} links to: {list(G.successors(node))}")

'''
Calculate degree centrality and average degree of the graph
'''
def calculating_measurements(G):
    degree_centrality = nx.degree_centrality(G)
    rounded_centrality = {node: round(centrality, 3) for node, centrality in degree_centrality.items()}
    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
    print("Degree Centrality:")
    for node, centrality in rounded_centrality.items():
        print(f"Node {node}: {centrality}")
    print("Average Degree:", avg_degree)
    print("-" * 50, "\n")
    return rounded_centrality, avg_degree

'''
Plot the degree distribution of the graph
'''
def plot_degree_distribution(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), align='left', rwidth=0.7)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution")
    plt.show()

    fit = Fit(degrees, discrete=True, xmin=1)
    fig = fit.plot_pdf(color='r', linewidth=2, label='Empirical Data')
    fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig, label='Power Law Fit')

    plt.title("Power Law Fit Analysis")
    plt.legend()
    plt.show()

    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f"Power-law and Exponential comparison: R = {R:3f}, p = {p:3f}")

'''
Remove a random node from the graph
'''
def remove_random_node(G):
    node_to_remove = random.choice(list(G.nodes()))
    G.remove_node(node_to_remove)
    return node_to_remove

'''
Remove nodes until only one node remains
'''
def remove_until_one(G):
    centrality_values = []
    while len(G.nodes()) > 1:
        visualize_graph(G)
        node_removed = remove_random_node(G)
        print("-" * 50)
        print(f"Removed node: {node_removed}")
        print("-" * 50)
        degree_centrality, _ = calculating_measurements(G)
        centrality_values.append(degree_centrality)
        plot_degree_distribution(G)
    return centrality_values

'''
Code for analyzing the karate club data
'''
def analyze_karate_club_data(filename='karate_club_coords.pkl'):
    with open(filename, 'rb') as f:
        karate_club_coords = pickle.load(f, encoding='latin1')

    G = nx.Graph()

    for node in sorted(karate_club_coords.keys(), key=int):
        G.add_node(node, pos=karate_club_coords[node])

    # Add edges based on the provided list of edges from Wikipedia (I dob't see the provided edges from Moodle)
    edges = [
        ("2", "1"), ("3", "1"), ("3", "2"), ("4", "1"), ("4", "2"), ("4", "3"), 
        ("5", "1"), ("6", "1"), ("7", "1"), ("7", "5"), ("7", "6"), ("8", "1"), 
        ("8", "2"), ("8", "3"), ("8", "4"), ("9", "1"), ("9", "3"), ("10", "3"),
        ("11", "1"), ("11", "5"), ("11", "6"), ("12", "1"), ("13", "1"), ("13", "4"), 
        ("14", "1"), ("14", "2"), ("14", "3"), ("14", "4"), ("17", "6"), ("17", "7"), 
        ("18", "1"), ("18", "2"), ("20", "1"), ("20", "2"), ("22", "1"), ("22", "2"), 
        ("26", "24"), ("26", "25"), ("28", "3"), ("28", "24"), ("28", "25"), ("29", "3"),
        ("30", "24"), ("30", "27"), ("31", "2"), ("31", "9"), ("32", "1"), ("32", "25"), 
        ("32", "26"), ("32", "29"), ("33", "3"), ("33", "9"), ("33", "15"), ("33", "16"), 
        ("33", "19"), ("33", "21"), ("33", "23"), ("33", "24"), ("33", "30"), ("33", "31"), 
        ("33", "32"), ("34", "9"), ("34", "10"), ("34", "14"), ("34", "15"), ("34", "16"), 
        ("34", "19"), ("34", "20"), ("34", "21"), ("34", "23"), ("34", "24"), ("34", "27"), 
        ("34", "28"), ("34", "29"), ("34", "30"), ("34", "31"), ("34", "32"), ("34", "33")
    ]

    # Add edges to the graph
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    nx.draw(G, pos=karate_club_coords, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.show()

    sorted_nodes = sorted(G.nodes(), key=int)
    print(G.nodes())
    adj_matrix = nx.adjacency_matrix(G).todense()
    print("Adjacency Matrix:\n", adj_matrix)
    
    degree_centrality = nx.degree_centrality(G)
    largest_component = max(nx.connected_components(G), key=len)
    largest_subgraph = G.subgraph(largest_component)
    smallest_component = min(nx.connected_components(G), key=len)
    smallest_subgraph = G.subgraph(smallest_component)
    
    print("Degree Centrality:")
    for node, centrality in degree_centrality.items():
        print(f"Node {node}: {round(centrality, 3)}")

    # d) Identify the largest and smallest components
    largest_component = max(nx.connected_components(G), key=len)
    largest_subgraph = G.subgraph(largest_component)

    smallest_component = min(nx.connected_components(G), key=len)
    smallest_subgraph = G.subgraph(smallest_component)

    print("\nLargest Component Size:", len(largest_component))
    print("Smallest Component Size:", len(smallest_component))

    print("Largest subgraph: ", largest_subgraph)

    # e) Draw the degree distribution of the largest component 
    plot_degree_distribution(largest_subgraph)

    # f) Compute the diameter of the entire graph and the largest component
    diameter = nx.diameter(G)
    largest_diameter = nx.diameter(largest_subgraph)
    
    print("\nGraph Diameter:", diameter)
    print("Largest Component Diameter:", largest_diameter)

if __name__ == "__main__":
    G = generate_random_graph()
    visualize_graph(G)
    calculating_measurements(G)
    plot_degree_distribution(G)
    remove_until_one(G)
    analyze_karate_club_data()
