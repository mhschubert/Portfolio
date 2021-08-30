# Code by 29AjayKumar; adapted by myself

import sys
import numpy as np
# Function to find index of max-weight
# vertex from set of unvisited vertices
def findMaxVertex(V, visited, weights):
    # Stores the index of max-weight vertex
    # from set of unvisited vertices
    index = -1

    # Stores the maximum weight from
    # the set of unvisited vertices
    maxW = -sys.maxsize

    # Iterate over all possible
    # Nodes of a graph
    for i in range(V):

        # If the current Node is unvisited
        # and weight of current vertex is
        # greater than maxW
        if (visited[i] == False and weights[i] > maxW):
            # Update maxW
            maxW = weights[i]

            # Update index
            index = i
    return index


# Utility function to find the maximum
# spanning tree of graph
def calc_edges(V, graph, parent):

    rows = []
    columns = []

    # Iterate over all possible Nodes
    # of a graph
    for i in range(1, V):
        rows.append(parent[i])
        columns.append(i)

    return rows, columns


# Function to find the maximum spanning tree
def maximumSpanningTree(V, graph):
    # visited[i]:Check if vertex i
    # is visited or not
    visited = [True] * V

    # weights[i]: Stores maximum weight of
    # graph to connect an edge with i
    weights = [0] * V

    # parent[i]: Stores the parent Node
    # of vertex i
    parent = [0] * V

    # Initialize weights as -INFINITE,
    # and visited of a Node as False
    for i in range(V):
        visited[i] = False
        weights[i] = -sys.maxsize

    # Include 1st vertex in
    # maximum spanning tree
    weights[0] = sys.maxsize
    parent[0] = -1

    # Search for other (V-1) vertices
    # and build a tree
    for i in range(V - 1):

        # Stores index of max-weight vertex
        # from a set of unvisited vertex
        maxVertexIndex = findMaxVertex(V, visited, weights)

        # Mark that vertex as visited
        visited[maxVertexIndex] = True

        # Update adjacent vertices of
        # the current visited vertex
        for j in range(V):

            # If there is an edge between j
            # and current visited vertex and
            # also j is unvisited vertex
            if (graph[j][maxVertexIndex] != 0 and visited[j] == False):

                # If graph[v][x] is
                # greater than weight[v]
                if (graph[j][maxVertexIndex] > weights[j]):
                    # Update weights[j]
                    weights[j] = graph[j][maxVertexIndex]

                    # Update parent[j]
                    parent[j] = maxVertexIndex

    return calc_edges(V, graph, parent)




def get_maximum_cost(cost_matrix:np.ndarray):
    vertices = cost_matrix.shape[0]
    return maximumSpanningTree(V=vertices, graph=cost_matrix.tolist())