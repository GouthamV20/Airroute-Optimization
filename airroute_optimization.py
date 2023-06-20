import networkx as nx
import matplotlib.pyplot as plt
import time
import sys

class ShortestPathAlgorithm:
    def init(self, name):
        self.name = name

    def dijkstra(self, graph, source, target):
        # Implementation of Dijkstra's algorithm
        distances = {node: float('inf') for node in graph}
        distances[source] = 0

        visited = set()

        while target not in visited:
            # Find the node with the minimum distance
            current_node = min(
                [node for node in graph if node not in visited],
                key=lambda node: distances[node]
            )

            # Update distances of neighbors
            for neighbor, edge_weight in graph[current_node]:
                new_distance = distances[current_node] + edge_weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance

            visited.add(current_node)

        shortest_path_length = distances[target]
        return shortest_path_length

    def bellman_ford(self, graph, source, target):
        # Implementation of Bellman-Ford algorithm
        distances = {node: float('inf') for node in graph}
        distances[source] = 0

        for _ in range(len(graph) - 1):
            for node in graph:
                for neighbor, edge_weight in graph[node]:
                    new_distance = distances[node] + edge_weight
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance

        shortest_path_length = distances[target]
        return shortest_path_length

    def floyd_warshall(self, graph, source, target):
        # Implementation of Floyd-Warshall algorithm
        distances = {node: {n: float('inf') for n in graph} for node in graph}
        for node in graph:
            distances[node][node] = 0
        for node in graph:
            for neighbor, edge_weight in graph[node]:
                distances[node][neighbor] = edge_weight

        for k in graph:
            for i in graph:
                for j in graph:
                    distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])

        shortest_path_length = distances[source][target]
        return shortest_path_length

    def bidirectional_search(self, graph, source, target):
        # Implementation of bidirectional search algorithm
        forward_distances = {node: float('inf') for node in graph}
        forward_distances[source] = 0
        forward_visited = set()

        backward_distances = {node: float('inf') for node in graph}
        backward_distances[target] = 0
        backward_visited = set()

        while True:
            # Expand forward direction
            forward_node = min(
                [node for node in graph if node not in forward_visited],
                key=lambda node: forward_distances[node]
            )

            forward_visited.add(forward_node)
            if forward_node in backward_visited:
                break

            for neighbor, edge_weight in graph[forward_node]:
                new_distance = forward_distances[forward_node] + edge_weight
                if new_distance < forward_distances[neighbor]:
                    forward_distances[neighbor] = new_distance

            # Expand backward direction
            backward_node = min(
                [node for node in graph if node not in backward_visited],
                key=lambda node: backward_distances[node]
            )

            backward_visited.add(backward_node)
            if backward_node in forward_visited:
                break

            for neighbor, edge_weight in graph[backward_node]:
                new_distance = backward_distances[backward_node] + edge_weight
                if new_distance < backward_distances[neighbor]:
                    backward_distances[neighbor] = new_distance

        shortest_path_length = float('inf')
        for node in forward_visited.intersection(backward_visited):
            distance = forward_distances[node] + backward_distances[node]
            if distance < shortest_path_length:
                shortest_path_length = distance

        return shortest_path_length

    def a_star(self, graph, source, target):
        # Implementation of A* algorithm
        open_set = {source}
        closed_set = set()

        g_scores = {node: float('inf') for node in graph}
        g_scores[source] = 0

        f_scores = {node: float('inf') for node in graph}
        f_scores[source] = self.heuristic(source, target)

        while open_set:
            current_node = min(open_set, key=lambda node: f_scores[node])

            if current_node == target:
                return g_scores[target]

            open_set.remove(current_node)
            closed_set.add(current_node)

            for neighbor, edge_weight in graph[current_node]:
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_scores[current_node] + edge_weight

                if tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = g_scores[neighbor] + self.heuristic(neighbor, target)
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return float('inf')

    def heuristic(self, node, target):
        # Heuristic function for A* algorithm (example)
        return 0

    def measure_execution_time(self, algorithm, graph, source, target):
        start_time = time.time()
        shortest_path = algorithm(graph, source, target)
        end_time = time.time()
        execution_time = end_time - start_time

        # Calculate space complexity
        space_complexity = sys.getsizeof(graph) + sys.getsizeof(source) + sys.getsizeof(target)
        space_complexity += sys.getsizeof(shortest_path) + sys.getsizeof(execution_time)

        return shortest_path, execution_time, space_complexity

    def run_algorithms(self, graph, source, target):
        algorithms = {
            "Dijkstra": self.dijkstra,
            "Bellman-Ford": self.bellman_ford,
            "Floyd-Warshall": self.floyd_warshall,
            "Bidirectional Search": self.bidirectional_search,
            "A*": self.a_star
        }

        execution_times = []
        space_complexities = []

        for name, algorithm in algorithms.items():
            shortest_path, execution_time, space_complexity = self.measure_execution_time(algorithm, graph, source, target)
            execution_times.append(execution_time)
            space_complexities.append(space_complexity)

            print(name)
            print("Shortest Path Length:", shortest_path)
            print("Execution Time:", execution_time, "seconds")
            print("Space Complexity:", space_complexity, "bytes")
            print()

        # Plot the graph
        algorithm_names = list(algorithms.keys())
        plt.plot(algorithm_names, execution_times, marker='o')
        plt.xlabel('Algorithms')
        plt.ylabel('Execution Time (s)')
        plt.title('Comparison of Shortest Path Algorithms')
        plt.grid(True)
        plt.show()

        plt.plot(algorithm_names, space_complexities, marker='o')
        plt.xlabel('Algorithms')
        plt.ylabel('Space Complexity (bytes)')
        plt.title('Comparison of Space Complexity')
        plt.grid(True)
        plt.show()

        # Visualize the graph
        G = nx.Graph()
        for node, neighbors in graph.items():
            G.add_node(node)
            for neighbor, edge_weight in neighbors:
                G.add_edge(node, neighbor, weight=edge_weight)

        pos = nx.spring_layout(G, k=0.8)  # Adjust the 'k' parameter to increase/decrease node spacing
        plt.figure(figsize=(10, 8))  # Adjust the figure size as desired
        nx.draw(G, pos, with_labels=True, font_size=12)  # Increase font size here
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)  # Increase font size here
        plt.title('Graph Visualization')
        plt.show()

# Define the graph
graph = {
    "Mumbai": [("Delhi", 100), ("Jaipur", 150), ("Ahmedabad", 75)],
    "Delhi": [("Jaipur", 50), ("Agra", 25)],
    "Jaipur": [("Ahmedabad", 75), ("Agra", 25)],
    "Ahmedabad": [("Mumbai", 75), ("Jaipur", 75), ("Surat", 50)],
    "Agra": [("Delhi", 25), ("Jaipur", 25), ("Lucknow", 50)],
    "Surat": [("Ahmedabad", 50), ("Mumbai", 125)],
    "Lucknow": [("Agra", 50), ("Kanpur", 25)],
    "Kanpur": [("Lucknow", 25), ("Allahabad", 50)],
    "Allahabad": [("Kanpur", 50), ("Varanasi", 75)],
    "Varanasi": [("Allahabad", 75),("Hyderabad",200)],
    "Hyderabad": [("Coimbatore",150)],
    "Coimbatore": [("Mumbai",200),("Delhi",400)]
}

# Define source and target
source = "Coimbatore"
target = "Kanpur"

# Create an instance of the ShortestPathAlgorithm class
spa = ShortestPathAlgorithm("Shortest Path Algorithms")

# Run the algorithms and visualize the results
spa.run_algorithms(graph, source, target)