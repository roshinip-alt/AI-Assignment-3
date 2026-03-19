import pandas as pd
import math
import heapq
from collections import defaultdict

# ---------------------------------
# LOAD CSV
# ---------------------------------
file_name = "in.csv"   # change this to your file name
df = pd.read_csv(file_name)

# Keep only Indian cities
df = df[df["country"].str.strip().str.lower() == "india"]

# Keep needed columns only
df = df[["city", "lat", "lng"]].dropna()

# Clean city names
df["city"] = df["city"].str.strip().str.lower()

# Remove duplicate city names
df = df.drop_duplicates(subset="city")

# Convert dataframe to list of dictionaries
cities = df.to_dict("records")

# ---------------------------------
# HAVERSINE DISTANCE FUNCTION
# ---------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

# ---------------------------------
# BUILD GRAPH USING K NEAREST NEIGHBORS
# ---------------------------------
graph = defaultdict(list)
k = 20   # increase if needed

for i in range(len(cities)):
    current_city = cities[i]["city"]
    current_lat = cities[i]["lat"]
    current_lng = cities[i]["lng"]

    distances = []

    for j in range(len(cities)):
        if i == j:
            continue

        neighbor_city = cities[j]["city"]
        neighbor_lat = cities[j]["lat"]
        neighbor_lng = cities[j]["lng"]

        d = haversine(current_lat, current_lng, neighbor_lat, neighbor_lng)
        distances.append((d, neighbor_city))

    distances.sort()

    for d, neighbor in distances[:k]:
        graph[current_city].append((neighbor, d))
        graph[neighbor].append((current_city, d))   # undirected graph

print("Total cities in graph:", len(graph))

# ---------------------------------
# DIJKSTRA FUNCTION
# ---------------------------------
def dijkstra(graph, start, goal):
    pq = [(0, start)]
    dist = {node: float("inf") for node in graph}
    parent = {node: None for node in graph}

    dist[start] = 0

    while pq:
        curr_dist, u = heapq.heappop(pq)

        if curr_dist > dist[u]:
            continue

        if u == goal:
            break

        for v, w in graph[u]:
            new_dist = curr_dist + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))

    if goal not in dist or dist[goal] == float("inf"):
        return None, float("inf")

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path, dist[goal]

# ---------------------------------
# USER INPUT
# ---------------------------------
start = input("Enter start city: ").strip().lower()
goal = input("Enter goal city: ").strip().lower()

if start not in graph:
    print(f"Start city '{start}' not found in dataset.")
elif goal not in graph:
    print(f"Goal city '{goal}' not found in dataset.")
else:
    path, distance = dijkstra(graph, start, goal)

    if path is None:
        print("No path found. Try increasing k from 20 to 30.")
    else:
        print("\nShortest Path:")
        print(" -> ".join(city.title() for city in path))
        print("Approx Distance:", round(distance, 2), "km")