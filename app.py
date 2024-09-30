from flask import Flask, render_template, request
import random
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        data = request.get_json()
        func = data["func"]
        if func == "generate":
            k = int(data["k"])
            mode = data["mode"]
            points = data["points"]
            if mode == "r":
                centroids = random.sample(points, k)
            elif mode == "ff":
                centroids = [random.choice(points)]
                while len(centroids) < k:
                    distances = np.array([
                        min(np.linalg.norm(np.array([point['x'], point['y']]) - np.array([centroid['x'], centroid['y']]))
                            for centroid in centroids) for point in points
                    ])
                    farthest_point = points[np.argmax(distances)]
                    centroids.append(farthest_point)
            elif mode == "kmpp":
                centroids = [random.choice(points)]
                for _ in range(1, k):
                    distances = np.array([
                        min(np.linalg.norm(np.array([point['x'], point['y']]) - np.array([centroid['x'], centroid['y']]))
                            for centroid in centroids) ** 2 for point in points
                    ])
                    probabilities = distances / distances.sum()
                    next_centroid = np.random.choice(points, p=probabilities)
                    centroids.append(next_centroid)
            elif mode == "manual":
                centroids = []
            return {
                "centroids": centroids,
            }
        elif func == "reset":
            points = [{"x": float(x), "y": float(y)} for x, y in zip(np.random.uniform(-10, 10, size=300), np.random.uniform(-10, 10, size=300))]
            return {"points": points}
        elif func == "step" or func == "run":
            converged = False
            points = np.array([[p['x'], p['y']] for p in data['points']])
            centroids = np.array([[c['x'], c['y']] for c in data['centroids']])
            k = int(data['k'])
            for _ in range(1 if func == "step" else 1000):
                distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)
                new_centroids = np.array([points[labels == i].mean(axis=0) for i in range(k)])
                if np.all(centroids == new_centroids):
                    converged = True
                    break
                centroids = new_centroids
            groups = {f'Clusters {i+1}': [] for i in range(k)}
            for i, label in enumerate(labels):
                groups[f'Clusters {label+1}'].append({'x': points[i, 0], 'y': points[i, 1]})
            result = {
                'clusters': {**groups},
                'centroids': [{'x': c[0], 'y': c[1]} for c in centroids],
                'converged': converged
            }
            return result
