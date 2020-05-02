''' Defines the Weighted K-Means class that can be called in a similar fashion as scikit-learn kmeans,
        and the evaluation function eval_error for observing the performance of a certain redistricting

    Dependencies: numpy, copy and geopy.distance
'''

import numpy as np
import copy
from geopy.distance import great_circle


class Weighted_K_Means:

    def __init__(self, k=13, tolerance=0.0001, max_iterations=50,
                 weight_by_pop=False, weight_by_race=False, weight_by_dem=False):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.weight_by_pop = weight_by_pop
        self.weight_by_race = weight_by_race
        self.weight_by_dem = weight_by_dem

    def distance(self, point_1, point_2):
        return great_circle(point_1, point_2).km

    def fit(self, VTD_centers, population, black_population, dem_votes, total_votes,
            beta=0.5, alpha_pop=0.0, alpha_race=0.0, alpha_dem=0.0):

        self.VTD_centers = VTD_centers
        self.population = population
        self.black_population = black_population
        self.dem_votes = dem_votes
        self.total_votes = total_votes

        # 1. initialize centoroids at k random VTDs
        self.centroid_idx = np.random.choice(VTD_centers.id.values, size=self.k, replace=False)
        self.centroids = VTD_centers.iloc[self.centroid_idx, 1:].values

        # 2. initialize weights at 1/k
        self.weights = [1 / self.k for i in range(self.k)]

        self.total_population = sum(population)

        for iteration in range(self.max_iterations):

            self.clusters = {}  # stores actual lat long of each VTD assigned per cluster
            self.clusters_by_ids = {}  # stores the id of each VTD assigned per cluster
            for i in range(self.k):
                self.clusters[i] = []
                self.clusters_by_ids[i] = []

            # 3. find the distance between the point and cluster; choose the nearest centroid
            for VTD in self.VTD_centers.values:
                VTD_id = int(VTD[0])
                VTD_latlong = VTD[1:]

                # 3.1 calculate distances and weigh them
                distances = [self.distance(VTD_latlong, centroid) for centroid in self.centroids]
                weighted_distances = [d * w for d, w in zip(distances, self.weights)]

                # 3.2 find the closest cluster and assign the VTD to it as well as store its id
                cluster = weighted_distances.index(min(weighted_distances))
                self.clusters[cluster].append(VTD_latlong)
                self.clusters_by_ids[cluster].append(VTD_id)

            # 3.3 save old centroids to compare the improvement later
            previous = self.centroids.copy()

            # 4. average the cluster datapoints to re-calculate the centroids
            for cluster in self.clusters:
                self.centroids[cluster] = np.average(self.clusters[cluster], axis=0)

                # 5. check how far the centroids have moved
            isOptimal = True

            for i in range(len(self.centroids)):
                original_centroid = previous[i]
                curr = self.centroids[i]

                if np.sum(np.abs((curr - original_centroid) / original_centroid * 100.0)) > self.tolerance:
                    isOptimal = False

            # 5.1 break out if the centroids don't change their positions much
            if isOptimal:
                print('optimal after ' + str(iteration) + ' iterations')
                convergence = True
                break

            # 5.2 if the centroids still change a lot by the last iteration, we assume it converged
            if iteration == self.max_iterations - 1:
                convergence = True

            # 6. update the weights
            old_weights = copy.deepcopy(self.weights)

            for i in self.clusters:
                new_weight = 1
                if self.weight_by_pop:
                    new_weight *= sum(population[self.clusters_by_ids[i]]) ** alpha_pop
                if self.weight_by_race:
                    new_weight *= sum(black_population[self.clusters_by_ids[i]]) ** alpha_race
                if self.weight_by_dem:
                    if len(self.clusters_by_ids[i]) > 1:
                        new_weight *= (sum(dem_votes[self.clusters_by_ids[i]]) / sum(
                            total_votes[self.clusters_by_ids[i]])) ** alpha_dem
                    else:
                        convergence = False
                        break

                self.weights[i] = new_weight

            for i in self.clusters:
                self.weights[i] /= sum(self.weights)
                # 6.1 gradually update the weights
                self.weights[i] = beta * self.weights[i] + (1 - beta) * old_weights[i]

        return self.clusters, self.clusters_by_ids, convergence


def eval_error(clusters_by_ids, population, black_population, dem_votes, total_votes,
               k=13, weight_pop=1., weight_race=1., weight_dem=1.):
    fair_population = sum(population) / k
    fair_share_black = sum(black_population) / sum(population)
    fair_share_dem = sum(dem_votes) / sum(total_votes)

    total_error = 0

    for cluster in clusters_by_ids:
        pop_error = abs(sum(population[clusters_by_ids[cluster]]) - fair_population) / fair_population
        race_error = abs(sum(black_population[clusters_by_ids[cluster]]) / sum(
            population[clusters_by_ids[cluster]]) - fair_share_black)
        dem_error = abs(
            sum(dem_votes[clusters_by_ids[cluster]]) / sum(total_votes[clusters_by_ids[cluster]]) - fair_share_dem)

        total_error += weight_pop * pop_error + weight_race * race_error + weight_dem * dem_error

    return total_error