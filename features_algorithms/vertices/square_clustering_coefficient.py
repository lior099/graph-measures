import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class Square_ClusteringCoefficientCalculator(NodeFeatureCalculator):
    def __init__(self, *args, **kwargs):
        super(Square_ClusteringCoefficientCalculator, self).__init__(*args, **kwargs)

    def _calculate(self, include: set):
        self._features = nx.networkx.algorithms.cluster.square_clustering(self._gnx)

    def is_relevant(self):
        return True


feature_entry = {
    "square_clustering_coefficient": FeatureMeta(Square_ClusteringCoefficientCalculator, {"square_clustering"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(Square_ClusteringCoefficientCalculator, is_max_connected=True)