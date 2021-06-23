# Topological Graph Features

Topological feature calculators infrastructure.

## Calculating Features
This repository helps one to calculate features for a given graph. All features are implemented in python codes, 
and some features have also an accelerated version written in C++. Among the accelerated features, one can find 
a code for calculating 3- and 4-motifs using VDMC, a distributed algorithm to calculate 3- and 4-motifs in a 
GPU-parallelized way.

## What Features Can Be Calculated Here?
The set of all vertex features implemented in graph-measures is the following. 
The features that have an accelerated version are written in bold:
* Average neighbor degree
* General (i.e. degree if undirected, else or (in-degree, out-degree))
* Louvain (i.e. implement Louvain community detection method, then associate to each vertex the number of vertices 
in its community)
* Hierarchy energy
* **Motifs**
* **K core**
* **Attraction basin** 
* **Page Rank**
* Fiedler vector
* Closeness centrality
* Eccentricity
* Load centrality
* **BFS moments**
* **Flow** 
* Betweenness centrality
* Communicability betweenness centrality
* Eigenvector centrality
* Clustering coefficient
* Square clustering coefficient
* Generalized degree
* All pairs shortest path length
* All pairs shortest path

Aside from those, there are some other [edge features](features_algorithms/edges).
Some more information regarding the features can be found in the files of [features_meta](features_meta).

**NOTE:** For codes relating the motifs and their calculations, one might need to create the motif variation pickle files
in _features_algorithms/motif_variations_. To do so, one needs to run [isomorphic.py](features_algorithms/motif_variations/isomorphic.py). 
   
 
## Calculating Features

There are two main methods to calculate features:
1. Using [features_for_any_graph.py](features_for_any_graph.py) - A file for calculating any requested features on a given graph.
The class for calculating features in this file is _FeatureCalculator_. \
The graph is input to this file as a text-like file of edges, with a comma delimiter. 
For example, the graph [example_graph.txt](measure_tests/example_graph.txt) is the following file: 
    ```
    0,1
    0,2
    1,3
    3,2
    ```
    Now, an implementation of feature calculations on this graph looks like this:
    ```python
   import os
   from features_for_any_graph import FeatureCalculator
   feats = ["motif3", "louvain"]  # Any set of features
   path = os.path.join("measure_tests", "example_graph.txt") 
   head = "" # The path in which one would like to keep the pickled features calculated in the process. 
   # More options are shown here. For infomation about them, refer to the file.
   ftr_calc = FeatureCalculator(path, head, feats, acc=True, directed=False, gpu=True, device=0, verbose=True)
   ftr_calc.calculate_features()
    ``` 
    More information can be found in [features_for_any_graph.py](features_for_any_graph.py).
2. By the calculations as below:
The calculations require an input graph in NetworkX format, later referred as gnx, and a logger.
For this example, we build a gnx and define a logger:
    ```python
   import networkx as nx
   from loggers import PrintLogger
    
   gnx = nx.DiGraph()  # should be a subclass of Graph
   gnx.add_edges_from([(0, 1), (0, 2), (1, 3), (3, 2)])
    
   logger = PrintLogger("MyLogger")
    ```
    On the gnx we have, we will want to calculate the topological features.
    There are two options to calculate topological features here, depending on the number of features we want to calculate: 
    * Calculate a specific feature:

    ```python
    import numpy as np
    # Import the feature. 
    # If simple, import it from vertices folder, otherwise from accelerated_graph_features: 
    from features_algorithms.vertices.louvain import LouvainCalculator  
    
    feature = LouvainCalculator(gnx, logger=logger)  
    feature.build()  # The building happens here
    
    mx = feature.to_matrix(mtype=np.matrix)  # After building, one can request to get features the a matrix 
    ```

    * Calculate a set of features (one feature can as well be calculated as written here):

    ```python
   import numpy as np
   from features_infra.graph_features import GraphFeatures
    
   from features_algorithms.vertices.louvain import LouvainCalculator
   from features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator
    
   features_meta = {
       "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
      "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
   }  # Hold the set of features as written here. 
    
   features = GraphFeatures(gnx, features_meta, logger=logger) 
   features.build()
    
   mx = features.to_matrix(mtype=np.matrix)
    ```

## How to use accelerated features?
The accelerated feature calculation option requires some prior work, since its files are C++ files which require making.
The complete manual can be found [here](https://drive.google.com/file/d/1SMGWsGpiegR1ZkA2zffyAJO4HNhM53dD/view?usp=sharing). 

In short, for the GPU accelerated version using existing anaconda 3:
* In case of problems with pickle files (e.g. inability to download), one can use the following [file](features_algorithms/motif_variations/rewrite_variations.py)
to recreate the motif variation pickle files.
1. Move into the path _features_algorithms/accelerated_graph_features/src_. 
2. Create the conda environment required in which the work will be done: _"conda env create -f env.yml"_ 
3. Activate the new environment: _"conda activate boost"_. \
Then, make the files required for accelerated graph measures including motifs on GPU: \
_"make -f Makefile-gpu"_ (this might take a while).
4. Now, the accelerated graph features (including the GPU based motif calculations) can be used.


## VDMC Results directory
The directory called _vdmc_results_ includes results used for plotting some figures in the VDMC paper.
The code files for receiving these results are located there, as well as the results files.