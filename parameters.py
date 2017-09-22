class Parameters():

    def __init__(self, M=3, radius=.3, n_estimators=10, max_depth=20):

        # Feature parameters
        self.M = M
        self.radius = radius

        # RDF Parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth