
class Market():
    def __init__(self, p):
        self.p = [p]
        self.traders = []
        self.buyers = []
        self.sellers = []
        self.clusters = []
        self.T = 20
        self.k = 3.5
        self.mu = 1.01
        self.hist_vol = 0.1 # TODO: aanpassen aan historische vol
        self.sigma = self.update_sigma()

    def update_sigma(self):
        return self.k*self.hist_vol

    def get_equilibrium_p(self):
        pass
