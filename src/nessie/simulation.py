class Simulation:
    def __init__(self, _name, _electricFields=None, _weightingPotential=None, 
                    _cceField=None, _chargeCaptureField=None, _electronicResponse=None):        
        self.name = _name
        self.electricFields = _electricFields
        self.weightingPotential = _weightingPotential
        self.cceField = _cceField
        self.chargeCaptureField = _chargeCaptureField
        self.electronicResponse = _electronicResponse

    def setElectricField(self):
        return None

    def setWeightingField(self):
        return None

    def setChargeCollectionEfficiencyField(self):
        return None

    def setChargeCaptureField(self):
        return None

    def setElectronicResponse(self):
        return None

    def simulate(self, events, plasma=False, diffusion=False, capture=False):
        return None
