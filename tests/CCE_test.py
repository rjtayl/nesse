# -*- coding: utf-8 -*-
import os
import sys
# sys.path.append(os.getcwd()+"/src/")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import nesse
import numpy as np
import matplotlib.pyplot as plt

def test_CCE():
    ID = "4"

    xs = np.zeros(101)
    ys = np.zeros(101)
    zs = np.linspace(0,100e-9,101)
    pos = np.array([xs,ys,zs]).T
    Es = np.ones(101)*1000
    times = zs
    event = nesse.Event(0,pos,Es,times)

    Events= [event]

    EF_filename = "config/Fields/4e10/NessieEF_Base4e7Linear0-150.0V.hf"
    WP_filename = "config/Fields/NessieWP_4e7Linear0-150V_grid.hf"

    Efield=nesse.fieldFromH5(EF_filename, rotate90=True)  
    weightingPotential = nesse.potentialFromH5(WP_filename, rotate90=True)

    sim = nesse.Simulation("Example_sim", 125, Efield, _weightingPotential=weightingPotential, contacts=1)

    bounds = [[axis[0],axis[-1]] for axis in Efield.grid]
    bounds[2][0] = 0
    sim.setBounds(bounds)

# test default no dead layer
    sim.setChargeCaptureField("hard")
    
    cc_e, cc_h = sim.pairCreation(event, 100, False)
    qp_zs = [np.round(qp.pos[0][2],11) for qp in cc_e]
    bins = np.arange(-0.1e-9,100e-9,1e-9)

    
    vals, _, _= plt.hist(qp_zs, bins=bins, histtype="step")
    expected_vals = np.concatenate([np.zeros(1), np.ones(99)*100])
    assert np.array_equal(expected_vals, vals), f"Default dead layer failed. \n Expected pairs: {expected_vals} \n New pairs: {vals}"

    # test hard dead layer
    sim.setChargeCaptureField("hard", depth=50e-9)
    
    cc_e, cc_h = sim.pairCreation(event, 100, False)
    qp_zs = [np.round(qp.pos[0][2],11) for qp in cc_e]

    vals, _, _= plt.hist(qp_zs, bins=bins, histtype="step")
    expected_vals = np.concatenate([np.zeros(51), np.ones(49)*100])

    assert np.array_equal(expected_vals, vals), f"Hard dead layer failed. \n Expected pairs: {expected_vals} \n New pairs: {vals}"

# test soft dead layer
    sim.setChargeCaptureField("soft", 50e-9)
    
    cc_e, cc_h = sim.pairCreation(event, 100, False)
    qp_zs = [np.round(qp.pos[0][2],11) for qp in cc_e]

    vals, _, _= plt.hist(qp_zs, bins=bins, histtype="step")
    expected_vals = np.round((1-np.exp(-zs[:-1]/50e-9))*100,0)
    assert np.array_equal(expected_vals, vals), f"Soft dead layer failed. \n Expected pairs: {expected_vals} \n New pairs: {vals}"

# test Auger dead layer
    sim.setChargeCaptureField("Auger")
    
    cc_e, cc_h = sim.pairCreation(event, 100, False)
    qp_zs = [np.round(qp.pos[0][2],11) for qp in cc_e]

    vals, _, _= plt.hist(qp_zs, bins=bins, histtype="step")
    expected_vals = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0, 14.0, 16.0, 17.0, 19.0, 20.0, 22.0, 23.0, 24.0,
                      26.0, 27.0, 29.0, 29.0, 31.0, 33.0, 35.0, 36.0, 38.0, 38.0, 39.0, 41.0, 41.0, 43.0, 43.0, 44.0, 
                      45.0, 46.0, 48.0, 49.0, 49.0, 50.0, 51.0, 53.0, 54.0, 57.0, 60.0, 65.0, 71.0, 76.0, 79.0, 81.0, 
                      84.0, 86.0, 88.0, 89.0, 88.0, 87.0, 84.0, 78.0, 75.0, 71.0, 67.0, 62.0, 59.0, 57.0, 55.0, 54.0, 
                      55.0, 55.0, 59.0, 66.0, 72.0, 75.0, 76.0, 79.0, 83.0, 83.0, 83.0, 83.0, 83.0, 83.0, 84.0, 83.0, 
                      83.0, 83.0, 84.0, 84.0, 83.0, 83.0, 83.0, 83.0, 84.0, 84.0, 85.0, 84.0, 84.0, 83.0, 84.0, 84.0, 
                      84.0, 84.0, 83.0]
    assert np.array_equal(expected_vals, vals), f"Auger dead layer failed. \n Expected pairs: {expected_vals} \n New pairs: {vals}"


if __name__ == "__main__":
    test_CCE() 