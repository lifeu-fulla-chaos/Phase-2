import numpy as np
from tried_many.lorentz import lorenz_equations

def backstepping_controller(master_state, slave_state, params, gain=5.0):
    """Backstepping controller for synchronization"""
    e = master_state - slave_state
    # Simple proportional controller with gain
    u = gain * e
    return u

def controlled_lorenz_system(t, state, params, control):
    """Lorenz system with control input for synchronization"""
    base_derivatives = lorenz_equations(t, state, params)
    return base_derivatives + control