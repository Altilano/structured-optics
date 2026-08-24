from structured_optics.struct_opt import *
import numpy as np
from structured_optics.utils import *
import time


def slm_hologram(Beam, x_grating, y_grating, method, input_beam, eps, max_range):
    #generate a hologram for a slm
    
    dx = (Beam.x[0,1] - Beam.x[0,0])
    dy = (Beam.y[1,0] - Beam.y[0,0])
    if input_beam is None:
        input_field = np.exp(-((Beam.x**2 + Beam.y**2)/(2*np.max(Beam.x))**2))
    else:
        input_field = input_beam.field
    desired = Beam.field/np.max(np.abs(Beam.field))
    lamb_x = dx*x_grating
    lamb_y = dy*y_grating
    Ain = np.abs(input_field)
    Ain = Ain / Ain.max()
    Ades = np.abs(desired)
    s = np.min(Ain / (Ades + eps))
    Ades = s * Ades
    A_rel = Ades / Ain
    phi_g = np.mod(2*np.pi*(Beam.x/lamb_x + Beam.y/lamb_y), 2*np.pi)
    phi_relg = np.angle(desired) - np.angle(input_field) + phi_g
    if method == 'simple':
        H = A_rel*phi_relg
    elif method == 'g_holo':
        edg = desired*np.exp(1j*phi_g)
        N = np.min(np.abs(input_field)/(np.abs(desired) + 1e-9))
        H = np.angle(N*edg + input_field)
    elif method == 'davis':
        H = (1-inv_sinc(A_rel)/np.pi)*phi_relg
    elif method == 'bolduc':
        M = 1+inv_sinc(A_rel)/np.pi
        H = M*(phi_relg - np.pi*M)
    elif method == 'bessel0':
        H = phi_relg + inv_J0(A_rel)*np.sin(phi_relg)
    elif method == 'bessel1':
        H = inv_J1(A_rel)*np.sin(phi_relg)
    else:
        print('Error! method must be simple, g_holo, davis, bolduc, bessel0 or bessel1.')
    H = max_range * (H - H.min()) / (H.max() - H.min())
    return H.astype(np.uint8)

def dmd_hologram(Beam, cx, cy, sign):
    #generates a hologram for a dmd
    U = np.abs(Beam.field)
    U = U/np.amax(U)
    phi = Beam.phase()
    A = np.arcsin(U)

    fx = cx*Beam.Dx/(2*Beam.nix)
    fy = cy*Beam.Dy/(2*Beam.niy)
    g = [fx, fy]
    G = g[0]*Beam.x + g[1]*Beam.y

    H = 0.5 + sign*0.5*np.sign(np.cos(phi + 2*np.pi*G) - np.cos(A))
    return H




