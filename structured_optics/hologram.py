import numpy as np
from .utils import inv_J0, inv_J1, inv_sinc

__all__ =["slm_hologram", "dmd_hologram"]



def slm_hologram(beam, x_grating:int, y_grating:int, method:str = 'bessel1', input_beam:object= None, 
                 eps:float = np.finfo(float).eps, max_range:int = 255)-> np.ndarray:
    """
    Parameters
    ----------
    x_grating : int
        Carrier grating spatial frequency along x (grid pixels).
    y_grating : int
        Carrier grating spatial frequency along y (grid pixels).
    method : str, optional
        Amplitude/phase encoding scheme used to build the hologram.
    input_beam : Beam, optional
        Reference illumination beam; defaults to self if None.
    eps : float, optional
        Small value added to avoid division by zero during encoding.
    max_range : int, optional
        Output gray-level range (e.g. 255 for an 8-bit SLM lookup table).

    Returns
    -------
    ndarray
        Encoded hologram pattern, shape (Dy, Dx), ready for SLM display.
    """
    
    dx = (beam.x[0,1] - beam.x[0,0])
    dy = (beam.y[1,0] - beam.y[0,0])
    if input_beam is None:
        input_field = np.exp(-((beam.x**2 + beam.y**2)/(2*np.max(beam.x))**2))
    else:
        input_field = input_beam.Ex
    desired = beam.Ex/np.max(np.abs(beam.Ex))
    lamb_x = dx*x_grating
    lamb_y = dy*y_grating
    Ain = np.abs(input_field)
    Ain = Ain / Ain.max()
    Ades = np.abs(desired)
    s = np.min(Ain / (Ades + eps))
    Ades = s * Ades
    A_rel = Ades / Ain
    phi_g = np.mod(2*np.pi*(beam.x/lamb_x + beam.y/lamb_y), 2*np.pi)
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
        raise ValueError('Error! method must be simple, g_holo, davis, bolduc, bessel0 or bessel1.')
    H = max_range * (H - H.min()) / (H.max() - H.min())
    return H.astype(np.uint8)


def dmd_hologram(beam, cx:float, cy:float, sign:int=1, input:object=None)-> np.ndarray:
    """ 
    Generate a binary-amplitude DMD hologram. 
    
    Parameters 
    ---------- 
    beam : Beam 
        Desired output beam. Its Ex component defines the target amplitude and phase. 
    cx, cy : float 
        Carrier frequencies expressed in DMD pixel units. 
    sign : int 
        Controls the binary hologram orientation. Usually +1 or -1. 
    input : Beam 
        Optional Incident beam illuminating the DMD. If None, the incident beam is assumed 
        to have uniform amplitude and zero phase, reproducing the original behavior. 
    
    Returns 
    ------- 
    H : ndarray 
        Binary DMD hologram. """ 
    
    # Target field 
    target = beam.Ex 
    # Target amplitude and phase 
    target_amp = np.abs(target) 
    target_phase = np.angle(target) 

    if input is None: 
        U = target_amp / np.amax(target_amp) 
        phi = target_phase 
    else: 
        input_field = input.Ex 
        input_amp = np.abs(input_field) 
        input_phase = np.angle(input_field) 
  
        eps = np.finfo(float).eps 
        U = target_amp / np.maximum(input_amp, eps) 
        phi = target_phase - input_phase 
        max_U = np.amax(U) 
        if max_U > 0: 
            U = U / max_U 

    A = np.arcsin(U) 
    fx = cx * beam.Dx / (2 * beam.nix) 
    fy = cy * beam.Dy / (2 * beam.niy) 
    G = fx * beam.x + fy * beam.y 

    H = 0.5 + sign * 0.5 * np.sign( np.cos(phi + 2 * np.pi * G) - np.cos(A) ) 
    return H




