import numpy as np
from .utils import get_section


class BeamDiagnostics:
    """
    Mixin providing measurement/diagnostic methods for Beam: power,
    intensity/phase profiles, centroid and width statistics, angular
    sections, normalization and more.
    """

    def Power(self, pol_index=None) -> float:
        """Get the total optical Power via numerical integration of the intensity profile 
        (trapezoidal-like Riemann sum with the grid's cell area)
        
        Parameters
        ----------
        pol_index: int, optional
            Polarization component into which the resulting power is measured. Default is power of 
            all field components.

        Returns
        -------
        float
        """
        return np.sum(self.int_profile(pol_index))*(4*self.nix/self.Dx)*(self.niy/self.Dy)

    
    def zr(self) -> float:
        """
        Compute the Rayleigh range of the beam from its `waist` and `lamb`.

        Returns
        -------
        float
            Rayleigh range, zr = pi*waist**2/lamb.
        """
        return np.pi*self.waist**2/self.lamb

    def k(self) -> float:
        """
        Compute k vector magnitude in vacum.
        
        Returns
        -------
        float
            k vector magniture in vaccum"""
        return 2*np.pi/self.lamb
    
    def int_profile(self, pol_index=None) -> np.ndarray:
        """
        Get intensity profile.

        pol_index=None -> total intensity
        pol_index=0    -> Ex intensity
        pol_index=1    -> Ey intensity
        pol_index=2    -> Ez intensity

        Parameters
        ----------
        pol_index: int, optional
            Polarization component into which the resulting intensity profile is measured. Default is sum of 
            all intensity components.

        Returns
        ------
        ndarray
            Array with intensity profile (x,y).
        """
        if pol_index is None:
            return np.sum(np.abs(self.field)**2, axis=0)
        return np.abs(self.field[pol_index])**2

    
    def phase(self, pol_index=None, twopi=False) -> np.ndarray:
        """
        Parameters
        ----------
        pol_index : int or None, optional
            None returns phase of all components; 0/1/2 returns the phase of
            Ex/Ey/Ez respectively.
        twopi : bool, optional
            If True, wrap phase into [0, 2*pi) instead of the default (-pi, pi].

        Returns
        -------
        ndarray
            Phase profile, shape (Dy, Dx) if pol_index is given or pol==1,
            otherwise shape (pol, Dy, Dx).
        """
        if pol_index is None:
            field = self.field
        else:
            field = self.field[pol_index]
        p = np.angle(field)
        if twopi:
            p = np.mod(p, 2*np.pi)
        if self.pol == 1:
            return p[0]
        return p
    

    def center_mass(self, pol_index=None) -> tuple: 
        """ 
        Compute the intensity-weighted center of mass of the field in physical x and y coordinates. 
        
        Parameters
        ---------- 
        pol_index : int, optional 
            Polarization component to use; None uses total intensity. 
            
        Returns
        ------- 
        tuple (x_cm, y_cm) 
            center-of-mass coordinates in physical units. """ 
        I = self.int_profile(pol_index) 

        x = np.asarray(self.x) 
        y = np.asarray(self.y) 
        if x.ndim == 2: 
            x = x[0, :] if x.shape[0] == 1 else x[0, :] 
        if y.ndim == 2: 
            y = y[:, 0] if y.shape[1] == 1 else y[:, 0] 
        # Total intensity 
        norm = np.sum(I) 

        if norm == 0: 
            raise ValueError("Cannot calculate center of mass of zero intensity.") 

        # Intensity marginalized along each direction 
        Ix = np.sum(I, axis=0) 
        Iy = np.sum(I, axis=1) 

        # Physical center of mass 
        x_cm = np.sum(x * Ix) / norm 
        y_cm = np.sum(y * Iy) / norm 

        return x_cm, y_cm
  
    def std(self, pol_index=None): 
        """
        Calculate the intensity-weighted spatial standard deviation
        along x and y.

        Parameters
        ---------- 
        pol_index : int, optional 
            Polarization component to use; None uses total intensity. 

        Returns
        -------
        sigma_x, sigma_y : float
            Intensity-weighted spatial standard deviations.
        """
        I = self.int_profile(pol_index) 
        x = np.asarray(self.x) 
        y = np.asarray(self.y) 
        x = x[0, :] if x.ndim == 2 else x 
        y = y[:, 0] if y.ndim == 2 else y 
        norm = np.sum(I) 
        Ix = np.sum(I, axis=0) 
        Iy = np.sum(I, axis=1) 
        x_cm, y_cm = self.center_mass(pol_index) 
        var_x = np.sum(Ix * (x - x_cm)**2) / norm 
        var_y = np.sum(Iy * (y - y_cm)**2) / norm 
        return np.sqrt(var_x), np.sqrt(var_y)
        
    def radial_std(self, pol_index=None) -> float:
        """
        Calculate the intensity-weighted radial standard deviation of the field
        about its center of mass.

        Parameters
        ----------
        pol_index : int, optional
            Polarization component to use; None uses total intensity.

        Returns
        -------
        float
            Intensity-weighted radial standard deviation.
        """
        sx, sy = self.std(pol_index)

        return np.sqrt(sx**2 + sy**2)
    
    def section(self, ang_min:float, ang_max:float, pol_index:int=0) -> np.ndarray:
        """
        Parameters
        ----------
        ang_min : float
            Minimum angle (radians) defining the angular wedge.
        ang_max : float
            Maximum angle (radians) defining the angular wedge.
        pol_index : int, optional
            Polarization component to extract the section from.

        Returns
        -------
        ndarray
            Complex field values restricted to the angular wedge [ang_min, ang_max].
        """
        return get_section(self, ang_min, ang_max, pol_index = pol_index)
    
    def int_section(self, ang_min:float, ang_max:float, pol_index:int=0)-> np.ndarray:
        """
        Parameters
        ----------
        ang_min : float
            Minimum angle (radians) defining the angular wedge.
        ang_max : float
            Maximum angle (radians) defining the angular wedge.
        pol_index : int, optional
            Polarization component to extract the section from.

        Returns
        -------
        ndarray
            Intensity |field|^2 within the angular wedge [ang_min, ang_max].
        """
        return np.abs(self.section(ang_min, ang_max, pol_index = pol_index))**2
    
    def Power_section(self, ang_min:float, ang_max:float, pol_index:int=0)-> float:
        """
        Parameters
        ----------
        ang_min : float
            Minimum angle (radians) defining the angular wedge.
        ang_max : float
            Maximum angle (radians) defining the angular wedge.
        pol_index : int, optional
            Polarization component to compute power for.

        Returns
        -------
        float
            Optical power contained within the angular wedge [ang_min, ang_max].
        """
        return np.sum(np.abs(self.section(ang_min, ang_max, pol_index = pol_index))**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def norm_beam(self)-> object:                                       
        """
        Returns
        -------
        Beam
            self, with field rescaled in place so that self.Power() == 1.
        """
        self.field = self.field/np.sqrt(self.Power())
        return self
    
    def Max_int1(self)-> object:                                         
        """
        Returns
        -------
        Beam
            self, with field rescaled in place so that the peak intensity equals 1.
        """
        self.field = self.field/np.sqrt(np.max(self.int_profile()))
        return self
    



