"""
OpenTorsion assembly definition and state-space conversion.

This module defines the physical test bench assembly using OpenTorsion library,
including shaft elements, disk inertias, and gear ratios. It provides functions
to create the assembly and convert it to a minimal state-space representation
suitable for estimation and control.

The test bench model includes:
- 24 disk elements with various inertias
- 21 shaft elements with stiffness and damping
- 4 gear elements representing gearbox ratios
- State-space transformation to minimal representation
"""

import opentorsion as ot

from scipy.linalg import pinv
from utils import minimize
import numpy as np

# PARAMETERS
I1 = 7.94e-4
k1 = 1.90e5
c1 = 8.08
d1 = 0.0030

I2 = 3.79e-6
k2 = 6.95e3
c2 = 0.29

I3 = 3.00e-6
k3 = 90.00
c3 = 0.24

I4 = 2.00e-6
k4 = 90.00
c4 = 0.24

I5 = 7.81e-3
k5 = 90.00
c5 = 0.24

I6 = 2.00e-6
k6 = 90.00
c6 = 0.24

I7 = 3.29e-6
k7 = 94.06
c7 = 0.00

I8 = 5.01e-5
k8 = 4.19e4  
c8 = 1.78

I9 = 6.50e-6
k9 = 5.40e3     
c9 = 0.23

I10 = 5.65e-5
k10 = 4.19e4
c10 = 1.78

I11 = 4.27e-6
k11 = 1.22e3
c11 = 0.52

I12 = 3.25e-4
k12 = 4.33e4
c12 = 1.84
d12 = 0.0042

I13 = 1.20e-4
k13 = 3.10e4
c13 = 1.32

I14 = 1.15e-5
k14 = 1.14e3
c14 = 0.05

I15 = 1.32e-4
k15 = 3.10e4
c15 = 1.32

I16 = 4.27e-6
k16 = 1.22e4
c16 = 0.52

I17 = 2.69e-4
k17 = 4.43e4
c17 = 1.88
d17 = 0.0042

I18 = 1.80e-4
k18 = 1.38e5
c18 = 5.86

I19 = 2.00e-5
k19 = 2.00e4
c19 = 0.85

I20 = 2.00e-4
k20 = 1.38e5
c20 = 5.86

I21 = 4.27e-6
k21 = 1.22e4
c21 = 0.52

I22 = 4.95e-2
d22 = 0.24

def test_bench():
    """
    Create and return the OpenTorsion assembly for the test bench.
    
    Constructs a complete assembly model with shaft elements (representing
    torsional stiffness and damping), disk elements (representing rotational
    inertias), and gear elements (representing gearbox ratios).
    
    Returns:
    --------
    ot.Assembly
        Configured OpenTorsion assembly object ready for state-space
        conversion or simulation.
    """
    # Shaft elements
    shafts = []
    shafts.append(ot.Shaft(0, 1, k=k1, c=c1))
    shafts.append(ot.Shaft(1, 2, k=k2, c=c2))
    shafts.append(ot.Shaft(2, 3, k=k3, c=c3))
    shafts.append(ot.Shaft(3, 4, k=k4, c=c4))
    shafts.append(ot.Shaft(4, 5, k=k5, c=c5))
    shafts.append(ot.Shaft(5, 6, k=k6, c=c6))
    shafts.append(ot.Shaft(6, 7, k=k7, c=c7))
    shafts.append(ot.Shaft(7, 8, k=k8, c=c8))
    shafts.append(ot.Shaft(8, 9, k=k9, c=c9))
    shafts.append(ot.Shaft(9, 10, k=k10, c=c10))
    shafts.append(ot.Shaft(10, 11, k=k11, c=c11))

    shafts.append(ot.Shaft(12, 13, k=k12, c=c12))
    shafts.append(ot.Shaft(13, 14, k=k13, c=c13))
    shafts.append(ot.Shaft(14, 15, k=k14, c=c14))
    shafts.append(ot.Shaft(15, 16, k=k15, c=c15))
    shafts.append(ot.Shaft(16, 17, k=k16, c=c16))

    shafts.append(ot.Shaft(18, 19, k=k17, c=c17))
    shafts.append(ot.Shaft(19, 20, k=k18, c=c18))
    shafts.append(ot.Shaft(20, 21, k=k19, c=c19))
    shafts.append(ot.Shaft(21, 22, k=k20, c=c20))
    shafts.append(ot.Shaft(22, 23, k=k21, c=c21))

    """ Disk elements """
    disks = []
    disks.append(ot.Disk(0, I=I1, c=d1))
    disks.append(ot.Disk(1, I=I2))
    disks.append(ot.Disk(2, I=I3))
    disks.append(ot.Disk(3, I=I4))
    disks.append(ot.Disk(4, I=I5))
    disks.append(ot.Disk(5, I=I6))
    disks.append(ot.Disk(6, I=I7))
    disks.append(ot.Disk(7, I=I8))
    disks.append(ot.Disk(8, I=I9))
    disks.append(ot.Disk(9, I=I10))
    disks.append(ot.Disk(10, I=I11))
    disks.append(ot.Disk(11, I=I12/2, c=d12))
    disks.append(ot.Disk(12, I=I12/2, c=d12))
    disks.append(ot.Disk(13, I=I13))
    disks.append(ot.Disk(14, I=I14))
    disks.append(ot.Disk(15, I=I15))
    disks.append(ot.Disk(16, I=I16))
    disks.append(ot.Disk(17, I=I17/2, c=d17))
    disks.append(ot.Disk(18, I=I17/2, c=d17))
    disks.append(ot.Disk(19, I=I18))
    disks.append(ot.Disk(20, I=I19))
    disks.append(ot.Disk(21, I=I20))
    disks.append(ot.Disk(22, I=I21))
    disks.append(ot.Disk(23, I=I22, c=d22))

    """ Gear elements """
    gear1 = ot.Gear(11, 0, 10)
    gear2 = ot.Gear(12, 0, 30, parent=gear1)

    gear3 = ot.Gear(17, 0, 10)
    gear4 = ot.Gear(18, 0, 40, parent=gear3)

    gears = [gear1, gear2, gear3, gear4]

    """ Assembly """
    return ot.Assembly(shafts, disks, gear_elements=gears)