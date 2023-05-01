# Chapter KNP-EMI #

### Dependencies P1 code ###

Get the environment needed (all dependencies etc.), build and
and run the Docker container *ceciledc/fenics_mixed_dimensional:13-03-20* by:

* Installing docker: https://docs.docker.com/engine/installation/
* Build and start docker container with:

        docker run -t -v $(pwd):/home/fenics -i ceciledc/fenics_mixed_dimensional:13-03-20
        pip3 install vtk
        cd ulfy-master
        python3 setup.py install 

### Dependencies P0 code ###

Get the environment needed (all dependencies etc.), build and
and run the Docker container *ceciledc/fenics_mixed_dimensional:13-03-20* by:

* Installing docker: https://docs.docker.com/engine/installation/
* Build and start docker container with:

        docker run -t -v $(pwd):/home/fenics -i quay.io/fenicsproject/stable
        cd ulfy-master
        python3 setup.py install

### Geometry ###

This `Geometry/bench_geometry.py` contains functionality for creating
GMSH geometries of connected rectangular cells enclosed in some larger
rectangle forming the extracellular space.

  <p align="center">
    <img src="https://github.com/adajel/emi-benchmark/blob/main/doc/geometry.png">
  </p>
  
 #### Dependencies ####
In addition to standard FEniCS stack (`2019.1.0` and higher) you will need
* `networkx`
* [`gmsh`](https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/api/gmsh.py)
* [`gmshnics`](https://github.com/MiroK/gmshnics)

 #### TODO ####
 - [ ] Geometry in 3d (based on boolean operations or maybe with hand-coded boxes for speed?)

### Membrane ###

This `Membrane/membrane.py` contains functionality for solving ODEs localized on
facets and interaction with HDiv Trace space in FEniCS. 
  
 #### Dependencies ####
In addition to standard FEniCS stack (`2019.1.0` and higher) you will need
* [`gotran`](https://finsberg.github.io/docs.gotran/index.html) is used to generate the Python module defining the ODE
* [`numbalsoda`](https://github.com/Nicholaswogan/numbalsoda) and [`numba`](https://numba.pydata.org/) are used to gain speed.

Note that standard way of using `gotran` (spefically `gotran2py`) is to generate
a module which specifies the ODE model (as python) for the membrane. Then, to integrate
the model forward in time one would e.g. use scipy. However, this solution is very slow
as the integrator needs to constantly call into to Python. To gain speed, what we do instead is

- use `numba` to generate a C code for evaluation of the ODE. This requires switching
the representations in `gotran2py` from "named" to "array"
- the resulting function is passed to `numbalsoda` integrator (the entire integration
should then happen in C)
- the speed-up of this compiled approach seems to be 30-40x relative to the scipy+gotran
approach (eg. 50k ODEs can be solved in about 2.5s)

 #### TODO ####
 - [ ] consider a test where we have dy/dt = A(x)y with y(t=0) = y0
 - [ ] after stepping u should be fine
 - [ ] add forcing:  dy/dt = A(x)y + f(t) with y(t=0) = y0
 - [ ] do we gain anything by additional compilation and multiprocessing?
 - [ ] (**maybe**) rely on cbc.beat instead (mostly for speed reason)?

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.
