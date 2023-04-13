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
* [`gotran`](https://finsberg.github.io/docs.gotran/index.html)

which is used to generate the Python module defining the ODE.

 #### TODO ####
 - [ ] consider a test where we have dy/dt = A(x)y with y(t=0) = y0
 - [ ] after stepping u should be fine
 - [ ] add forcing:  dy/dt = A(x)y + f(t) with y(t=0) = y0
 - [ ] things are currently quite slow -> multiprocessing?
 - [ ] rely on cbc.beat instead (mostly for speed reason)?

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.
