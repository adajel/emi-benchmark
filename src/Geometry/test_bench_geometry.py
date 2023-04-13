from bench_geometry import benchmark_mesh
import dolfin as df
import numpy as np
import json


df.set_log_level(50)


def test_marking():
    '''Connectivity is fine'''
    nx, ny = 2, 5
    dx, dy = 0.2, 0.1
    padx, pady = 0.1, 0.1

    mesh, entity_fs, connectivity = benchmark_mesh(clscale=1,
                                                   ncells=(nx, ny), dxs=(dx, dy), pads=(padx, pady),
                                                   view=False)

    # Some checks that the geometry is sane (as declared)
    # We have all the cells as promised
    cell_f = entity_fs[2]
    cell_tags = set(np.unique(cell_f.array()))
    assert cell_tags == set(range(1, 2+nx*ny))

    # Connectivity works
    # Each interface is either cell-cell or cell-extracellular
    _, = set(map(len, connectivity.values()))

    facet_f = entity_fs[1]
    
    dS = df.Measure('dS', domain=mesh, subdomain_data=facet_f)
    ds = df.Measure('ds', domain=mesh, subdomain_data=facet_f)    
    # Next check that we got the neighbors right
    # The idea is that if we integrate jump over sort of a characteristic
    # function we get the length of the interface
    intercell_lines = [line for line in connectivity if 1 not in connectivity[line]]
    cell_f = cell_f.array()

    Q = df.FunctionSpace(mesh, 'DG', 0)
    f = df.Function(Q)
    f_values = f.vector().get_local()

    for facet_tag in connectivity:
        cell_tag0, cell_tag1 = connectivity[facet_tag]
        f_values[np.where(cell_f == cell_tag0)] = 1
        f_values[np.where(cell_f == cell_tag1)] = 2
        f.vector().set_local(f_values)

        dB = ds if facet_tag in (1, 2, 3, 4) else dS
        restrict = (lambda x:x) if facet_tag in (1, 2, 3, 4) else df.jump
        # NOTE: this is more like a sanity check        
        target = df.assemble(df.Constant(1)*dB(facet_tag))
        ref = df.assemble(abs(restrict(f))*dB(facet_tag))

        assert abs(target - ref) < 1E-10
                             
        f_values *= 0

    _, f2c = mesh.init(1, 2), mesh.topology()(1, 2)

    # The interface connected cells have the right color
    for facet_tag in connectivity:
        for marked_facet in df.SubsetIterator(facet_f, facet_tag):
            facet_cells = f2c(marked_facet.index())
            assert set(cell_f[facet_cells]) == set(connectivity[facet_tag])

    xmin, xmax = map(df.Constant, np.min(mesh.coordinates(), axis=0))
    ymin, ymax = map(df.Constant, np.max(mesh.coordinates(), axis=0))
    # The outer boundaries are marked as promised
    x, y = df.SpatialCoordinate(mesh)
    
    assert abs(df.assemble(df.inner(x - xmin, x - xmin)*ds(1))) < 1E-10
    assert abs(df.assemble(df.inner(x - xmax, x - xmax)*ds(2))) < 1E-10
    assert abs(df.assemble(df.inner(y - ymin, y - ymin)*ds(3))) < 1E-10
    assert abs(df.assemble(df.inner(y - ymax, y - ymax)*ds(4))) < 1E-10


def test_load_marking():
    '''Things are fine when we write and load'''
    nx, ny = 3, 4
    dx, dy = 0.2, 0.3
    padx, pady = 0.1, 0.15

    mesh, entity_fs, connectivity = benchmark_mesh(clscale=0.5,
                                                   ncells=(nx, ny), dxs=(dx, dy), pads=(padx, pady),                                  
                                                   view=False)
    
    # Dumping to HDF5 ...
    with df.HDF5File(mesh.mpi_comm(), 'test.h5', 'w') as out:
        out.write(mesh, 'mesh/')
        out.write(entity_fs[2], 'subdomains/')
        out.write(entity_fs[1], 'interfaces/')        

    with open('test_connectivity.json', 'w') as out:
        json.dump(connectivity, out)

    # ... and loading        
    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), 'test.h5', 'r') as out:
        out.read(mesh, 'mesh', False)

    cell_f = df.MeshFunction('size_t', mesh, 2)
    facet_f = df.MeshFunction('size_t', mesh, 1)
    with df.HDF5File(mesh.mpi_comm(), 'test.h5', 'r') as out:
        out.read(cell_f, 'subdomains')
        out.read(facet_f, 'interfaces')
        # Just plot it for eye ball check
        df.File('test_subdomains.pvd') << cell_f
        df.File('test_interfaces.pvd') << facet_f

    with open('test_connectivity.json', 'r') as out:
        connectivity_ = json.load(out, parse_int=int)
        connectivity1 = {int(k): tuple(v) for k, v in connectivity_.items()}
    assert connectivity == connectivity1

    # Things didn't get messed up in loading
    cell_tags = set(np.unique(cell_f.array()))
    assert cell_tags == set(range(1, 2+nx*ny))

    # Connectivity works
    # Each interface is either cell-cell or cell-extracellular
    _, = set(map(len, connectivity.values()))

    dS = df.Measure('dS', domain=mesh, subdomain_data=facet_f)
    ds = df.Measure('ds', domain=mesh, subdomain_data=facet_f)    
    # Next check that we got the neighbors right
    # The idea is that if we integrate jump over sort of a characteristic
    # function we get the length of the interface
    intercell_lines = [line for line in connectivity if 1 not in connectivity[line]]
    cell_f = cell_f.array()

    Q = df.FunctionSpace(mesh, 'DG', 0)
    f = df.Function(Q)
    f_values = f.vector().get_local()

    for facet_tag in connectivity:
        cell_tag0, cell_tag1 = connectivity[facet_tag]
        f_values[np.where(cell_f == cell_tag0)] = 1
        f_values[np.where(cell_f == cell_tag1)] = 2
        f.vector().set_local(f_values)

        dB = ds if facet_tag in (1, 2, 3, 4) else dS
        restrict = (lambda x:x) if facet_tag in (1, 2, 3, 4) else df.jump
        # NOTE: this is more like a sanity check        
        target = df.assemble(df.Constant(1)*dB(facet_tag))
        ref = df.assemble(abs(restrict(f))*dB(facet_tag))

        assert abs(target - ref) < 1E-10
                             
        f_values *= 0

    _, f2c = mesh.init(1, 2), mesh.topology()(1, 2)

    # The interface connected cells have the right color
    for facet_tag in connectivity:
        for marked_facet in df.SubsetIterator(facet_f, facet_tag):
            facet_cells = f2c(marked_facet.index())
            assert set(cell_f[facet_cells]) == set(connectivity[facet_tag])

    xmin, xmax = map(df.Constant, np.min(mesh.coordinates(), axis=0))
    ymin, ymax = map(df.Constant, np.max(mesh.coordinates(), axis=0))
    # The outer boundaries are marked as promised
    x, y = df.SpatialCoordinate(mesh)
    
    assert abs(df.assemble(df.inner(x - xmin, x - xmin)*ds(1))) < 1E-10
    assert abs(df.assemble(df.inner(x - xmax, x - xmax)*ds(2))) < 1E-10
    assert abs(df.assemble(df.inner(y - ymin, y - ymin)*ds(3))) < 1E-10
    assert abs(df.assemble(df.inner(y - ymax, y - ymax)*ds(4))) < 1E-10
