from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
from collections import defaultdict
import numpy as np
import networkx
import gmsh


def benchmark_mesh(clscale, *, ncells, dxs, pads, view=False):
    '''
    EMI geometry with prod(ncells) (hyper)box cells of size (dx0 x dx1 x ... x dxd)
    for d in R^d which are enclosed in (hyper)box with pad distances. Mesh is 
    meshed with clscale size. Return mesh, mesh functions marking cells and 
    interfaces/boundaries and lookup table of facet tags -> connected cells

    In marking we will have these conventions: extracellular space is tagged
    as 1. Outer boundary of the extracellular space is labeled as

      x[0] == xmin -> 1
      x[0] == xmax -> 2
      x[1] == ymin -> 3
      x[1] == ymax -> 4
      ...

    The EMI cells are >= 2 and their boundaries are >= 2*d + 1,  
    '''
    # NOTE: this guy is a dispatcher
    assert len(ncells) == len(dxs) == len(pads)
    assert all(n > 0 for n in ncells)
    assert all(dx > 0 for dx in dxs)
    assert all(pad > 0 for pad in pads)

    # Dispatch
    # TODO
    which_mesh = {2: benchmark_mesh_2d}[len(ncells)]
    
    return which_mesh(clscale, ncells=ncells, dxs=dxs, pads=pads, view=view)


# Workers for 2d ---


def benchmark_mesh_2d(clscale, *, ncells, dxs, pads, view=False):
    '''
    EMI geometry with nx x ny rectangle cells of size dx x dywhich are enclosed 
    in rectangle box with pad distances. Mesh is meshed with clscale size. 
    Return mesh, mesh functions marking cells and interfaces/boundaries and 
    lookup table of facet tags -> connected cells

    In marking we will have these conventions: extracellular space is tagged
    as 1. Outer boundary of the extracellular space is labeled as

      x[0] == xmin -> 1
      x[0] == xmax -> 2
      x[1] == ymin -> 3
      x[1] == ymax -> 4
    
    The EMI cells are >= 2 and their boundaries are >= 5
    '''
    nx, ny = ncells
    dx, dy = dxs
    padx, pady = pads
    
    points = np.array([[i*dx, j*dy] for j in range(ny+1) for i in range(nx+1)])

    grid = np.arange((nx+1)*(ny+1)).reshape((ny+1, nx+1))
    squares = [(grid[i, j], grid[i, j+1], grid[i+1, j+1], grid[i+1, j]) for j in range(nx) for i in range(ny)]

    # We are now going to fill the model
    gmsh.initialize(['', '-v', '0', '-algo', 'front2d', '-clscale', str(clscale)])

    model = gmsh.model
    model, connectivity = benchmark_geometry_2d(model, points, squares, padx=padx, pady=pady, view=view)
    
    nodes, topologies = msh_gmsh_model(model, 2)
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions, connectivity

    
def benchmark_geometry_2d(model, points, squares, *, padx, pady, view=False):
    '''Define the EMI model'''
    fac = model.occ
    
    cell_points = [fac.addPoint(*p, z=0) for p in points]
    # Next we want to define each cell in terms of lines (their ids)
    # We will see the shared lines several times - keep track of them
    vertex_pair_lines = {}
    nvtx, = set(map(len, squares)) # As we have 4 vertices

    cells_as_lines = {}   # Gmsh id -> [Gmsh id]
    for square in squares:
        cell_lines = []
        for i in range(nvtx):  
            vi, vj = cell_points[square[i]], cell_points[square[(i+1) % nvtx]]
            line_key = tuple(sorted((vi, vj)))
            if line_key not in vertex_pair_lines:
                # We need to insert it
                line_id = fac.addLine(vi, vj)
                vertex_pair_lines[line_key] = line_id
            # For the purpose 
            line_id = vertex_pair_lines[line_key]
            cell_lines.append(line_id if (vi, vj) == line_key else -line_id)
        # With line ids in place we have a curve loop and a plane surface
        loop = fac.addCurveLoop([line for line in cell_lines])
        cell_tag = fac.addPlaneSurface([loop])
        
        cells_as_lines[cell_tag] = list(map(abs, cell_lines))

    # Let's add the rectangle (outer lines) that bounds the cells
    xmin, ymin = np.min(points, axis=0)
    xmax, ymax = np.max(points, axis=0)

    # 4 3
    # 1 2
    outer_points = [np.array([xmin, ymin]) + np.array([-padx, -pady]),
                    np.array([xmax, ymin]) + np.array([padx, -pady]),
                    np.array([xmax, ymax]) + np.array([padx, pady]),
                    np.array([xmin, ymax]) + np.array([-padx, pady])]
    
    outer_points = [fac.addPoint(*p, z=0) for p in outer_points]
    nopts = len(outer_points)
    outer_lines = [fac.addLine(outer_points[i], outer_points[(i+1)%nopts]) for i in range(nopts)]
    outer_loop = fac.addCurveLoop(outer_lines)

    # For the extracellular space we need to figure the inner boundary
    # This one is made from lines that are only connected to one cell
    lines_to_cells = defaultdict(list)
    for cell in cells_as_lines:
        [lines_to_cells[line].append(cell) for line in cells_as_lines[cell]]

    interface_lines = tuple(line for line in lines_to_cells if len(lines_to_cells[line]) == 1)
    
    fac.synchronize()    
    # The idea is that we can create a graph which is a cycle and we just need to walk it
    g = networkx.Graph()
    [g.add_edge(*model.getAdjacencies(1, line)[1], tag=line) for line in interface_lines]

    cycle = networkx.algorithms.cycles.find_cycle(g)
    inner_loop = fac.addCurveLoop([g.get_edge_data(*l)['tag'] for l in cycle])

    extracellular = fac.addPlaneSurface([outer_loop, inner_loop])
    # Update that cell boundary now neighbors extracellular
    [lines_to_cells[line].append(extracellular) for line in interface_lines]

    fac.synchronize()

    # Now what remains is tagging in terms into physical group ...
    cell_id_to_group = {extracellular: 1}
    model.addPhysicalGroup(2, [extracellular], 1)
    
    for group, cell_id in enumerate(cells_as_lines, 2):
        cell_id_to_group[cell_id] = group
        model.addPhysicalGroup(2, [cell_id], group)

    # Recall that we have bounded the cells
    xmin, xmax = xmin-padx, xmax+padx
    ymin, ymax = ymin-pady, ymax+pady
    # Here's the outer boundary labeling convention
    line_id_to_group = {match_line(fac, outer_lines, [xmin, 0.5*(ymin+ymax)]): 1,
                        match_line(fac, outer_lines, [xmax, 0.5*(ymin+ymax)]): 2,
                        match_line(fac, outer_lines, [0.5*(xmin+xmax), ymin]): 3,
                        match_line(fac, outer_lines, [0.5*(xmin+xmax), ymax]): 4}
    # The remaining start at 5
    all_lines = set(e[1] for e in model.getEntities(1))  # Their IDs
    inner_lines = all_lines - set(outer_lines)        
    for group, line_id in enumerate(inner_lines, 5):
        line_id_to_group[line_id] = group
        model.addPhysicalGroup(1, [line_id], group)

    # ... and translate the connectivity in terms of physical labels
    connectivity = {}  # Group id of line -> group ids of connected cells
    for line_id in lines_to_cells:
        cell_ids = lines_to_cells[line_id]
        connectivity[line_id_to_group[line_id]] = tuple(cell_id_to_group[cell_id] for cell_id in cell_ids)

    fac.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    return model, connectivity


def match_line(factory, candidates, center, tol=1E-13):
    '''Which candidate line match_linees the center'''
    found = False
    for entity in candidates:
        if np.linalg.norm(factory.getCenterOfMass(1, entity)[:2]-np.array(center)) < tol:
            found = True
            break
    assert found

    return entity

# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df
    import json
    
    nx, ny = 2, 5
    dx, dy = 0.2, 0.1
    padx, pady = 0.1, 0.1

    mesh, entity_fs, connectivity = benchmark_mesh(clscale=0.2,
                                                   ncells=(nx, ny), dxs=(dx, dy), pads=(padx, pady),
                                                   view=False)

    # Just show of dumping to HDF5 ...
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
