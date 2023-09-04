# Convert Medit mesh file to xdmf/h5 suitable for FEniCS. The tagging 
# convention is different than microcar:
# - we label extracellular space as 0 (there's no extracellular space of a cell as in microcard)
# - the EMI cells are than labeled starting from 1
# - for facet marking an interface between extrac. and the cell gets the tag
#   from the cell volume
# - the interfaces between the cells as labeled as they come and we make a lookup
#   table for them

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, os, time
    import numpy as np
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('medit_path', type=str, help='Medit file to convert')
    parser.add_argument('-pvdcells', type=int, default=0, choices=(0, 1))
    parser.add_argument('-test', type=int, default=0, choices=(0, 1))    
    args, _ = parser.parse_known_args()    

    name, ext = os.path.splitext(os.path.basename(args.medit_path))

    if ext != '.mesh': raise ValueError

    time_print = lambda arg, t0=time.time(): print(f'Ellapsed time after "{arg.lower()}" {(time.time()-t0):.2f} s')

    out_xdmf = f'{name}.xdmf'
    # Meshio convert
    if not os.path.exists(out_xdmf):
        import meshio

        mesh = meshio.medit.read(args.medit_path)    
        mesh.write(out_xdmf)
        time_print('XDMF conversion')
    # Get cell_data. FIXME: can't this be done with fenics directly?
    import h5py

    with h5py.File(f'{name}.h5', 'r') as f:
        cell_data = np.array(f.get('data3'), dtype='uintp')
        cell_data = np.where(np.mod(cell_data, 2), (cell_data-101)//2, 0)
        time_print('Reading cell data')
        
    import dolfin as df

    mesh = df.Mesh()
    with df.XDMFFile(mesh.mpi_comm(), out_xdmf) as f:
        f.read(mesh)
        
    tdim = mesh.topology().dim()
    cell_f = df.MeshFunction('size_t', mesh, tdim, 0)
    cell_f.array()[:] = cell_data
    time_print('Mesh and cell function loading')
    
    # Now we want to get interfaces
    facet_f = df.MeshFunction('size_t', mesh, tdim-1, 0)
    df.DomainBoundary().mark(facet_f, 0)

    mesh.init(tdim-1)
    mesh.init(tdim-1, tdim)

    time_print('Interface computation setup')    
    
    intracel_interfaces, next_iface_tag = {}, int(cell_f.array().max())
    for facet in df.facets(mesh):
        cs = facet.entities(3)
        # Boundary
        if len(cs) != 2:
            continue
    
        c0, c1 = cell_f[cs[0]], cell_f[cs[1]]
        # Inside cell
        if c0 == c1:
            continue

        # Boundary with extracellular
        if c0 == 0:
            facet_f[facet.index()] = c1
        elif c1 == 0:
            facet_f[facet.index()] = c0
        else:
            key = tuple(sorted((c0, c1)))
            if key not in intracel_interfaces:
                next_iface_tag += 1
                intracel_interfaces[key] = next_iface_tag
            facet_f[facet.index()] = intracel_interfaces[key]
    time_print('Computing interfaces')
        
    out_h5 = f'{name}_converted.h5'            
    with df.HDF5File(mesh.mpi_comm(), out_h5, 'w') as f:
        f.write(mesh, 'mesh')
        f.write(cell_f, 'cells')
        f.write(facet_f, 'interfaces')
    time_print('Writing mesh to HDF5')
        
    out_txt = f'{name}_converted.txt'        
    # Save the connectivity
    with open(out_txt, 'w') as out:
        out.write('# Tripplet intercellular-facet-tag cell-tag-0 cell-tag-1\n')
        for c0c1 in sorted(intracel_interfaces):
            iface_tag = intracel_interfaces[c0c1]
            c0, c1 = c0c1
            out.write(f'{iface_tag} {c0} {c1}\n')

    print(f'\nMesh at {out_h5}\nConnectivity at {out_txt}')

    if args.pvdcells:
        df.File(f'{name}_cells.pvd') << cell_f

    if args.test:
        mesh = df.Mesh()
        with df.HDF5File(mesh.mpi_comm(), out_h5, 'r') as f:
            f.read(mesh, 'mesh', False)
        assert mesh.num_cells() > 0

        lookup = np.loadtxt(out_txt)
        iface_tags = lookup[:, 0]
        cell_tags = np.unique(lookup[:, 1:].flatten())

        cell_f = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)        

        with df.HDF5File(mesh.mpi_comm(), out_h5, 'r') as f:
            f.read(cell_f, 'cells')
            f.read(facet_f, 'interfaces')

        assert set(np.unique(cell_f.array())) == set(cell_tags) | set((0, ))
        assert set(np.unique(facet_f.array())) == set(iface_tags) | set((0, )) | set(cell_tags)
