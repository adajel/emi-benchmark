# VTK for visualizing `pi3 install --user vtk`
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkLine, vtkPolyData, vtkTriangle
from vtk.util.numpy_support import numpy_to_vtk
from vtk import vtkPolyDataWriter
import vtk

from dlt_dof_extraction import is_dlt_scalar, get_indices, get_values
import dolfin as df
import numpy as np
import os


def vtk_plot_data(f, facet_f, tags):
    '''Simple plot of f on marked subdomains. One value per facet!'''
    V = f.function_space()
    assert is_dlt_scalar(V)
    
    mesh = V.mesh()
    fdim = mesh.topology().dim() - 1
    assert fdim == 1 or fdim == 2
    
    marked_facets, indices = get_indices(V, facet_f, tags)
    # NOTE: Here we are going to insert redundant vertices in the interest
    # of not having to compute some mappings
    x = mesh.coordinates()

    _, f2v = mesh.init(fdim, 0), mesh.topology()(fdim, 0)
    
    coords = []
    for facet in marked_facets:
        coords.extend(x[f2v(facet)])
    coords = np.array(coords)
    npts, gdim = coords.shape

    if gdim == 2:
        coords = np.c_[coords, np.zeros(npts)]

    points = vtkPoints()
    [points.InsertNextPoint(*x) for x in coords]

    facetsPolyData = vtkPolyData()    
    # Add the points to the polydata container
    facetsPolyData.SetPoints(points)

    data = get_values(f, indices.ravel())
    data = data.reshape(indices.shape)
    # NOTE: we do one value per facet so
    data = np.mean(data, axis=1)

    VTK_data = numpy_to_vtk(num_array=data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    facetsPolyData.GetCellData().SetScalars(VTK_data)

    facets = vtkCellArray()
    vtkFacet, nvertices_per_facet = (vtkLine, 2) if fdim == 1 else (vtkTriangle, 3)

    # Lines
    global_index = 0
    for _ in enumerate(marked_facets):
        facet = vtkFacet()        
        for local_index in range(nvertices_per_facet):
            facet.GetPointIds().SetId(local_index, global_index)
            global_index += 1
        facets.InsertNextCell(facet)
    facetsPolyData.SetPolys(facets)
    
    return facetsPolyData


def vtk_plot(f, facet_f, tags, path):
    '''Dump'''
    data = vtk_plot_data(f, facet_f, tags)


    comm = df.MPI.comm_world
    if comm.size > 0:
        # NOTE: if we are running in parallel we prepend some rank info
        dirname, filename = os.path.dirname(path), os.path.basename(path)
        base, ext = os.path.splitext(filename)
        filename = f'{base}_rank{comm.rank}{ext}'
        path = os.path.join(dirname, filename)
    # Dump it
    writer = vtkPolyDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(path)
    writer.Write()

    return path
    
