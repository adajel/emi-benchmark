from dolfin import *
from petsc4py import PETSc
import sys
import json
 
parameters['form_compiler']['optimize']           = True
parameters['form_compiler']['cpp_optimize']       = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

# time disc. parameters
dt         = 0.1
time_steps = 10
t          = 0.0

# physical parameters
sigma_e = 1
sigma_i = 1
C_M     = 1

# output files
out_ui = XDMFFile(MPI.comm_world, "CG/output/sol_ui.xdmf")
out_ue = XDMFFile(MPI.comm_world, "CG/output/sol_ue.xdmf")	

mesh = Mesh()

print('Reading mesh and creating spaces...')

hdf = HDF5File(mesh.mpi_comm(), "Geometry/data/test.h5", "r")
hdf.read(mesh, "/mesh", False)
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
hdf.read(subdomains, "/subdomains")
interfaces = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
hdf.read(interfaces, "/interfaces")

# read dictionary for interfaces 
with open('Geometry/data/test_connectivity.json', 'r') as file:
	interfaces_dict = json.load(file)

# read number of cells
N_cells = 1
N_inter = 1
for i in interfaces_dict:
	N_cells = max(max(interfaces_dict[i]) - 1 , N_cells)
	N_inter = max(int(i), N_inter)

print('#cells =', N_cells)
print('#intersections =', N_inter)

# list for spaces
W_list = []

# extra
extra_mesh = MeshView.create(subdomains, 1) # exterior  mesh
W_list.append(FunctionSpace(extra_mesh, 'CG', 1))

# intra
intra_meshes = []

for i in range(N_cells):
	intra_meshes.append(MeshView.create(subdomains, i + 2))    # interior  meshes
	W_list.append(FunctionSpace(intra_meshes[i], 'CG', 1) )    # interior  spaces
	
print('Create variational form...')

W = MixedFunctionSpace(*W_list)

# functions
u = TrialFunctions(W)
v =  TestFunctions(W)

# setup linear and bilinear form
a00 = 0; a01 = 0; a11 = 0 
L0 = 0; L1 = 0

# define measures
dxe = Measure('dx', domain=extra_mesh) 
dxi = []

for i in range(N_cells):
	dxi.append(Measure('dx', domain=intra_meshes[i]))		

# weak form of equation for extra-extra
a11 += dt * inner(sigma_e*grad(u[0]), grad(v[0]))*dxe + C_M * inner(u[0], v[0])*ds

# weak form of equation for intra-intra
for i in range(N_cells):	
	a00 += dt * inner(sigma_i*grad(u[i+1]), grad(v[i+1]))*dxi[i] + C_M * inner(u[i+1], v[i+1])*ds

# intra-extra terms

# interface
interface_meshes = [0] * (N_inter + 1)
dxg 			 = [0] * (N_inter + 1)

for i_string in interfaces_dict:			

	i = int(i_string)

	interface_meshes[i] = MeshView.create(interfaces, i)
	dxg[i] = (Measure('dx', domain=interface_meshes[i]))	

	domains = interfaces_dict[i_string]
	domain1 = domains[0] - 1
	domain2 = domains[1] - 1
	
	#a01 -= C_M * inner(u[domain1], v[domain2])*dxg[i]

	v0 = Constant(1.0)

	# linear forms 
	fg = v0 - (dt/C_M) * v0

	L0 += C_M * inner(fg, v[domain1])*dxg[i]
	L1 -= C_M * inner(fg, v[domain2])*dxg[i] 


# sum up various contributions
a = a00 + a11 + a01 
L = L0 + L1

wh = Function(W)      

# Assembly
system = assemble_mixed_system(a == L, wh)
matrix_blocks = system[0]
rhs_blocks    = system[1]
sol_blocks    = system[2]

solver = PETScLUSolver()  

# allocate Petsc structures
b = Vector()
w = Vector()	
A   = PETScNestMatrix(matrix_blocks)
Aij = PETScNestMatrix(matrix_blocks)	
A.init_vectors(w, sol_blocks)
Aij.convert_to_aij()

# Time-stepping
for i in range(time_steps):

	print('Time step', i + 1) 	

	# update current time
	t += dt
	
	# Reassembling RHS blocks
	Llist = extract_blocks(L)               
	rhs_blocks = [assemble_mixed(LL) for LL in Llist] 
	A.init_vectors(b, rhs_blocks)
	
	# Solve 
	solver.solve(Aij, w, b)

	start_idx = 0
	
	# loop for all cells + extra
	for j in range(N_cells + 1):

		w0 = Function(W_list[j]).vector()
		w0.set_local(w.get_local()[start_idx:start_idx + W_list[j].dim()])
		w0.apply('insert')
		wh.sub(j).assign(Function(W_list[j],w0))

		start_idx += W_list[j].dim()

		# w_petsc = as_backend_type(w).vec()
		# wi_petsc = w_petsc.getNestSubVecs()[j]

		# wi_fun = Function(W_list[j])
		# wi = wi_fun.vector()	
		# wi.set_local(wi_petsc.array)				
		# wi.apply('insert')
		# wh.sub(j).assign(wi_fun)

		# write results 		
		if j == 0:
			
			#wh.sub(j).rename('sol_extra', '')	
			#wi_fun.rename(   'sol_extra', '')  
			out_ue.write(wh.sub(j), t)	
		else:		
			#wh.sub(j).rename('sol_intra', '')	
			#wi_fun.rename(   'sol_intra', '')  
			out_ui.write(wh.sub(j), t)

	# update previous membrane potential		
# 	v.assign(interpolate(wh.sub(0), Wg) - interpolate(wh.sub(1), Wg))
	



