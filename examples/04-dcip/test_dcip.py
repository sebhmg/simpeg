from geoh5py.workspace import Workspace
import numpy as np
from scipy.spatial import cKDTree
from SimPEG import dask
import dask
from SimPEG import (
    maps,
    utils,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    directives,
    inversion,
    objective_function,
    data
)
from SimPEG.utils.drivers import create_nested_mesh
from SimPEG.electromagnetics.static import resistivity as dc, utils as DCutils
from SimPEG.electromagnetics.static import induced_polarization as ip
from dask.distributed import Client, LocalCluster, get_client

from geoapps.utils import octree_2_treemesh, treemesh_2_octree
from discretize import utils as d_utils
from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from discretize import TensorMesh
from pymatsolver.direct import Pardiso as Solver


def create_tile_ip(source, obs, uncert, global_mesh, global_active, sigma, tile_id, buffer=200.):
    print(f"Processing tile {tile_id}")
    local_survey = ip.Survey(source)
    electrodes = np.vstack((local_survey.locations_a,
                            local_survey.locations_b,
                            local_survey.locations_m,
                            local_survey.locations_n))
    local_survey.dobs = obs
    local_survey.std = uncert
    local_mesh = create_nested_mesh(
        electrodes, global_mesh, method="radial", max_distance=buffer
    )
    local_map = maps.TileMap(global_mesh, global_active, local_mesh)

    eta_map = maps.InjectActiveCells(
        local_mesh, indActive=local_map.local_active, valInactive=0.
    )
    act_map = maps.InjectActiveCells(
        local_mesh, indActive=local_map.local_active, valInactive=np.log(1e-8)
    )

    exp_map = maps.ExpMap(local_mesh)
    sigma_map = exp_map * act_map

    # Create the local misfit
    max_chunk_size = 256
    simulation = ip.Simulation3DCellCentered(
        local_mesh, survey=local_survey, etaMap=eta_map, sigma=sigma,
        store_sensitivities=True,
        Solver=Solver, max_ram=1
    )

    simulation.mesh = TensorMesh([1])  # Light dummy
    del local_mesh,
    local_map.local_mesh = None
    act_map.mesh = None
    #     expmap.mesh = None

    simulation.sensitivity_path = './sensitivity/Tile' + str(tile_id) + '/'
    data_object = data.Data(
        local_survey,
        dobs=obs,
        standard_deviation=uncert,
    )
    data_object.dobs = obs
    data_object.standard_deviation = uncert
    local_misfit = data_misfit.L2DataMisfit(
        data=data_object, simulation=simulation, model_map=local_map
    )
    local_misfit.W = 1 / uncert

    return local_misfit

cluster = LocalCluster(processes=False)
client = Client(cluster)

ws = Workspace("C:/Users/Benjamin/Downloads/FlinFlon_simulation.geoh5")
rx_obj = ws.get_entity("DC_Survey_new")[0]
tx_obj = ws.get_entity("DC_Survey_new (currents)")[0]
topo = ws.get_entity("TopoMT")[0].vertices

ab_id = np.unique(rx_obj.ab_cell_id.values).astype(int).tolist()
value_map = {value: key for key, value in rx_obj.ab_cell_id.value_map.map.items()}
src_lists = []
data_id = []
lines = {ii: {"sources": [], "data_id": []} for ii in np.unique(tx_obj.parts)}
for ab, cell in enumerate(tx_obj.cells.tolist()):

    rx_id = np.where(rx_obj.ab_cell_id.values.astype(int) == value_map[str(ab + 1)])[0]

    if len(rx_id) == 0:
        continue

    rx_M = rx_obj.vertices[rx_obj.cells[rx_id, 0]]
    rx_N = rx_obj.vertices[rx_obj.cells[rx_id, 1]]
    receivers = ip.receivers.Dipole(
        rx_M,
        rx_N
    )
    src_lists.append(
        ip.sources.Dipole(
            [receivers],
            tx_obj.vertices[cell[0]],
            tx_obj.vertices[cell[1]]
        )
    )
    line_id = tx_obj.parts[cell[0]]
    lines[line_id]["sources"].append(src_lists[-1])
    lines[line_id]["data_id"].append(rx_id)
    data_id.append(rx_id)

survey_ip = ip.Survey(src_lists)
data_id = np.hstack(data_id)

# Generate data
octree = ws.get_entity("DC_mesh")[0]
model = octree.get_data("Forward_con")[0]
mesh = octree_2_treemesh(octree)

chg = np.zeros_like(model.values)
medium_conductor_ind = model.values == 0.2
chg[medium_conductor_ind] = 0.1
strong_conductor_ind = model.values == 50
chg[strong_conductor_ind] = 0.05

activeCells = d_utils.active_from_xyz(mesh, topo)
mstart = chg[np.argsort(mesh._ubc_order)]
print(f"Size of starting model after active cells: {len(mstart)}")
sigma_model = model.values[np.argsort(mesh._ubc_order)]
survey_ip.drape_electrodes_on_topography(mesh, activeCells, option='top')

local_misfits = []
for ab_id, part in lines.items():
    local_misfits.append(
        create_tile_ip(
            part["sources"],
            np.ones_like(np.hstack(part["data_id"])).astype(float),
            np.ones_like(np.hstack(part["data_id"])).astype(float),
            mesh,
            activeCells,
            sigma_model,
            ab_id,
            buffer=200
        )
    )

local_misfits = client.gather(local_misfits)
global_misfit = objective_function.ComboObjectiveFunction(
    local_misfits
)

dpreds = []
client = get_client()
compute_J = False
m = mstart
for i, objfct in enumerate(global_misfit.objfcts):
    if hasattr(objfct, "simulation"):
        if objfct.model_map is not None:
            vec = objfct.model_map @ m
        else:
            vec = m

        print(f"Processing tile: {i}")
        future = client.compute(
            objfct.simulation.dpred(
                vec, compute_J=compute_J and (objfct.simulation._Jmatrix is None)
            ), workers=objfct.workers
        )
        dpreds += [future]

    else:
        dpreds += []

client.gather(dpreds)