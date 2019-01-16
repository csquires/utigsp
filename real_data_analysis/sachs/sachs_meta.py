import causaldag as cd
import os
from config import PROJECT_FOLDER

nnodes = 11
true_dag = cd.DAG(nodes=set(range(nnodes)), arcs={
    (0, 1),
    (1, 5),
    (2, 3),
    (2, 8),
    (3, 8),
    (4, 2),
    (4, 3),
    (4, 6),
    (7, 0),
    (7, 1),
    (7, 5),
    (7, 6),
    (7, 9),
    (7, 10),
    (8, 0),
    (8, 1),
    (8, 9),
    (8, 10)
})
SACHS_FOLDER = os.path.join(PROJECT_FOLDER, 'real_data_analysis', 'sachs')
SACHS_DATA_FOLDER = os.path.join(SACHS_FOLDER, 'data')
ESTIMATED_FOLDER = os.path.join(SACHS_FOLDER, 'estimated')

