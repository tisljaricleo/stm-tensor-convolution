def initialize_metadata():
    global AUTHOR
    global LICENCE
    global EMAIL
    global STATUS
    global STATUS
    global DOCFORMAT

    AUTHOR = "Leo Tisljaric"
    LICENCE = "GNU General Public License v3.0"
    EMAIL = "tisljaricleo@gmail.com"
    STATUS = "Development"
    DOCFORMAT = "reStructuredText"


def initialize_paths():
    """
    Paths for loading and saving data.
    :return: void
    """
    global DATA_PATH
    global LINKS_SPEED_LIMIT_PATH
    global ROUTES_PATH
    global ALL_TRANSITIONS_PKL_NAME
    global LIST_TRANSITIONS_PKL_NAME
    global SPATIAL_MATRIX_PKL_NAME

    # DATA_PATH = r'D:\DATA_\jadranski_most\okolica_jadranski.txt'    # Path to CSV raw data.
    # DATA_PATH = r'D:\DATA_\veliki_graf\rute.txt'
    DATA_PATH = r'D:\DATA_\Sordito_ZAGREB\data_zg.txt'
    LINKS_SPEED_LIMIT_PATH = r'speed_limits_processed.csv'
    ROUTES_PATH = r'outputs\routes'  # Name to save routes pickle.
    ALL_TRANSITIONS_PKL_NAME = r'outputs\transitions.pkl'   # Name for saving all transitions as pickle.
    LIST_TRANSITIONS_PKL_NAME = r'outputs\list_of_transitions.pkl'   # Name for saving list of transitions as pickle.
    SPATIAL_MATRIX_PKL_NAME = r'outputs\spatial_matrix.pkl'


def initialize_stm_setup():
    """
    Speed transition matrix setup.
    :return:
    """
    global RESOLUTION
    global MAX_INDEX
    global MAX_ITER
    global SPEED_LIST
    global SPEED_TYPE
    global SPEED_LIMIT_TRESH
    global SL_DOWN
    global SL_UP
    global DIAG_LOCS

    RESOLUTION = int(5)  # Resolution of the speed transition matrix in km/h
    MAX_INDEX = int(100 / RESOLUTION)  # Maximum index of the numpy array.
    MAX_ITER = int(100 + RESOLUTION)  # Maximal iteration for the range() function.
    SPEED_LIST = list(range(RESOLUTION, MAX_ITER, RESOLUTION))  # All speed values for rows/columns of the matrix.
    SPEED_TYPE = 'rel'
    SL_DOWN = 50
    SL_UP = 80
    SPEED_LIMIT_TRESH = 50

    diag_locs = []
    for i in range(0, MAX_INDEX):
        for j in range(0, MAX_INDEX):
            if i == j:
                diag_locs.append((i, j))
    DIAG_LOCS = diag_locs



def initialize_db_setup():
    global DB_NAME
    global ROUTE_COLLECTION
    global TRANSITION_COLLECTION
    global SM_COLLECTION
    global TENSOR_COLLECTION

    DB_NAME = 'SpeedTransitionDB'
    ROUTE_COLLECTION = 'routesNEW'
    TRANSITION_COLLECTION = 'transitionsNEW'
    SM_COLLECTION = 'spatialMatrixRWLNEWrel'
    TENSOR_COLLECTION = 'spatialTensors'
