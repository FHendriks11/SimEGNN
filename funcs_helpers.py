# %%
import os
import numpy as np
import warnings

def new_path(path, n=2, always_number=True):
    '''Create a new path for a file 
    with a filename that does not exist yet,
    by appending a number to path (before the extension) with at least n digits.
    (Higher n adds extra leading zeros). If always_number=True,
    the filename will always have a number appended, even if the
    bare filename does not exist either.'''

    # initial path is path + name + n zeros + extension if always_number=True
    # and just path + filename if always_number=False
    name, ext = os.path.splitext(path)
    if always_number:
        savepath = f'{name}_{str(0).zfill(n)}{ext}'
    else:
        savepath = path

    # If filename already exists, add number to it
    i = 1
    while os.path.exists(savepath):
        savepath = f'{name}_{str(i).zfill(n)}{ext}'
        i += 1
    return savepath

def split_trajs(trajs, n_sets, random_seed=42, split_sizes=None, 
        return_trajs=False):
    """Return n_sets arrays containing indices that split dataset, such that data points from the same trajectory (as indicated by the array trajs) end up in the same set.

    Parameters
    ----------
    trajs : (N,) array-like
        For each data point an id of the trajectory it belongs to
    n_sets : int
        Nr of sets to split the data into
    random_seed : int, optional
        By default 42
    split_sizes : (n_sets,) array_like or None, optional
        The relative size (measured in number of trajectories) of each split (will be normalized to sum to 1), if None all splits will be equal, by default None
    return_trajs : bool, optional
        If True, will return a tuple with the indices of data points per set as well as a tuple of the trajectories per set, by default False

    Returns
    -------
    inds : tuple of two lists of ndarrays
        For each set, the indices of data points in it.
    traj_splits : list of ndarrays
        For each set, the trajectories that are in it.
    
    Raises
    ------
    ValueError
        If n_sets is not an integer or less than 1.
    ValueError
        If split_sizes has negative values or if its length is not equal to n_sets.
    """    

    # sanity checks
    trajs = np.asarray(trajs)

    if split_sizes is None:
        split_sizes = np.ones(n_sets)
    else:
        split_sizes = np.asarray(split_sizes)

    if not isinstance(n_sets, int):
        raise ValueError(f'n_sets is of type {type(n_sets)}, must be an integer')
    if n_sets < 1:
        raise ValueError(f'n_sets must be at least 1')

    if (split_sizes < 0).any():
        raise ValueError(f'no negative split size allowed')
    if (len(split_sizes) != n_sets):
        raise ValueError(f'length of split_sizes (currently {len(split_sizes)}) must be equal to n_sets (currently {n_sets})')

    # get list of trajectory names
    unique_trajs, counts = np.unique(trajs, return_counts=True)

    # warn
    if np.max(counts) > 20*np.min(counts):
        warnings.warn(f'nr of data points per trajectory is very unbalanced! Max {np.max(counts)}, min {np.min(counts)}')

    # shuffle the trajectories
    rng = np.random.default_rng(seed=random_seed)
    rng.shuffle(unique_trajs)

    # split trajectories
    split_inds = np.cumsum(split_sizes/np.sum(split_sizes))[:-1]*len(unique_trajs)
    traj_splits = np.array_split(unique_trajs, indices_or_sections=split_inds.astype(int))
    traj_splits = [sorted(split) for split in traj_splits]

    # indices that split the data points
    bools = [np.isin(trajs, split) for split in traj_splits]  # one array of bools for each split, it indicates if a data point is in that split
    inds = [np.where(bool_temp)[0] for bool_temp in bools]  # for each split an array of ints that are the indices of data points in this split

    if return_trajs:
        return (inds, traj_splits)
    else:
        return inds
    
def orientation(triangles):
    """For each triangle, gives its orientation:
    0: points are collinear
    1: points are clockwise
    2: points are counterclockwise

    Parameters
    ----------
    triangles : ndarray of dimension >=2, with shape (..., 3, 2)
        2D Cartesian coordinates of points making up triangles.
        The second to last dimension indexes the three points of the triangle,
        last dimension indexes the x and y coordinate,
        broadcasted over remaining dimensions.

    Returns
    -------
    ndarray of integers (of same shape as input with the last two dimensions removed)
        orientation of each triangle, using broadcasting.
    """    
    # gives order of points of a triangle:
    # 0 : collinear
    # 1 : clockwise
    # 2 : counterclockwise

    triangles = np.asarray(triangles)
    
    p = triangles[..., 0, :]  # 1st point of each triangle
    q = triangles[..., 1, :]  # 2nd point of each triangle
    r = triangles[..., 2, :]  # 3rd point of each triangle
    val = (q[..., 1]-p[..., 1]) * (r[..., 0]-q[..., 0]) - (q[..., 0]-p[..., 0]) * (r[..., 1]-q[..., 1])

    val[val>0] = 1
    val[val<0] = 2

    return val.astype(int)

def between(vals):
    """"Check if value C is between values A and B.

    Parameters
    ----------
    vals : ndarray of dimension >=1, with shape (..., 3)
        The second to last dimension indexes the values A, B, C,
        broadcasted over remaining dimensions.

    Returns
    -------
    ndarray of booleans (of same shape as input with the last dimension removed)
        Whether value C is between values A and B (not equal to either), regardless of whether A is larger than B or vice versa.
    """    
    bools1 = vals[..., 2] < np.maximum(vals[..., 0], vals[..., 1])
    bools2 = vals[..., 2] > np.minimum(vals[..., 0], vals[..., 1])
    return bools1*bools2

def onSegment(points):
    """Check if a point C is on a line segment from point A to B.

    Parameters
    ----------
    vals : ndarray of dimension >=2, with shape (..., 3, 2)
        2D Cartesian coordinates of points defining pairs of line segments.
        The second to last dimension indexes the points A, B, C,
        last dimension indexes the x and y coordinate,
        broadcasted over remaining dimensions.

    Returns
    -------
    ndarray of booleans (of same shape as input with the last two dimensions removed)
        Whether C is on line segment AB, but not the same as either A or B.
    """    
    points = np.asarray(points)

    boolsx = between(points[..., 0])
    boolsy = between(points[..., 1])
    
    return boolsx + boolsy

def intersect(points):
    """Check whether a line segment from point A to point B
    intersects a line segment from point C to point D (in 2D Cartesian coordinates). (Only the end points coinciding does not count.)

    Parameters
    ----------
    points : ndarray of dimension >=2, with shape (..., 4, 2)
        2D Cartesian coordinates of points defining pairs of line segments.
        The second to last dimension indexes the four points A, B, C, D,
        last dimension indexes the x and y coordinate.

    Returns
    -------
    ndarray of booleans (of same shape as input with the last two dimensions removed)
        Whether the line segments intersect or not.
    """    
    points = np.asarray(points)  # shape [..., 4, 2]

    # different combinations of points to take as triangles
    combis = [[0,1,2], [0,1,3],[2,3,0],[2,3,1]]

    # orientation of each triangle
    ori = orientation(points[..., combis, :])

    # line segments cross and nothing is collinear
    bools = (ori[..., 0] != ori[..., 1])*(ori[..., 2] != ori[..., 3]) * (ori != 0).all(axis=-1)
    
    # collinear combinations
    bools_coll = ori == 0

    # any one of the combinations is collinear
    bools_coll2 = bools_coll.any(axis=-1)

    # all cases where 3 or more points are collinear
    coll_points = points[bools_coll2]

    # check overlap in case of collinear points
    bools_onSeg = onSegment(coll_points[..., combis, :])  

    # if collinear and onSegment, then also return True
    bools_coll[bools_coll2] *= bools_onSeg

    return bools + bools_coll.any(axis=-1)

def replace(arr, orig, repl, inplace=False):
    """In arr, replace the elements in orig by the elements in repl.

    Parameters
    ----------
    arr : array-like
        arr in which some values should be replaced
    orig : 1D array-like
        values in arr that should be replaced by the corresponding values in repl
    repl : 1D array-like of same length as orig
        values to replace the values of orig with
    inplace : bool, optional
        if False, return a changed copy of arr, by default False

    Returns
    -------
    array 
        arr with the values in orig replaced by the values in repl
    """

    if inplace:
        arr = np.asarray(arr)
    else:
        arr = np.copy(arr)

    orig = np.asarray(orig)
    repl = np.asarray(repl)

    if not len(orig) == len(repl):
        raise ValueError(f'orig and repl should have the same length, current lengths are {len(orig)} and {len(repl)}')

    # indices that sort orig (so we can use np.searchsorted)
    inds = np.argsort(orig)
    orig_sorted = orig[inds]
    repl_sorted = repl[inds]

    # find arr values in orig
    inds2 = np.searchsorted(orig_sorted, arr) % len(orig_sorted)

    # whether the values in arr are actually in orig
    bools = orig_sorted[inds2] == arr

    # replace the values of orig with those of repl
    arr[bools] = repl_sorted[inds2[bools]]

    if not inplace:
        return arr

# %%
# test
if __name__ == '__main__':
    print(split_trajs([0, 0, 1,2,3,4,0,1,2,3,2,3,4,5,6,6,6, 'test', 'test'], 2, split_sizes=(1,3), return_trajs=True))

    print(
    intersect([[[0,0],[4.0,4.0],[1,2],[3,2]],  # crossing --> True
            [[0,0],[4.0,4.0],[2,1],[3,2]],  # no crossing, parallel --> False
            [[0,0],[1.0,0.0],[2,0],[2,1]],  # collinear, no overlap --> False
            [[0,0],[1.0,0.0],[1,0],[1,1]],  # right angle point overlap --> False
            [[0,0],[1.0,0.0],[2,0],[0.5,1]],  # no crossing --> False
            [[0,0],[1.0,0.0],[2,0],[0.5,0]],  # collinear, overlap --> True 
            [[0,0],[1.0,0.0],[1,1],[1,-1]],  # end point on other segment --> True 
            ])
    )

