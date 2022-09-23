import numpy as np


def prepad_zeros(demos, max_length=400):

    vector_size = demos[0][0].shape[0]

    pre_padded_demos = np.zeros((len(demos), max_length, vector_size))

    for j,demo in enumerate(demos):

        demo_length = len(demo)
        ind_to_start = max_length - demo_length

        for i in range(len(demo)):
            pre_padded_demos[j][ind_to_start + i] = demo[i]

    return np.array(pre_padded_demos)


def select_obs_act_in_episode(demos):

    demos_reformatted = [[np.append(demo['obs'][i], demo['act'][i]) for i,_ in enumerate(demo['act'])] for demo in demos]

    return np.array(demos_reformatted)




'''above_dict = {
    0:[0,1],
    1:[1,0],
    2:[0,2],
    3:[2,0],
    4:[1,2],
    5:[2,1]
}'''


above_dict = {
0:[0, 1],
1:[0, 2],
2:[1, 0],
3:[1, 2],
4:[2, 0],
5:[2, 1]
}


close_dict = {
    0:[0,1],
    1:[0,2],
    2:[1,2]
}


def close2_far1(goal, cube1, cube2, cube3):

    if goal[0] == 1:
        # close(cube1, cube2)
        cube_a = cube1
        cube_b = cube2

    if goal[1] == 1:
        # close(cube1, cube3)
        cube_a = cube1
        cube_b = cube3

    if goal[2] == 1:
        # close(cube2, cube3)
        cube_a = cube2
        cube_b = cube3

    return cube_a, cube_b

def close2a2(goal, cube1, cube2, cube3):

    if goal[0] == 1 and goal[1] == 1:

        cube_central = cube1
        cube_a = cube2
        cube_b = cube3

    if goal[0] == 1 and goal[2] == 1:

        cube_central = cube2
        cube_a = cube1
        cube_b = cube3


    if goal[1] == 1 and goal[2] == 1:

        cube_central = cube3
        cube_a = cube2
        cube_b = cube1

    return cube_central, cube_a, cube_b


def all3close(goal, cube1, cube2, cube3):

    shuffle = np.array([cube1, cube2, cube3])
    #np.random.shuffle(shuffle)

    cube_a = shuffle[0]
    cube_b = shuffle[1]
    cube_c = shuffle[2]

    return cube_a, cube_b, cube_c


def stack_1far(goal, cube1, cube2, cube3):

    above_predicates = goal[3:]

    ind = np.where(np.array(above_predicates)==1)

    cubes = [cube1, cube2, cube3]

    cubes_ind = above_dict[ind[0][0]]

    cube_a = cubes[cubes_ind[0]]
    cube_b = cubes[cubes_ind[1]] 

    return cube_a, cube_b

def stack_1close(goal, cube1, cube2, cube3):

    cubes = [cube1, cube2, cube3]

    above_predicates = goal[3:]

    ind = np.where(np.array(above_predicates)==1)

    cubes_ind = above_dict[ind[0][0]]

    cube_a = cubes[cubes_ind[0]]
    cube_b = cubes[cubes_ind[1]]

    if 2 not in [cubes_ind[0], cubes_ind[1]]:
        last_cube_ind = 2
    elif 1 not in [cubes_ind[0], cubes_ind[1]]:
        last_cube_ind = 1
    elif 0 not in [cubes_ind[0], cubes_ind[1]]:
        last_cube_ind = 0

    cube_close = cubes[last_cube_ind]

    return cube_a, cube_b, cube_close

def pyramid(goal, cube1, cube2, cube3):

    cubes = [cube1, cube2, cube3]

    above_predicates = goal[3:]

    ind = np.where(np.array(above_predicates)==1)

    pair1 = above_dict[ind[0][0]]
    pair2 = above_dict[ind[0][1]]

    if 2 in pair1 and 2 in pair2:
        cube_close_a = cubes[0]
        cube_close_b = cubes[1]
        cube_above = cubes[2]

    if 1 in pair1 and 1 in pair2:
        cube_close_a = cubes[0]
        cube_close_b = cubes[2]
        cube_above = cubes[1]

    if 0 in pair1 and 0 in pair2:
        cube_close_a = cubes[2]
        cube_close_b = cubes[1]
        cube_above = cubes[0]

    return cube_close_a, cube_close_b, cube_above

def stack3(goal, cube1, cube2, cube3):

    cubes = [cube1, cube2, cube3]

    above_predicates = goal[3:]

    ind = np.where(np.array(above_predicates)==1)

    pair1 = above_dict[ind[0][0]]
    pair2 = above_dict[ind[0][1]]

    if 2 in pair1 and 2 in pair2:

        cube_middle = cubes[2]

        if 2 == pair1[1]:
            cube_above = cubes[pair1[0]]
            cube_below = cubes[pair2[1]]
        else:
            cube_above = cubes[pair2[0]]
            cube_below = cubes[pair1[1]]

    elif 1 in pair1 and 1 in pair2:

        cube_middle = cubes[1]

        if 1 == pair1[1]:
            cube_above = cubes[pair1[0]]
            cube_below = cubes[pair2[1]]
        else:
            cube_above = cubes[pair2[0]]
            cube_below = cubes[pair1[1]]

    elif 0 in pair1 and 0 in pair2:

        cube_middle = cubes[0]

        if 0 == pair1[1]:
            cube_above = cubes[pair1[0]]
            cube_below = cubes[pair2[1]]
        else:
            cube_above = cubes[pair2[0]]
            cube_below = cubes[pair1[1]]


    return cube_above, cube_middle, cube_below

