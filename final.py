import gym
import pix_main_arena
import time
import pybullet as p
import pybullet_data
import cv2
import os
import numpy as np
import math
import cv2.aruco as aruco

n = 12
position = np.arange(n*n).reshape(n,n).transpose()
matrix = np.zeros((n,n))
contours_of = {}
one_way = {}
hospital = {}
patient = {}
graph = {}
margin = 0
side = 0
pseudo_side = 0
avg_gap = 0
bot_error = 0

colors = {
"white": [np.array([0,0,210]), np.array([40,40,255])],
"green": [np.array([45,150,50]), np.array([65,255,255])],
"yellow": [np.array([25, 150, 50]), np.array([35, 255, 255])],
"red": [np.array([0,150,50]), np.array([10,255,255])],
"pink": [np.array([135,105,195]), np.array([265,135,225])],
"blue": [np.array([115,150,0]), np.array([125,255,255])]
}

cost = {
"white": 1,
"green": 2,
"yellow": 3,
"red": 4,
"pink": 1000,
"blue": 1001
}

env = gym.make("pix_main_arena-v0")
img = env.camera_feed()

def shape_of(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # M = cv2.moments(contour)
    area = cv2.contourArea(contour)
    if area == 0:
        return "point"
    perimeter = cv2.arcLength(contour, True)
    figure_constant = int(perimeter*perimeter/area)
    if figure_constant is 24 or len(approx) is 3:
        return 'triangle'

    elif figure_constant is 16 and len(approx) is 4:
        return 'square'

    elif figure_constant in [12, 13, 14, 15, 16] and len(approx) > 4:
        return 'circle'

    else:
        return 'not_found'

def find_contours():
    for color in colors:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, colors[color][0], colors[color][1])
        # result = cv2.bitwise_and(img, img, mask = mask)
        contours, _1 = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        to_delete = []
        for contour in contours:
            if shape_of(contour) != "square" and shape_of(contour) != "circle" and shape_of(contour) != "triangle":
                to_delete.append(contour)
            elif color != "blue" and (shape_of(contour) == "circle" or shape_of(contour) == "triangle"):
                to_delete.append(contour)

        contours = list(contours)

        for el in to_delete:
            contours.remove(el)

        contours = tuple(contours)

        contours_of[color] = contours

def center_of(contour):
    center_x = 0
    center_y = 0
    if len(contour[0]) == 2:
        for i in range(4):
            center_x += contour[i][0]
            center_y += contour[i][1]
        center_x /= 4
        center_y /= 4
        center = np.array([center_x, center_y])
        return center
    for i in range(len(contour)):
        center_x += contour[i][0][0]
        center_y += contour[i][0][1]
    center_x /= len(contour)
    center_y /= len(contour)
    center = np.array([center_x, center_y])
    return center

def get_parameters():
    global side
    global margin
    global pseudo_side
    global avg_gap

    find_contours()
    all_contours = ()
    for color in contours_of:
        if color == "blue":
            continue
        all_contours += contours_of[color]
    gap = {}
    sq_side = {}
    run = 0
    previous_center = center_of(contours_of[color][0])
    for contour in all_contours:
        run += 1
        if run <= 1:
            continue
        if shape_of(contour) == "square":
            perimeter = cv2.arcLength(contour, True)
            side = perimeter/4
            if side in sq_side:
                sq_side[side] += 1
            else:
                sq_side[side] = 0
                sq_side[side] += 1
            center = center_of(contour)

            if math.isclose(center[0], previous_center[0], abs_tol = 2) and abs(center[1] - previous_center[1]) > side and abs(center[1] - previous_center[1]) < 2*side:
                center_distance = abs(center[1] - previous_center[1])
                square_distance = center_distance - side
                if square_distance in gap:
                    gap[square_distance] += 1
                else:
                    gap[square_distance] = 0
                    gap[square_distance] += 1

            elif math.isclose(center[1], previous_center[1], abs_tol = 2) and abs(center[0] - previous_center[0]) > side and abs(center[0] - previous_center[0]) < 2*side:
                center_distance = abs(center[0] - previous_center[0])
                square_distance = center_distance - side
                if square_distance in gap:
                    gap[square_distance] += 1
                else:
                    gap[square_distance] = 0
                    gap[square_distance] += 1

            # else:
            previous_center = center

    avg_gap = 0
    total = 0
    for key, value in gap.items():
        avg_gap += key*value
        total += value
    avg_gap /= total

    avg_side = 0
    total = 0
    for key, value in sq_side.items():
        avg_side += key*value
        total += value
    avg_side /= total

    side_of_img = (img.shape[0]+img.shape[1])/2
    pseudo_side = avg_side + avg_gap

    __margin = (side_of_img - n*pseudo_side)/2

    margin = __margin
    side = avg_side
    pseudo_side = avg_gap + side

    return __margin, avg_side, avg_gap


def error_free(P, C):
    m = (C[1]-P[1])/(C[0]-P[0])
    sec = math.sqrt((m*m) + 1)
    to_add = pseudo_side/sec/math.sqrt(2)
    if P[0] > C[0]:
        P[0] = C[0] + to_add
        P[1] = C[1] + (m*to_add)
    else:
        P[0] = C[0] - to_add
        P[1] = C[1] - (m*to_add)

    return np.array(P)

ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
def bot(key="error-free"):
    image = env.camera_feed()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids,_ = aruco.detectMarkers(gray, ARUCO_DICT, parameters = ARUCO_PARAMETERS)
    res = corners[0][0]

    if key == "with_margin":
        return res
    res -= margin
    bc = center_of(res)
    if key != "error-free":
        return res

    res[0] = error_free(res[0], bc)
    res[1] = error_free(res[1], bc)
    res[2] = error_free(res[2], bc)
    res[3] = error_free(res[3], bc)

    if bot_error > 0:
        x_c = n/2*pseudo_side
        x_error = (bc[0]-x_c)*bot_error/x_c
        y_c = n/2*pseudo_side
        y_error = (bc[1]-x_c)*bot_error/x_c
        for i in range(4):
            res[i][0] -= x_error
            res[i][1] -= y_error

    return res

def i_index(y):
    y //= pseudo_side
    return int(y)

def j_index(x):
    x //= pseudo_side
    return int(x)

def x_point(j, key="pseudo"):
    j *= pseudo_side
    j += pseudo_side/2
    if key != "pseudo":
        j += margin
    return j

def y_point(i, key="pseudo"):
    i *= pseudo_side
    i += pseudo_side/2
    if key != "pseudo":
        i += margin
    return i

def fill_matrix():
    for color, contours in contours_of.items():
        if color == "blue":
            continue

        for contour in contours:
            if shape_of(contour) != "square":
                print("Error: not square, expected square")
            remove_margin(contour)
            center = center_of(contour)
            matrix[i_index(center[1])][j_index(center[0])] = cost[color]

    corners = bot()
    center = center_of(corners)
    matrix[i_index(center[1])][j_index(center[0])] = 1

def remove_margin(contour):
    for i in range(len(contour)):
        contour[i][0][0] -= margin
        contour[i][0][1] -= margin

def manage_blues_and_pink():
    for contour in contours_of["blue"]:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        contour = approx
        remove_margin(contour)
        center = center_of(contour)
        if shape_of(contour) == "triangle":
            for i in range(3):
                if math.isclose(center[0], contour[i][0][0], abs_tol = 2):
                    pos = position[i_index(center[1])][j_index(center[0])]
                    if center[1] < contour[i][0][1]:
                        one_way[pos] = pos+1
                    elif center[1] > contour[i][0][1]:
                        one_way[pos] = pos-1
                    break
                elif math.isclose(center[1], contour[i][0][1], abs_tol = 2):
                    pos = position[i_index(center[1])][j_index(center[0])]
                    if center[0] < contour[i][0][0]:
                        one_way[pos] = pos+n
                    elif center[0] > contour[i][0][0]:
                        one_way[pos] = pos-n
                    break
        elif shape_of(contour) == "square":
            matrix[i_index(center[1])][j_index(center[0])] = cost["blue"]
            hospital["square"] = position[i_index(center[1])][j_index(center[0])]

        elif shape_of(contour) == "circle":
            matrix[i_index(center[1])][j_index(center[0])] = cost["blue"]
            hospital["circle"] = position[i_index(center[1])][j_index(center[0])]

        else:
            print("Error: not expected")
    for contour in contours_of["pink"]:
        center = center_of(contour)
        patient[position[i_index(center[1])][j_index(center[0])]] = 0

def make_graph():

    for i in range(n*n):
        graph.update({i:{}})

    for j in range(n):
        for i in range(n):
            if matrix[i][j] != 0:
                if (j+1 < n) and (matrix[i][j+1] != 0):
                    graph[position[i][j]].update({position[i][j+1]:matrix[i][j+1]})
                if i+1 < n and (matrix[i+1][j] != 0):
                    graph[position[i][j]].update({position[i+1][j]:matrix[i+1][j]})
                if j-1 >= 0 and (matrix[i][j-1] != 0):
                    graph[position[i][j]].update({position[i][j-1]:matrix[i][j-1]})
                if i-1 >= 0 and (matrix[i-1][j] != 0):
                    graph[position[i][j]].update({position[i-1][j]:matrix[i-1][j]})

    for one, way in one_way.items():
        del graph[way][one]
        graph[one] = {way : graph[one][way]}

def dij(graph, start, goal):
    shortest_distance = {}
    track_predecessor = {}
    unseenNodes = graph
    infinity = 999999
    track_path = []
    new = {}

    for node in unseenNodes:
        shortest_distance[node] = infinity
    shortest_distance[start] = 0

    while unseenNodes:
        min_distance_node = None

        for node in unseenNodes:

            if min_distance_node is None:
                min_distance_node = node

            elif shortest_distance[node] < shortest_distance[min_distance_node]:
                min_distance_node = node

        path_options = graph[min_distance_node].items()

        for child_node, weight in path_options:
            if weight + shortest_distance[min_distance_node] < shortest_distance[child_node]:

                shortest_distance[child_node] = weight + shortest_distance[min_distance_node]
                track_predecessor[child_node] = min_distance_node

        new[min_distance_node] = unseenNodes.pop(min_distance_node)

    currentNode = goal

    while currentNode != start:
        try:
            track_path.insert(0, currentNode)
            currentNode = track_predecessor[currentNode]

        except KeyError:

            print("path is not reachable")
            break

    track_path.insert(0, start)

    if shortest_distance[goal] != infinity:
        #print("cost for the obtained least-cost path :", shortest_distance[goal])
        return track_path, new, shortest_distance[goal]

#__________________________________________________________________________________________________________________

def cost_patient():
    global graph
    pos = position_of_bot()
    for pat in patient:
        dij_ = dij(graph, pos, pat)
        patient[pat] = dij_[2]
        graph = dij_[1]

def position_of_bot():
    center = center_of(bot())
    return position[i_index(center[1])][i_index(center[0])]

def distance(position):
    pos_i = position%n
    pos_j = position//n
    pos_x = x_point(pos_j)
    pos_y = y_point(pos_i)
    bc = center_of(bot())
    dis = (pos_x-bc[0])*(pos_x-bc[0]) + (pos_y-bc[1])*(pos_y-bc[1])
    dis = math.sqrt(dis)
    return dis

def bot_is_on(position, abs_tol=0.5):
    dis = distance(position)
    if dis <= abs_tol:
        return True
    return False

def block(position):
    x = x_point(position//n)
    y = y_point(position%n)
    c = pseudo_side/2
    res = [[x-c, y+c],[x-c, y-c],[x+c, y-c],[x+c, y+c]]
    return res

def solve_error():
    global bot_error
    bot_error = center_of(bot())[0] - x_point(11)

def straight(position, back=True):
    corner = bot()
    if math.isclose(corner[0][0], corner[1][0], abs_tol = 3):
        index = 0
    else:
        index = 1

    if index == 1:
        position %= n
    else:
        position //= n
    if corner[1][index] > corner[2][index]:
        greater = True
    else:
        greater = False


    while True:
        run = 1
        while run%500:
            p.stepSimulation()
            env.move_husky(0.5, 0.5, 0.5, 0.5)
            run += 1

        if greater:
            if center_of(bot())[index] >= x_point(position):
                if back:
                    while center_of(bot())[index] > x_point(position):
                        p.stepSimulation()
                        env.move_husky(-0.5, -0.5, -0.5, -0.5)
                break
        else:
            if center_of(bot())[index] <= x_point(position):
                if back:
                    while center_of(bot())[index] < x_point(position):
                        p.stepSimulation()
                        env.move_husky(-1, -1, -1, -1)
                break

def right(back = False):
    i = 140
    corner = bot()
    if math.isclose(corner[0][0], corner[1][0], abs_tol = 3):
        index = 1
    else:
        index = 0
    position = position_of_bot()
    if index == 1:
        position %= n
    else:
        position //= n
    if corner[0][index] > corner[1][index]:
        greater = True
    else:
        greater = False
    while True:
        run = 1
        while run%1300:
            p.stepSimulation()
            env.move_husky(0.2, 0, 0.2, 0)
            run += 1
        run = 1
        while run%1000:
            p.stepSimulation()
            env.move_husky(0, 0, 0, 0)
            run += 1
        run = 1
        while run%1000:
            p.stepSimulation()
            env.move_husky(0, -0.2, 0, -0.2)
            run += 1
        run = 1
        while run%1000:
            p.stepSimulation()
            env.move_husky(0, 0, 0, 0)
            run += 1
        corner = bot()
        if greater:
            if corner[0][index] <= corner[1][index]:
                while corner[0][index] < corner[1][index]:
                    corner = bot()

                    p.stepSimulation()
                    env.move_husky(-0.5, 0.5, -0.5, 0.5)

                if back == True:
                    while not math.isclose(center_of(bot())[index], y_point(position), abs_tol = 0.5):
                        run = 1
                        while run%10:
                            p.stepSimulation()
                            env.move_husky(-1, -1, -1, -1)
                            run += 1
                break
        else:
            if corner[0][index] >= corner[1][index]:
                while corner[0][index] > corner[1][index]:
                    corner = bot()

                    p.stepSimulation()
                    env.move_husky(-0.5, 0.5, -0.5, 0.5)

                if back == True:
                    while not math.isclose(center_of(bot())[index], y_point(position), abs_tol = 0.5):
                        run = 1
                        while run%10:
                            p.stepSimulation()
                            env.move_husky(-1, -1, -1, -1)
                            run += 1
                break

def left(back = False):
    corner = bot()
    if math.isclose(corner[0][0], corner[1][0], abs_tol = 3):
        index = 1
    else:
        index = 0
    position = position_of_bot()
    if index == 1:
        position %= n
    else:
        position //= n
    if corner[0][index] > corner[1][index]:
        greater = True
    else:
        greater = False
    while True:
        run = 1
        while run%1300:
            p.stepSimulation()
            env.move_husky(0, 0.2, 0, 0.2)
            run += 1
        run = 1
        while run%1000:
            p.stepSimulation()
            env.move_husky(0, 0, 0, 0)
            run += 1
        run = 1
        while run%1000:
            p.stepSimulation()
            env.move_husky(-0.2, 0, -0.2, 0)
            run += 1
        run = 1
        while run%1000:
            p.stepSimulation()
            env.move_husky(0, 0, 0, 0)
            run += 1
        corner = bot()
        if greater:
            if corner[0][index] <= corner[1][index]:
                while corner[0][index] < corner[1][index]:
                    corner = bot()

                    p.stepSimulation()
                    env.move_husky(0.5, -0.5, 0.5, -0.5)

                if back == True:
                    while not math.isclose(center_of(bot())[index], y_point(position), abs_tol = 0.5):
                        run = 1
                        while run%10:
                            p.stepSimulation()
                            env.move_husky(-1, -1, -1, -1)
                            run += 1
                break
        else:
            if corner[0][index] >= corner[1][index]:
                while corner[0][index] > corner[1][index]:
                    corner = bot()

                    p.stepSimulation()
                    env.move_husky(0.5, -0.5, 0.5, -0.5)

                if back == True:
                    while not math.isclose(center_of(bot())[index], y_point(position), abs_tol = 0.5):
                        run = 1
                        while run%10:
                            p.stepSimulation()
                            env.move_husky(-1, -1, -1, -1)
                            run += 1
                break

def motion(path):
    print(path)
    length = len(path)
    for i in range(length):
        corner = bot()
        if math.isclose(corner[0][0], corner[1][0], abs_tol = 3):
            if corner[1][0] > corner[2][0]:
                orientation = "right"
            else:
                orientation = "left"
        elif math.isclose(corner[0][1], corner[1][1], abs_tol = 3):
            if corner[1][1] > corner[2][1]:
                orientation = "down"
            else:
                orientation = "up"
        step = path[i]
        if position_of_bot == step:
            print(step)
            continue

        elif orientation == "right":
            if step == position_of_bot() + 1:
                print("right")
                right()
                if i+1 < length and path[i+1] == step + 1:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() - 1:
                print("left")
                left()
                if i+1 < length and path[i+1] == step - 1:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() + n:
                print("straight")
                if i+1 < length and path[i+1] == step + n:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() - n:
                print("U-Turn")
                right(True)
                right()
                if i+1 < length and path[i+1] == step - n:
                    straight(step, False)
                else:
                    straight(step)

        elif orientation == "left":
            if step == position_of_bot() + 1:
                print("left")
                left()
                if i+1 < length and path[i+1] == step + 1:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() - 1:
                print("right")
                right()
                if i+1 < length and path[i+1] == step - 1:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() + n:
                print("U-Turn")
                right(True)
                right()
                if i+1 < length and path[i+1] == step + n:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() - n:
                print("straight")
                if i+1 < length and path[i+1] == step - n:
                    straight(step, False)
                else:
                    straight(step)

        elif orientation == "up":
            if step == position_of_bot() + 1:
                print("U-Turn")
                right(True)
                right()
                if i+1 < length and path[i+1] == step + 1:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() - 1:
                print("straight")
                if i+1 < length and path[i+1] == step - 1:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() + n:
                print("right")
                right()
                if i+1 < length and path[i+1] == step + n:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() - n:
                print("left")
                left()
                if i+1 < length and path[i+1] == step - n:
                    straight(step, False)
                else:
                    straight(step)
        elif orientation == "down":
            if step == position_of_bot() + 1:
                print("straight")
                if i+1 < length and path[i+1] == step + 1:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() - 1:
                print("U-Turn")
                left(True)
                left()
                if i+1 < length and path[i+1] == step - 1:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() + n:
                print("left")
                left()
                if i+1 < length and path[i+1] == step + n:
                    straight(step, False)
                else:
                    straight(step)
            elif step == position_of_bot() - n:
                print("right")
                right()
                if i+1 < length and path[i+1] == step - n:
                    straight(step, False)
                else:
                    straight(step)

def prepare():
    get_parameters()
    fill_matrix()
    manage_blues_and_pink()
    make_graph()
    solve_error()

def run():
    global graph
    global patient
    while len(patient) > 0:
        cost_patient()
        patient = dict(sorted(patient.items(), key = lambda kv:(kv[1], kv[0])))
        for key, value in patient.items():
            pat = key
            break
        print(pat)
        #print(graph)
        path_graph = dij(graph, position_of_bot(), pat)
        path = path_graph[0]
        graph = path_graph[1]

        final_position = path.pop()

        motion(path)

        env.remove_cover_plate(final_position%n,final_position//n)

        img2 = env.camera_feed()

        end = [path.pop(), final_position]
        motion(end)

        x_0 = x_point(final_position//n, "exact")
        y_0 = y_point(final_position%n, "exact")
        x_0 -= pseudo_side/2
        y_0 -= pseudo_side/2

        crop2 = img2[int(y_0 + 0.5) : int(y_0 + pseudo_side + 0.5), int(x_0 + 0.5) : int(x_0 + pseudo_side + 0.5)]
        hsv2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)

        mask2 = cv2.inRange(hsv2, colors["blue"][0], colors["blue"][1])
        blue = cv2.bitwise_and(crop2, crop2, mask=mask2)

        contou, _1 = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if shape_of(contou[0]) == "square":
            path_graph = dij(graph, position_of_bot(), hospital["square"])
            path = path_graph[0]
            graph = path_graph[1]
        else:
            path_graph = dij(graph, position_of_bot(), hospital["circle"])
            path = path_graph[0]
            graph = path_graph[1]

        motion(path)
        patient.pop(pat)

prepare()
run()