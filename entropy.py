import coord as coord
import cubie as cubie
import twophase.solver as sv

# generate a random cube
cube = cubie.CubieCube()

'''

basicMoveCube[Color.U] = CubieCube(cpU, coU, epU, eoU)
basicMoveCube[Color.R] = CubieCube(cpR, coR, epR, eoR)
basicMoveCube[Color.F] = CubieCube(cpF, coF, epF, eoF)
basicMoveCube[Color.D] = CubieCube(cpD, coD, epD, eoD)
basicMoveCube[Color.L] = CubieCube(cpL, coL, epL, eoL)
basicMoveCube[Color.B] = CubieCube(cpB, coB, epB, eoB)

0 - U
1 - U2
2 - U'
3 - R
4 - R2
5 - R'
6 - F
7 - F2
8 - F'
9 - D
10 - D2
11 - D'
12 - L
13 - L2
14 - L'
15 - B
16 - B2
17 - B'
'''

print(cube)

def to_2d(cube):
    return cube.to_facelet_cube().to_2dstring()

def convert_sol_to_move(s):
    move_list = s.split(' ')[:-1]
    ret_list = []
    for i in move_list:
        if i == 'U1':
            ret_list.append(0)
        elif i == 'U2':
            ret_list.append(1)
        elif i == "U3":
            ret_list.append(2)
        elif i == 'R1':
            ret_list.append(3)
        elif i == 'R2':
            ret_list.append(4)
        elif i == "R3":
            ret_list.append(5)
        elif i == 'F1':
            ret_list.append(6)
        elif i == 'F2':
            ret_list.append(7)
        elif i == "F3":
            ret_list.append(8)
        elif i == 'D1':
            ret_list.append(9)
        elif i == 'D2':
            ret_list.append(10)
        elif i == "D3":
            ret_list.append(11)
        elif i == 'L1':
            ret_list.append(12)
        elif i == 'L2':
            ret_list.append(13)
        elif i == "L3":
            ret_list.append(14)
        elif i == 'B1':
            ret_list.append(15)
        elif i == 'B2':
            ret_list.append(16)
        elif i == "B3":
            ret_list.append(17)
    return ret_list

def convert_to_move(s):
    move_list = s.split(' ')
    ret_list = []
    for i in move_list:
        if i == 'U':
            ret_list.append(0)
        elif i == 'U2':
            ret_list.append(1)
        elif i == "U'":
            ret_list.append(2)
        elif i == 'R':
            ret_list.append(3)
        elif i == 'R2':
            ret_list.append(4)
        elif i == "R'":
            ret_list.append(5)
        elif i == 'F':
            ret_list.append(6)
        elif i == 'F2':
            ret_list.append(7)
        elif i == "F'":
            ret_list.append(8)
        elif i == 'D':
            ret_list.append(9)
        elif i == 'D2':
            ret_list.append(10)
        elif i == "D'":
            ret_list.append(11)
        elif i == 'L':
            ret_list.append(12)
        elif i == 'L2':
            ret_list.append(13)
        elif i == "L'":
            ret_list.append(14)
        elif i == 'B':
            ret_list.append(15)
        elif i == 'B2':
            ret_list.append(16)
        elif i == "B'":
            ret_list.append(17)
    return ret_list

def op_cube(s):
    move_list = s.split(' ')
    print(move_list)
    for i in move_list:
        if i == 'U':
            cube.multiply(cubie.moveCube[0])
        elif i == 'U2':
            cube.multiply(cubie.moveCube[1])
        elif i == "U'":
            cube.multiply(cubie.moveCube[2])
        elif i == 'R':
            cube.multiply(cubie.moveCube[3])
        elif i == 'R2':
            cube.multiply(cubie.moveCube[4])
        elif i == "R'":
            cube.multiply(cubie.moveCube[5])
        elif i == 'F':
            cube.multiply(cubie.moveCube[6])
        elif i == 'F2':
            cube.multiply(cubie.moveCube[7])
        elif i == "F'":
            cube.multiply(cubie.moveCube[8])
        elif i == 'D':
            cube.multiply(cubie.moveCube[9])
        elif i == 'D2':
            cube.multiply(cubie.moveCube[10])
        elif i == "D'":
            cube.multiply(cubie.moveCube[11])
        elif i == 'L':
            cube.multiply(cubie.moveCube[12])
        elif i == 'L2':
            cube.multiply(cubie.moveCube[13])
        elif i == "L'":
            cube.multiply(cubie.moveCube[14])
        elif i == 'B':
            cube.multiply(cubie.moveCube[15])
        elif i == 'B2':
            cube.multiply(cubie.moveCube[16])
        elif i == "B'":
            cube.multiply(cubie.moveCube[17])
    return cube
    
def cal_entropy(cube):
    cubie_num = 0
    for i in range(8):
        if cube.cp[i] != i:
            cubie_num += 1
        else:
            if cube.co[i] != 0:
                cubie_num += 1
    for i in range(12):
        if cube.ep[i] != i:
            cubie_num += 1
        else:
            if cube.eo[i] != 0:
                cubie_num += 1
    return cubie_num

def cal_face_entropy(cube):
    s = cube.to_facelet_cube().to_string()
    
    print(len(cube.to_facelet_cube().to_string())) 
    
    
def solve(cube):
    return sv.solve(cube.to_facelet_cube().to_string(), 0, 3)

mv = convert_to_move("R B' U F' L U' D R' D2 F' U2 F2 L2 D2 B2 R2 F L2 D2 F U")
for i in mv:
    cube.multiply(cubie.moveCube[i])
print(to_2d(cube))
cal_face_entropy(cube)
mv = convert_sol_to_move(solve(cube))
print(mv)

print(cal_entropy(cube))
for i in mv:
    cube.multiply(cubie.moveCube[i])
    print(cal_entropy(cube))