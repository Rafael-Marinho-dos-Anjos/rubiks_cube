"""Rubik's cube module"""

import numpy as np
from enum import Enum
from copy import deepcopy
import cv2
from win32api import GetSystemMetrics


class Colors(Enum):
    WHITE = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    ORANGE = 4
    YELLOW = 5


class Moves(Enum):
    R1 = 0
    R2 = 1
    R3 = 2
    L1 = 3
    L2 = 4
    L3 = 5
    U1 = 6
    U2 = 7
    U3 = 8
    D1 = 9
    D2 = 10
    D3 = 11
    F1 = 12
    F2 = 13
    F3 = 14
    B1 = 15
    B2 = 16
    B3 = 17

    def is_right(num) -> bool:
        num = num.value if isinstance(num, Moves) else num
        
        return num // 3 == 0
    
    def is_left(num) -> bool:
        num = num.value if isinstance(num, Moves) else num
        
        return num // 3 == 1
    
    def is_top(num) -> bool:
        num = num.value if isinstance(num, Moves) else num
        
        return num // 3 == 2
    
    def is_down(num) -> bool:
        num = num.value if isinstance(num, Moves) else num
        
        return num // 3 == 3
    
    def is_front(num) -> bool:
        num = num.value if isinstance(num, Moves) else num
        
        return num // 3 == 4
    
    def is_back(num) -> bool:
        num = num.value if isinstance(num, Moves) else num
        
        return num // 3 == 5
    
    def repetitions(num) -> int:
        num = num.value if isinstance(num, Moves) else num
        
        return num % 3 + 1
    
    def notation(num):
        num = num.value if isinstance(num, Moves) else num

        if Moves.is_right(num):
            nttn = "R"

        elif Moves.is_left(num):
            nttn = "L"

        elif Moves.is_top(num):
            nttn = "U"

        elif Moves.is_down(num):
            nttn = "D"

        elif Moves.is_front(num):
            nttn = "F"

        elif Moves.is_back(num):
            nttn = "B"

        if Moves.repetitions(num) == 2:
            nttn += "2"

        elif Moves.repetitions(num) == 3:
            nttn += "_"
        
        return nttn
    
    def is_same_side(side_1, side_2):
        side_1 = side_1.value if isinstance(side_1, Moves) else side_1
        side_2 = side_1.value if isinstance(side_2, Moves) else side_2
        return side_1 // 3 == side_2 // 3
    
    def is_opposite_side(side_1, side_2):
        side_1 = side_1.value if isinstance(side_1, Moves) else side_1
        side_2 = side_1.value if isinstance(side_2, Moves) else side_2
        if not Moves.is_same_side(side_1, side_2):
            return side_1 // 6 == side_2 // 6
        return False


class Cube():
    def __init__(self, get_colors = False) -> None:
        """
            face_top = White
            face_front = Red
            face_left = Green
            face_right = Blue
            face_back = Orange
            face_bottom = Yellow
        """
        if get_colors:
            self._select_colors()
        else:
            self.face_top = np.array(
                [[Colors.WHITE.value] * 3] * 3,
                 dtype=np.int8)
            
            self.face_front = np.array(
                [[Colors.RED.value] * 3] * 3,
                 dtype=np.int8)
            
            self.face_left = np.array(
                [[Colors.GREEN.value] * 3] * 3,
                 dtype=np.int8)
            
            self.face_right = np.array(
                [[Colors.BLUE.value] * 3] * 3,
                 dtype=np.int8)
            
            self.face_back = np.array(
                [[Colors.ORANGE.value] * 3] * 3,
                 dtype=np.int8)
            
            self.face_bottom = np.array(
                [[Colors.YELLOW.value] * 3] * 3,
                 dtype=np.int8)
    
    def _select_colors(self) -> None:
        """
        Initiates the colors selection interface.

        Keyboard commands:
            Chage selected color -> Arrow keys
            Turn cube view -> TAB
            Exit -> Esc
            Continue -> Enter
        """
        scale_factor = 0.75
        x = 0
        color = -1

        def __draw_pallete(img):
            square_side = 35
            i, j = 30, 740
            fill = [
                (255, 255, 255),
                (0, 0, 255),
                (255, 0, 0),
                (0, 255, 0),
                (35, 100, 255),
                (0, 255, 255)
            ]
            for ind in range(6):
                img = cv2.rectangle(img, (i, j), (i+square_side, j+square_side), fill[ind], -1)
                if ind == color:
                    img = cv2.rectangle(img, (i, j), (i+square_side, j+square_side), [0]*3, 5)
                else:
                    img = cv2.rectangle(img, (i, j), (i+square_side, j+square_side), [150]*3, 5)
                i += 50
            
            return img

        self.face_top = np.array(
            [[6] * 3] * 3,
            dtype=np.int8)
        self.face_top[1, 1] = 0

        self.face_front = np.array(
            [[6] * 3] * 3,
            dtype=np.int8)
        self.face_front[1, 1] = 1

        self.face_left = np.array(
            [[6] * 3] * 3,
            dtype=np.int8)
        self.face_left[1, 1] = 2

        self.face_right = np.array(
            [[6] * 3] * 3,
            dtype=np.int8)
        self.face_right[1, 1] = 3

        self.face_back = np.array(
            [[6] * 3] * 3,
            dtype=np.int8)
        self.face_back[1, 1] = 4

        self.face_bottom = np.array(
            [[6] * 3] * 3,
            dtype=np.int8)
        self.face_bottom[1, 1] = 5

        corner = 0

        points = [
            [
                [(400, 100), (500, 130), (630, 180)],
                [(280, 130), (400, 180), (500, 225)],
                [(160, 180), (260, 230), (400, 275)]
            ],
            [
                [(100, 275), (200, 340), (330, 380)],
                [(110, 400), (220, 470), (330, 525)],
                [(130, 550), (230, 600), (340, 660)]
            ],
            [
                [(450, 400), (580, 330), (690, 280)],
                [(450, 530), (580, 480), (680, 425)],
                [(450, 660), (570, 610), (670, 550)]
            ],
            # [[47 + i*50, 757] for i in range(6)]
        ]

        while x != 27:
            if x == 13:
                cv2.destroyAllWindows()
                return 

            if x == 0:
                color = (color + 1) % 6
                cv2.imshow('image', __draw_pallete(self.plot_cube(corner)))
            
            if x == 9:
                corner = 0 if corner else 7

            def mouse_callback(event, x, y, flags, params):
                if event == 1:
                    closest_dist = None

                    for i, face in enumerate(points):
                        # if i == 3:
                        #     for col, loc in enumerate(face):
                        #         dist = (y-loc[1])**2+(x-loc[0])**2
                        #         if dist < closest_dist:
                        #             painted_square = col
                        #             closest_dist = dist
                        #     continue
                        for y_, line in enumerate(face):
                            for x_, square in enumerate(line):
                                dist = (y-square[1])**2+(x-square[0])**2
                                if closest_dist is None or dist < closest_dist:
                                    painted_square = (i, y_, x_)
                                    closest_dist = dist
                    
                    # if isinstance(painted_square, int):
                    #     color = painted_square
                    #     cv2.imshow('image', __draw_pallete(self.plot_cube(corner)))
                    #     return

                    if closest_dist > 2500:
                        return

                    if corner:
                        if painted_square[0] == 0:
                            self.make_move(Moves.D3)
                            self.face_bottom[painted_square[1], painted_square[2]] = color
                            self.make_move(Moves.D1)
                        elif painted_square[0] == 1:
                            self.make_move(Moves.L2)
                            self.face_left[painted_square[1], painted_square[2]] = color
                            self.make_move(Moves.L2)
                        elif painted_square[0] == 2:
                            self.face_back[painted_square[1], painted_square[2]] = color
                    else:
                        if painted_square[0] == 0:
                            self.face_top[painted_square[1], painted_square[2]] = color
                        elif painted_square[0] == 1:
                            self.face_front[painted_square[1], painted_square[2]] = color
                        elif painted_square[0] == 2:
                            self.face_right[painted_square[1], painted_square[2]] = color

                    cv2.imshow('image', __draw_pallete(self.plot_cube(corner)))
                    

            img = __draw_pallete(self.plot_cube(corner))

            width = GetSystemMetrics(0)
            height = GetSystemMetrics(1)
            scale_width = width * scale_factor / img.shape[1]
            scale_height = height * scale_factor / img.shape[0]
            scale = min(scale_width, scale_height)
            window_width = int(img.shape[1] * scale)
            window_height = int(img.shape[0] * scale)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', window_width, window_height)

            cv2.setMouseCallback('image', mouse_callback)

            cv2.imshow('image', img)
            x = cv2.waitKey(0)
        
    def __turn_face(self, face, turn: str = "cw"):
        if turn == "cw":
            return np.array(
                [
                    [face[2, 0], face[1, 0], face[0, 0]],
                    [face[2, 1], face[1, 1], face[0, 1]],
                    [face[2, 2], face[1, 2], face[0, 2]]
                ],
                dtype=np.int8
            )
        if turn == "a-cw":
            return np.array(
                [
                    [face[0, 2], face[1, 2], face[2, 2]],
                    [face[0, 1], face[1, 1], face[2, 1]],
                    [face[0, 0], face[1, 0], face[2, 0]]
                ],
                dtype=np.int8
            )
        if turn == "inv":
            return np.array(
                [
                    [face[2, 2], face[2, 1], face[2, 0]],
                    [face[1, 2], face[1, 1], face[1, 0]],
                    [face[0, 2], face[0, 1], face[0, 0]]
                ],
                dtype=np.int8
            )

    def _move_r(self):
        self.face_right = self.__turn_face(self.face_right)

        temp = deepcopy(self.face_top[:, 2])
        self.face_top[:, 2] = self.face_front[:, 2]
        self.face_front[:, 2] = self.face_bottom[:, 2]
        self.face_bottom[:, 2] = self.face_back[:, 2]
        self.face_back[:, 2] = temp
    
    def _move_l(self):
        self.face_left = self.__turn_face(self.face_left)

        temp = deepcopy(self.face_top[:, 0])
        self.face_top[:, 0] = self.face_back[:, 0]
        self.face_back[:, 0] = self.face_bottom[:, 0]
        self.face_bottom[:, 0] = self.face_front[:, 0]
        self.face_front[:, 0] = temp
    
    def _move_t(self):
        self.face_top = self.__turn_face(self.face_top)

        temp = deepcopy(self.face_left[0, :])

        self.face_left[0, :] = self.face_front[0, :]
        self.face_front[0, :] = self.face_right[0, :]
        self.face_right[0, :] = self.__turn_face(self.face_back, "inv")[0, :]
        self.face_back[2, :] = np.flip(temp)
    
    def _move_bt(self):
        self.face_bottom = self.__turn_face(self.face_bottom)

        temp = np.flip(deepcopy(self.face_right[2, :]))

        self.face_right[2, :] = self.face_front[2, :]
        self.face_front[2, :] = self.face_left[2, :]
        self.face_left[2, :] = self.__turn_face(self.face_back, "inv")[2, :]
        self.face_back[0, :] = temp
    
    def _move_f(self):
        self.face_front = self.__turn_face(self.face_front)

        temp = deepcopy(self.face_left[:, 2])

        self.face_left[:, 2] = self.face_bottom[0, :]
        self.face_bottom[0, :] = np.flip(self.face_right[:, 0])
        self.face_right[:, 0] = self.face_top[2, :]
        self.face_top[2, :] = np.flip(temp)
    
    def _move_bk(self):
        self.face_back = self.__turn_face(self.face_back)

        temp = deepcopy(self.face_right[:, 2])

        self.face_right[:, 2] = np.flip(self.face_bottom[2, :])
        self.face_bottom[2, :] = self.face_left[:, 0]
        self.face_left[:, 0] = np.flip(self.face_top[0, :])
        self.face_top[0, :] = temp

    def make_move(self, move: Moves):
        if Moves.is_right(move):
            to_move = self._move_r
        elif Moves.is_left(move):
            to_move = self._move_l
        elif Moves.is_top(move):
            to_move = self._move_t
        elif Moves.is_down(move):
            to_move = self._move_bt
        elif Moves.is_front(move):
            to_move = self._move_f
        elif Moves.is_back(move):
            to_move = self._move_bk
        
        for _ in range(Moves.repetitions(move)):
            to_move()

    def sequence(self, moves: list):
        for move in moves:
            self.make_move(move)

    def plot_cube(self, vertex: int = 0):
        points = [
            [
                [(400, 100), (500, 130), (630, 180)],
                [(280, 130), (400, 180), (500, 225)],
                [(160, 180), (260, 230), (400, 275)]
            ],
            [
                [(100, 275), (200, 340), (330, 380)],
                [(110, 400), (220, 470), (330, 525)],
                [(130, 550), (230, 600), (340, 660)]
            ],
            [
                [(450, 400), (580, 330), (690, 280)],
                [(450, 530), (580, 480), (680, 425)],
                [(450, 660), (570, 610), (670, 550)]
            ]
        ]

        img = cv2.imread("utils\cube.png")

        if vertex == 0:
            cube_face = [
                self.face_top,
                self.face_front,
                self.face_right
            ]
        
        if vertex == 1:
            cube_face = [
                self.__turn_face(self.face_top),
                self.face_right,
                self.__turn_face(self.face_back, "inv")
            ]
        
        if vertex == 2:
            cube_face = [
                self.__turn_face(self.face_top, "inv"),
                self.__turn_face(self.face_back, "inv"),
                self.face_left,
            ]
        
        if vertex == 3:
            cube_face = [
                self.__turn_face(self.face_top, "a-cw"),
                self.face_left,
                self.face_front
            ]
        
        if vertex == 4:
            cube_face = [
                self.face_bottom,
                self.face_back,
                self.__turn_face(self.face_right, "inv")
            ]
        
        if vertex == 5:
            cube_face = [
                self.__turn_face(self.face_bottom),
                self.__turn_face(self.face_right, "inv"),
                self.__turn_face(self.face_front, "inv")
            ]
        
        if vertex == 6:
            cube_face = [
                self.__turn_face(self.face_bottom, "inv"),
                self.__turn_face(self.face_front, "inv"),
                self.__turn_face(self.face_left, "inv")
            ]
        
        if vertex == 7:
            cube_face = [
                self.__turn_face(self.face_bottom, "a-cw"),
                self.__turn_face(self.face_left, "inv"),
                self.face_back
            ]
        
        for i, face in enumerate(points):
            for y, line in enumerate(face):
                for x, square in enumerate(line):
                    color = (255, 255, 255) if cube_face[i][y][x] == 0 else \
                    (0, 0, 255) if cube_face[i][y][x] == 1 else \
                    (0, 255, 0) if cube_face[i][y][x] == 2 else \
                    (255, 0, 0) if cube_face[i][y][x] == 3 else \
                    (35, 100, 255) if cube_face[i][y][x] == 4 else \
                    (0, 255, 255) if cube_face[i][y][x] == 5 else \
                    (150, 150, 150)
                    cv2.floodFill(img, None, square, color)

        return img

    def _verify(self, sequence):
        cube = deepcopy(self)
        cube.sequence(sequence)
        return np.all(cube.face_back == Colors.ORANGE.value)\
            and np.all(cube.face_bottom == Colors.YELLOW.value) \
            and np.all(cube.face_front == Colors.RED.value) \
            and np.all(cube.face_left == Colors.GREEN.value)

    def solve(self, depth, moves = []):
        """
            Solve cube and return the moves sequence for a defined maximum movements count.
        """
        for i in range(18):
            if len(moves) > 0 and Moves.is_same_side(i, moves[-1]):
                continue

            elif len(moves) > 1 and Moves.is_same_side(i, moves[-2]) and Moves.is_opposite_side(i, moves[-1]):
                continue

            moves.append(i)
            if self._verify(moves):
                return moves
            
            if depth > 1:
                res = self.solve(depth-1, moves)
    
                if res:
                    return res

            moves = moves[:-depth]

        return None
    
    def __str__(self) -> str:
        cv2.imshow("", self.plot_cube(0))
        cv2.waitKey(0)
        return ""


if __name__ == "__main__":
    cube = Cube(True)

    print(cube)
    
    print("\nSolving...")
    solution = cube.solve(4)

    text = "\nSolution:"
    for movement in solution:
        text += " " + Moves.notation(movement)

    print(text)
    cube.sequence(solution)

    print(cube)
