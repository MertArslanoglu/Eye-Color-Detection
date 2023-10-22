import mediapipe
import cv2 as cv
import numpy as np
import argparse
import time
import base64
from PIL import Image
import io

class tools:
    def __init__(self):
        self.class_name = ("Blue", "Blue Gray", "Brown", "Brown", "Green Gray", "Brown Black", "Green", "Green Gray", "Black", "Other")
        self.EyeColor = {
            self.class_name[0]: ((83, 53, 127), (120, 255, 216)),
            self.class_name[1]: ((83, 5, 63), (150, 53, 191)),
            self.class_name[2]: ((10, 100, 20), (20, 255, 200)),
            self.class_name[3]: ((10, 8, 77), (15, 153, 153)),
            self.class_name[4]: ((15, 8, 77), (32, 153, 153)),
            self.class_name[5]: ((0, 25, 13), (20, 102, 64)),
            self.class_name[6]: ((30, 53, 128), (82, 255, 217)),
            self.class_name[7]: ((30, 5, 64), (82, 53, 166)),
            self.class_name[8]: ((0, 0, 0), (180, 255, 25))
        }


    def to_image_string(self,image_filepath):
        with open(image_filepath, "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
        return(my_string)

    def from_base64(self, base64_data):
        decoded_data = base64.b64decode((base64_data))
        image = Image.open(io.BytesIO(decoded_data))
        return cv.cvtColor(np.array(image), cv.COLOR_BGR2RGB)


    def white_balance(self, img):
        result = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv.cvtColor(result, cv.COLOR_LAB2RGB)
        return result

    def check_color(self, hsv, color):
        if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= (color[0][1])) and \
                (hsv[1] <= (color[1][1])) and (hsv[2] >= (color[0][2])) and (hsv[2] <= (color[1][2])):
            return True
        else:
            return False

    def find_class(self, hsv):
        color_id = 9
        for i in range(len(self.class_name) - 1):
            if self.check_color(hsv, self.EyeColor[self.class_name[i]]) == True:
                color_id = i
                if i == 3:
                    color_id = 2
                if i == 4:
                    color_id = 7
        return color_id

    def eye_color(self, b64):
        frame = self.from_base64(b64)
        init = time.time()


        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]

        ## INPUT:

        frame = cv.resize(frame, (300, 400), interpolation=cv.INTER_AREA)
        # cv.imshow("a", frame)
        # cv.waitKey(0)
        frame = self.white_balance(frame)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        ## Iris detection and color detection:
        mp_face_mesh = mediapipe.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
        ) as face_mesh:

            results = face_mesh.process(frame)
            if results.multi_face_landmarks is None:
                return "Brown Black"
            img_h, img_w = frame.shape[:2]

            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, center_left, int(int(l_radius / 3)), (255, 0, 0), 2, cv.LINE_AA)
            cv.circle(frame, center_right, int(int(r_radius / 3)), (255, 0, 0), 2, cv.LINE_AA)

            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv.circle(mask, center_left, int(l_radius), (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(mask, center_right, int(r_radius), (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(mask, center_left, int(int(l_radius / 3)), (0, 0, 0), -1, cv.LINE_AA)
            cv.circle(mask, center_right, int(int(r_radius / 3)), (0, 0, 0), -1, cv.LINE_AA)
            cv.circle(mask, center_left, int(int(l_radius)), (0, 0, 0), 2, cv.LINE_AA)
            cv.circle(mask, center_right, int(int(r_radius)), (0, 0, 0), 2, cv.LINE_AA)
            #cv.imshow("b", mask)
            #cv.waitKey(0)

            eye_class = np.zeros(len(self.class_name), float)
            for y in range(img_h):
                for x in range(img_w):
                    if mask[y, x] != 0:
                        # if (np.linalg.norm(np.array([y, x]) - np.array(center_left)) < l_radius and np.linalg.norm(
                        #             np.array([y, x]) - np.array(center_left)) > l_radius * 0.3) or (np.linalg.norm(np.array([y, x]) - np.array(center_right)) < r_radius and np.linalg.norm(
                        #             np.array([y, x]) - np.array(center_right)) > r_radius * 0.3):
                        a = self.find_class(hsv[y, x])
                        eye_class[self.find_class(hsv[y, x])] += 1
            main_color_index = np.argmax(eye_class[:len(eye_class) - 1])
            total_vote = eye_class.sum()
            if eye_class[np.argmax(eye_class[:len(eye_class) - 1])] < 4:
                return "Black Brown"
            print("Processing duration:" + str(time.time() - init))
            #print("\n **Eyes Color Percentage **")
            #for i in range(len(self.class_name)):
                #print(self.class_name[i], ": ", round(eye_class[i] / total_vote * 100, 2), "%")
            #print("\n\nDominant Eye Color: ", class_name[main_color_index])
            #cv.imshow("a", frame)
            #cv.waitKey(0)
            return(self.class_name[main_color_index])



