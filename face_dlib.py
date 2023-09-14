import dlib
import cv2
import os

def crop_face(input_folder, output_folder):
    face_detector = dlib.get_frontal_face_detector()
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        faces = face_detector(img, 1)
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_img = img[y1:y2, x1:x2]
            cropped_img_path = os.path.join(output_folder, f"{filename[:-4]}.jpg")
            cv2.imwrite(cropped_img_path, face_img)

        print(f"Processed {filename}, Detected {len(faces)} face(s)")

if __name__ == "__main__":
    train_input_folder = "./data1/train"
    test_input_folder = "./data1/val"
    train_output_folder = "./data/train"
    test_output_folder = "./data/val"

    # Create output folders if they don't exist
    #if not os.path.exists(train_output_folder):
        #$os.makedirs(train_output_folder)
    #if not os.path.exists(test_output_folder):
        #os.makedirs(test_output_folder)

    # Crop faces in train folder
    crop_face(train_input_folder, train_output_folder)
    # Crop faces in test folder
    crop_face(test_input_folder, test_output_folder)
