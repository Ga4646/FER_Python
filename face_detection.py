from fer.fer import FER
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


# Paramètres
INPUT_DIR = "image_/groupeA"

#INPUT_DIR = "image_/groupeB"

emotion_detector = FER(mtcnn=True)

def single_face(img,result):
        bounding_box = result[0]["box"]
        emotions = result[0]["emotions"]
        emotion_final = max(emotions, key=emotions.get)
        score_final = max(emotions.values())
        print(f"Emotion predicted {emotion_final} and emotion score is {score_final}")

        cv2.rectangle(img,(
            bounding_box[0], bounding_box[1]),(
                bounding_box[0] + bounding_box[2],bounding_box[1] + bounding_box[3]),
            (0, 155, 255), 4,)
        
        emotion_name, score = emotion_detector.top_emotion(img)
        for index, (emotion_name, score) in enumerate(emotions.items()):
            color = (0,0,0) if score < 0.01 else (211,211,211)
            emotion_score = "{}: {}".format(emotion_name,"{:.2f}".format(score))

            cv2.putText(img,emotion_score,
                        (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,color,3,cv2.LINE_AA,)
                
        return img


def multiple_faces(img,result):
      
    for i , faces in enumerate(result):
            bounding_box = faces['box']
            emotions = faces['emotions']
            emotion_final = max(emotions, key=emotions.get)
            score_final = max(emotions.values())
            print(f"Emotion predicted {emotion_final} and emotion score is {score_final}")

            cv2.rectangle(img,(
                bounding_box[0], bounding_box[1]),(
                    bounding_box[0] + bounding_box[2],bounding_box[1] + bounding_box[3]),
                (0, 155, 255), 1,)
            
            
            emotion_name, score = emotion_detector.top_emotion(img)

            for index, (emotion_name, score) in enumerate(emotions.items()):
                color = (0,0,0) if score < 0.01 else (211,211,211)
                emotion_score = "{}: {}".format(emotion_name,"{:.2f}".format(score))

                cv2.putText(img,emotion_score,
                            (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,1,color,3,cv2.LINE_AA,)
                
    return img


#liste pour récolter les scores des émotion.
dataset = []
for file_name in Path(INPUT_DIR).iterdir():
        img = cv2.imread(file_name)
        result_image = mpimg.imread(file_name)
        imgplot = plt.imshow(result_image)
        plt.show()
        plt.pause(0.05) 
        
        result = emotion_detector.detect_emotions(result_image)

        print(file_name)

        #verification du nombre de visage sur la photo. 
        #si elle est supérieur à 1, fonction pour identifier tous les visages. 
        if len(result) > 1:
            img = multiple_faces(img,result)
            cv2.imwrite("emotion.jpg", img)
            result_image = mpimg.imread('emotion.jpg') 
            imgplot = plt.imshow(result_image)
            # Display Output Image
            plt.show()

        # si le nombre de visage est inférieur à 1 on utilise la fonction single_face
        else:
            img = single_face(img,result)