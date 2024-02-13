from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np

class Pill:
    def task_1_Pillow(self):
        # Загрузка и открытие изображения
        image = Image.open('matrix.jpg')

        # Обрезка изображения
        cropped_image = image.crop((302, 14, 648, 524))

        # Нанесение текстового обозначения
        draw = ImageDraw.Draw(cropped_image)
        draw.text((97, 438), "Kianu Reeves", fill=(255, 255, 255))

        # Нанесение графического обозначения
        draw.rectangle((5, 5, 320, 500), outline=(0, 255, 0), width=5)

        # Размытие части изображения
        blurred_image = cropped_image.filter(ImageFilter.BLUR)

        # Сохранение изображения
        cropped_image.save('matrix_cropped.jpg')
        blurred_image.save('matrix_blurred.jpg')

        cropped_image.show()
        blurred_image.show()

    def task_2_Pillow(self):
        # Загрузка изображения
        image = Image.open('matrix.jpg')

        # Вырезание фрагмента
        cropped_image = image.crop((302, 14, 648, 524))

        # Увеличение размера изображения
        resized_image = cropped_image.resize((1000, 1000))

        # Сохранение увеличенного изображения
        resized_image.save('matrix_resized.jpg')

        # Поворот изображения
        rotated_image = resized_image.rotate(90)

        # Сохранение повернутого изображения
        rotated_image.save('matrix_rotated.jpg')

        resized_image.show()
        rotated_image.show()

class Cv:
    def task_1_OpenCV(self):
        image = cv2.imread('matrix.jpg')

        cropped_image = image[100:400, 100:400]

        # нанесение текста
        cv2.putText(cropped_image, 'Text', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Нанесение графического обозначения
        cv2.rectangle(cropped_image, (200, 200), (300, 300), (255, 255, 255), 7)

        # размытие изображения
        blurred_image = cv2.blur(cropped_image, (5, 5))

        # сохраняем изображение
        cv2.imwrite('matrix_cropped_CV.jpg', cropped_image)
        cv2.imwrite('matrix_blurred_CV.jpg', blurred_image)

        cv2.imshow('Обрезка', cropped_image)
        cv2.imshow('Размытие', blurred_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def task_2_OpenCV(self):
        image = cv2.imread('matrix.jpg')

        cropped_image = image[100:400, 100:400]         #обрезка

        resized_image = cv2.resize(cropped_image, (500, 500))    #увеличение изображения

        cv2.imwrite('images/resized_image_CV.jpg', resized_image)   #сохранение увеличенного изображения

        rows, cols, _ = resized_image.shape
        rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)    #поворот изображения
        rotated_image = cv2.warpAffine(resized_image, rotate, (cols, rows))

        cv2.imwrite('images/rotated_image_CV.jpg', rotated_image)   #cохранение повернутого изображения

        cv2.imshow('Увеличенное', resized_image)
        cv2.imshow('Повернутое', rotated_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def task_3_OpenCV(self):
        img_green = cv2.imread('matrix.png')

        hsv_image = cv2.cvtColor(img_green, cv2.COLOR_BGR2HSV)
        dark_green = np.array([30, 60, 40], np.uint8)
        light_green = np.array([80, 255, 255], np.uint8)
        mask = cv2.inRange(hsv_image, dark_green, light_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            print(f"Координаты объекта: x={x}, y={y}, ширина={w}, высота={h}")
            cv2.rectangle(img_green, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Контур", img_green)
        cv2.imshow("HSV", hsv_image)
        cv2.imshow("Маска", mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def task_4_OpenCV(self):

        image = cv2.imread('matrix.jpg')

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        white_lower = np.array([0, 0, 200], np.uint8)
        white_upper = np.array([180, 30, 255], np.uint8)

        white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
        color_mask = cv2.bitwise_not(white_mask)
        #объекты выделились в белый цвет, а фон в черный

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image, contours, -1, (255, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow("Контуры", image)
        cv2.imshow("Цветная маска", color_mask)
        cv2.imshow("Белая маска", white_mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

#Main programm
def main():
    pill = Pill()
    cv = Cv()

    while True:
        print()
        print("Меню:")
        print("1. Задание 1. Pillow.")
        print("2. Задание 1. OpenCV.")
        print("3. Задание 2. Pillow.")
        print("4. Задание 2. OpenCV.")
        print("5. Задание 3. OpenCV.")
        print("6. Задание 4. OpenCV.")
        print("7. Выход.")
        choice = input("Введите номер задания: ")

        if choice == "1":
            pill.task_1_Pillow()
        elif choice == "2":
            cv.task_1_OpenCV()
        elif choice == "3":
            pill.task_2_Pillow()
        elif choice == "4":
            cv.task_2_OpenCV()
        elif choice == "5":
            cv.task_3_OpenCV()
        elif choice == "6":
            cv.task_4_OpenCV()
        elif choice == "7":
            break
        else:
            print("Неверный ввод.")

if __name__ == "__main__":
    main()
