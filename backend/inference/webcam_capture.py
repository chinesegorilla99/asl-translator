import cv2import cv2





class WebcamCapture:class WebcamCapture:

    def __init__(self, camera_index=0, width=1280, height=720, fps=60):    def __init__(self, camera_index=0, width=1280, height=720, fps=60):

        self.camera_index = camera_index        self.camera_index = camera_index

        self.width = width        self.width = width

        self.height = height        self.height = height

        self.fps = fps        self.fps = fps

        self.cap = None        self.cap = None



    def start(self):    def start(self):

        self.cap = cv2.VideoCapture(self.camera_index)        self.cap = cv2.VideoCapture(self.camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.cap.set(cv2.CAP_PROP_FPS, self.fps)        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

                

        if not self.cap.isOpened():        if not self.cap.isOpened():

            raise RuntimeError(f"Failed to open camera {self.camera_index}")            raise RuntimeError(f"Failed to open camera {self.camera_index}")

                

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

                

        print(f"Webcam started: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")        print(f"Webcam started: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

        return self        return self



    def read(self):    def read(self):

        if self.cap is None:        if self.cap is None:

            raise RuntimeError("Webcam not started. Call start() first.")            raise RuntimeError("Webcam not started. Call start() first.")

                

        ret, frame = self.cap.read()        ret, frame = self.cap.read()

        if not ret:        if not ret:

            return None            return None

        return frame        return frame



    def stop(self):    def stop(self):

        if self.cap is not None:        if self.cap is not None:

            self.cap.release()            self.cap.release()

            self.cap = None            self.cap = None

            print("Webcam stopped")            print("Webcam stopped")



    def __enter__(self):    def __enter__(self):

        return self.start()        return self.start()



    def __exit__(self, exc_type, exc_val, exc_tb):    def __exit__(self, exc_type, exc_val, exc_tb):

        self.stop()        self.stop()

        return False        return False





def main():def main():

    print("Starting webcam capture...")    print("Starting webcam capture...")

    print("Press 'q' to quit")    print("Press 'q' to quit")

        

    with WebcamCapture() as webcam:    with WebcamCapture() as webcam:

        frame_count = 0        frame_count = 0

                

        while True:        while True:

            frame = webcam.read()            frame = webcam.read()

            if frame is None:            if frame is None:

                print("Failed to read frame")                print("Failed to read frame")

                break                break

                        

            frame_count += 1            frame_count += 1

                        

            cv2.putText(            cv2.putText(

                frame,                frame,

                f"Frame: {frame_count}",                f"Frame: {frame_count}",

                (10, 30),                (10, 30),

                cv2.FONT_HERSHEY_SIMPLEX,                cv2.FONT_HERSHEY_SIMPLEX,

                1,                1,

                (0, 255, 0),                (0, 255, 0),

                2                2

            )            )

                        

            cv2.imshow("ASL Translator - Webcam", frame)            cv2.imshow("ASL Translator - Webcam", frame)

                        

            if cv2.waitKey(1) & 0xFF == ord('q'):            if cv2.waitKey(1) & 0xFF == ord('q'):

                break                break

                

        print(f"Total frames captured: {frame_count}")        print(f"Total frames captured: {frame_count}")

        

    cv2.destroyAllWindows()    cv2.destroyAllWindows()





if __name__ == "__main__":if __name__ == "__main__":

    main()    main()

