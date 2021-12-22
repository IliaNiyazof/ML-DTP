import sys
from PyQt5.Qt import *
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# CUDA_VISIBLE_DEVICES = 0

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(500, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.label = QLabel(self.centralwidget)  # - QLabel, + Label   !!!
        self.label.setObjectName(u"label")
        self.label.setGeometry(0, 0, 500, 500)
        self.label.setScaledContents(True)
        self.label.setStyleSheet('border-style: solid; border-width: 1px; border-color: black;')
        self.label.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setMargin(-3)
        self.label.setOpenExternalLinks(False)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 502, 22))
        MainWindow.setMenuBar(self.menubar)
        self.lbl = QLabel('Ожидание', self.centralwidget)
        self.lbl.move(0, 500)
        self.lbl.setFixedWidth(500)
        self.lbl.setAlignment(Qt.AlignCenter)
        self.btn = QPushButton("Проверка", self.centralwidget)
        self.connect_box = QVBoxLayout(self.centralwidget)
        self.connect_box.setAlignment(Qt.AlignCenter)
        self.connect_box.addWidget(self.btn, alignment=Qt.AlignCenter)  # < ----
        self.connect_box.setContentsMargins(0, 525, 0, 0)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"ДТП-проверка", None))
        self.label.setText("")


class Label(QLabel):
    clicked = pyqtSignal()  # +++

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        self.clicked.emit()  # +++


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui2 = Ui_MainWindow()
        self.ui2.setupUi(self)

        mainMenu = self.menuBar()
        file = mainMenu.addMenu("Файл")
        photo = QAction("Вставить фото", self)
        file.addAction(photo)
        photo.triggered.connect(self.getImage)

        self.ui2.btn.clicked.connect(self.testclicl)

    def getImage(self):
        self.filename, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "All Files(*.*);;PNG(*.png);;JPEG(*.jpg *.jpeg)"
        )
        fil = self.filename

        if not self.filename:
            return
        self.ui2.label.setPixmap(QPixmap(u"{}".format(fil)))

    def testclicl(self):
        SIZE = 224

        def resize_image(img):
            img = tf.image.resize(img, (SIZE, SIZE))
            img = tf.cast(img, tf.float32)
            img = img / 255.0
            return img

        base_layers = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False)
        base_layers.trainable = False

        model = tf.keras.Sequential([
            base_layers,
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(5, activation="sigmoid"),
        ])

        model.load_weights('model_test/test.h5')

        test = ['Вмятина', 'Повреждение фары', 'Царапина', 'Тотал', 'Целая']
        for i in range(1):
            img = load_img(self.filename)
            img_array = img_to_array(img)
            img_resized = resize_image(img_array)
            img_expended = np.expand_dims(img_resized, axis=0)
            prediction = model.predict(img_expended)
            predicted_label = np.argmax(prediction)
            pred = np.max(prediction)
            pr = test[predicted_label]

            test_2 = int(pred * 100)

            if test_2 < 95:
                result = f'{pr}-{test_2}'
                self.ui2.lbl.setText(result)
                self.ui2.lbl.setStyleSheet('color: red')
            if test_2 >= 95:
                result = f'{pr}-{test_2}'
                self.ui2.lbl.setText(result)
                self.ui2.lbl.setStyleSheet('color: green')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
