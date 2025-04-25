import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
import numpy as np
from OpenGL.GL import *

# 测试 pyside 中的控件使用以及 Opengl 的基本功能
class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.current_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 默认纯红色

    def initializeGL(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glEnable(GL_DEPTH_TEST)

        self.setupShaders()
        self.setupGeometry()

    def setupShaders(self):
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;

        uniform mat4 model;

        void main()
        {
            gl_Position = model * vec4(position, 1.0);
        }
        """

        fragment_shader = """
        #version 330 core
        uniform vec3 color;
        out vec4 fragColor;

        void main()
        {
            fragColor = vec4(color, 1.0);
        }
        """

        self.shader = self.compileShaders(vertex_shader, fragment_shader)

    def compileShaders(self, vertex_source, fragment_source):
        vertexShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexShader, vertex_source)
        glCompileShader(vertexShader)
        if not glGetShaderiv(vertexShader, GL_COMPILE_STATUS):
            print("顶点着色器编译错误:", glGetShaderInfoLog(vertexShader).decode())

        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragmentShader, fragment_source)
        glCompileShader(fragmentShader)
        if not glGetShaderiv(fragmentShader, GL_COMPILE_STATUS):
            print("片段着色器编译错误:", glGetShaderInfoLog(fragmentShader).decode())

        program = glCreateProgram()
        glAttachShader(program, vertexShader)
        glAttachShader(program, fragmentShader)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            print("着色器程序链接错误:", glGetProgramInfoLog(program).decode())

        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)

        return program

    def setupGeometry(self):
        vertices = np.array([
            -0.5, -0.5, 0.0,  # 左下
             0.5, -0.5, 0.0,  # 右下
             0.0,  0.5, 0.0   # 顶部
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.shader)

        identity_matrix = np.identity(4, dtype=np.float32)
        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, identity_matrix)

        color_loc = glGetUniformLocation(self.shader, "color")
        glUniform3fv(color_loc, 1, self.current_color)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

    def setColor(self, color):
        self.current_color = np.array(color, dtype=np.float32)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("三角形颜色切换示例")
        self.setGeometry(100, 100, 800, 600)

        container = QWidget()
        layout = QVBoxLayout(container)

        self.gl_widget = OpenGLWidget()
        layout.addWidget(self.gl_widget)

        # 红色按钮
        btn_red = QPushButton("显示红色三角形")
        btn_red.clicked.connect(lambda: self.gl_widget.setColor([1.0, 0.0, 0.0]))
        layout.addWidget(btn_red)

        # 蓝色按钮
        btn_blue = QPushButton("显示蓝色三角形")
        btn_blue.clicked.connect(lambda: self.gl_widget.setColor([0.0, 0.0, 1.0]))
        layout.addWidget(btn_blue)

        self.setCentralWidget(container)


if __name__ == "__main__":
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3)
    gl_format.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(gl_format)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
