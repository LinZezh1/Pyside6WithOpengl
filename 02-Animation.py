import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
import numpy as np
from OpenGL.GL import *

# pyside 控件结合 Opengl 实现简单的 Morph 效果
class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateAnimation)
        self.timer.start(16)  # 60fps

        self.current_vertices = None
        self.target_vertices = None
        self.animation_progress = 0.0
        self.animating = False
        self.current_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 默认红色

    def initializeGL(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glEnable(GL_DEPTH_TEST)

        self.setupShaders()
        self.setupGeometry()

        self.setToTriangle()  # 默认显示红色三角形

    def setupShaders(self):
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec2 position;
        uniform vec3 color;
        out vec3 fragColor;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            fragColor = color;
        }
        """
        fragment_shader = """
        #version 330 core
        in vec3 fragColor;
        out vec4 outColor;
        void main() {
            outColor = vec4(fragColor, 1.0);
        }
        """
        self.shader = self.compileShaders(vertex_shader, fragment_shader)

    def compileShaders(self, vertex_src, fragment_src):
        vertexShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexShader, vertex_src)
        glCompileShader(vertexShader)
        if not glGetShaderiv(vertexShader, GL_COMPILE_STATUS):
            print(glGetShaderInfoLog(vertexShader))

        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragmentShader, fragment_src)
        glCompileShader(fragmentShader)
        if not glGetShaderiv(fragmentShader, GL_COMPILE_STATUS):
            print(glGetShaderInfoLog(fragmentShader))

        program = glCreateProgram()
        glAttachShader(program, vertexShader)
        glAttachShader(program, fragmentShader)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            print(glGetProgramInfoLog(program))

        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)

        return program

    def setupGeometry(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def setToTriangle(self):
        self.current_vertices = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.0, 0.5]
        ], dtype=np.float32)
        self.current_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 红色
        self.animating = False
        self.updateBuffer()

    def animateTriangleToSquare(self):
        self.target_vertices = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5]
        ], dtype=np.float32)
        self.target_color = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # 蓝色
        self.animation_progress = 0.0
        self.animating = True

    def animateTriangleToCircle(self):
        circle_points = []
        segments = 60
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = 0.5 * np.cos(angle)
            y = 0.5 * np.sin(angle)
            circle_points.append([x, y])
        self.target_vertices = np.array(circle_points, dtype=np.float32)
        self.target_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 绿色
        self.animation_progress = 0.0
        self.animating = True

    def updateAnimation(self):
        if self.animating:
            self.animation_progress += 0.02  # 动画速度
            if self.animation_progress >= 1.0:
                self.animation_progress = 1.0
                self.animating = False

            t = self.animation_progress
            # 插值顶点数量
            min_len = min(len(self.current_vertices), len(self.target_vertices))
            current_padded = np.vstack([
                self.current_vertices,
                np.tile(self.current_vertices[-1], (len(self.target_vertices) - len(self.current_vertices), 1))
            ]) if len(self.current_vertices) < len(self.target_vertices) else self.current_vertices[:len(self.target_vertices)]
            interpolated_vertices = (1 - t) * current_padded + t * self.target_vertices

            self.current_vertices = interpolated_vertices
            self.current_color = (1 - t) * self.current_color + t * self.target_color
            self.updateBuffer()

        self.update()

    def updateBuffer(self):
        if self.current_vertices is None:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.current_vertices.nbytes, self.current_vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.current_vertices is None:
            return  # 没顶点就不画

        glUseProgram(self.shader)
        color_loc = glGetUniformLocation(self.shader, "color")
        glUniform3fv(color_loc, 1, self.current_color)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, len(self.current_vertices))
        glBindVertexArray(0)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("PySide6 OpenGL 图形动画")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.gl_widget = OpenGLWidget()
        layout.addWidget(self.gl_widget)

        button_triangle = QPushButton("红色三角形")
        button_triangle.clicked.connect(self.gl_widget.setToTriangle)
        layout.addWidget(button_triangle)

        button_square = QPushButton("三角形 → 蓝色正方形")
        button_square.clicked.connect(self.gl_widget.animateTriangleToSquare)
        layout.addWidget(button_square)

        button_circle = QPushButton("三角形 → 绿色圆形")
        button_circle.clicked.connect(self.gl_widget.animateTriangleToCircle)
        layout.addWidget(button_circle)

        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3)
    gl_format.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(gl_format)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
