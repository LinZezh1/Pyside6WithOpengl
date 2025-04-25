import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
import numpy as np
from OpenGL.GL import *


class OpenGLWidget(QOpenGLWidget):
    def initializeGL(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        self.setupShaders()
        self.setupGeometry()

    def setupShaders(self):
        vertex_shader = """
        #version 330 core
        layout (location = 0) in float t;

        uniform vec2 p0;
        uniform vec2 p1;
        uniform vec2 p2;
        uniform vec2 p3;

        void main()
        {
            float u = 1.0 - t;
            vec2 pos = u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3;
            gl_Position = vec4(pos, 0.0, 1.0);
        }
        """
        fragment_shader = """
        #version 330 core
        out vec4 fragColor;
        void main()
        {
            fragColor = vec4(1.0, 1.0, 0.0, 1.0);
        }
        """
        self.shader = self.compileShaders(vertex_shader, fragment_shader)

    def compileShaders(self, vertex_source, fragment_source):
        vertexShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexShader, vertex_source)
        glCompileShader(vertexShader)
        if not glGetShaderiv(vertexShader, GL_COMPILE_STATUS):
            print("顶点着色器编译错误:", glGetShaderInfoLog(vertexShader))

        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragmentShader, fragment_source)
        glCompileShader(fragmentShader)
        if not glGetShaderiv(fragmentShader, GL_COMPILE_STATUS):
            print("片段着色器编译错误:", glGetShaderInfoLog(fragmentShader))

        program = glCreateProgram()
        glAttachShader(program, vertexShader)
        glAttachShader(program, fragmentShader)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            print("着色器程序链接错误:", glGetProgramInfoLog(program))

        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)
        return program

    def setupGeometry(self):
        # 生成 100 个 t 值 [0.0, 0.01, 0.02, ..., 1.0]
        t_values = np.linspace(0.0, 1.0, 100, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, t_values.nbytes, t_values, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader)

        # 固定控制点，传给GPU
        glUniform2f(glGetUniformLocation(self.shader, "p0"), -0.8, -0.8)
        glUniform2f(glGetUniformLocation(self.shader, "p1"), -0.4,  0.8)
        glUniform2f(glGetUniformLocation(self.shader, "p2"),  0.4, -0.8)
        glUniform2f(glGetUniformLocation(self.shader, "p3"),  0.8,  0.8)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINE_STRIP, 0, 100)
        glBindVertexArray(0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL 三阶贝塞尔曲线")
        self.setGeometry(100, 100, 800, 600)
        self.gl_widget = OpenGLWidget()
        self.setCentralWidget(self.gl_widget)


if __name__ == "__main__":
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3)
    gl_format.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(gl_format)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
