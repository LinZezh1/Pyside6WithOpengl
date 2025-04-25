import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat

import numpy as np
from OpenGL.GL import *


class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        # 设置定时器以更新动画
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # 约60fps

        self.rotation = 0.0

    def initializeGL(self):
        """初始化OpenGL资源和状态"""
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glEnable(GL_DEPTH_TEST)

        # 创建并编译着色器
        self.setupShaders()
        # 创建并绑定顶点数据
        self.setupGeometry()

    def setupShaders(self):
        # 顶点着色器
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 color;

        uniform mat4 model;

        out vec3 vertexColor;

        void main()
        {
            gl_Position = model * vec4(position, 1.0);
            vertexColor = color;
        }
        """

        # 片段着色器
        fragment_shader = """
        #version 330 core
        in vec3 vertexColor;
        out vec4 fragColor;

        void main()
        {
            fragColor = vec4(vertexColor, 1.0);
        }
        """

        # 编译着色器程序
        self.shader = self.compileShaders(vertex_shader, fragment_shader)

    def compileShaders(self, vertex_source, fragment_source):
        # 创建着色器
        vertexShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexShader, vertex_source)
        glCompileShader(vertexShader)

        # 检查编译错误
        if not glGetShaderiv(vertexShader, GL_COMPILE_STATUS):
            print("顶点着色器编译错误:", glGetShaderInfoLog(vertexShader))

        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragmentShader, fragment_source)
        glCompileShader(fragmentShader)

        if not glGetShaderiv(fragmentShader, GL_COMPILE_STATUS):
            print("片段着色器编译错误:", glGetShaderInfoLog(fragmentShader))

        # 链接着色器程序
        program = glCreateProgram()
        glAttachShader(program, vertexShader)
        glAttachShader(program, fragmentShader)
        glLinkProgram(program)

        if not glGetProgramiv(program, GL_LINK_STATUS):
            print("着色器程序链接错误:", glGetProgramInfoLog(program))

        # 删除着色器，它们已经链接到程序中，不再需要
        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)

        return program

    def setupGeometry(self):
        # 三角形顶点数据
        vertices = np.array([
            # 位置              # 颜色
            -0.5, -0.5, 0.0,   1.0, 0.0, 0.0,  # 左下
             0.5, -0.5, 0.0,   0.0, 1.0, 0.0,  # 右下
             0.0,  0.5, 0.0,   0.0, 0.0, 1.0   # 顶部
        ], dtype=np.float32)

        # 创建VAO和VBO
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)

        # 绑定VAO
        glBindVertexArray(self.vao)

        # 绑定VBO并传输数据
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # 设置顶点属性指针
        # 位置属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
        glEnableVertexAttribArray(0)

        # 颜色属性
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        # 解绑
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def resizeGL(self, width, height):
        """处理窗口大小变化"""
        glViewport(0, 0, width, height)

    def paintGL(self):
        """渲染OpenGL场景"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 使用着色器程序
        glUseProgram(self.shader)

        # 旋转三角形
        self.rotation += 1.0
        if self.rotation >= 360.0:
            self.rotation = 0.0

        # 创建旋转矩阵
        angle = np.radians(self.rotation)
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)

        rotation_matrix = np.array([
            [cos_val, -sin_val, 0.0, 0.0],
            [sin_val, cos_val, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # 设置uniform变量
        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, rotation_matrix)

        # 绘制三角形
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("PySide6 OpenGL 示例")
        self.setGeometry(100, 100, 800, 600)

        # 创建OpenGL小部件
        self.gl_widget = OpenGLWidget()
        self.setCentralWidget(self.gl_widget)


if __name__ == "__main__":
    # 设置OpenGL格式
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3)
    gl_format.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(gl_format)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())