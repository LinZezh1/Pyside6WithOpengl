import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat

import numpy as np
from OpenGL.GL import *
from PIL import Image


class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # 约 60 FPS

        self.rotation = 0.0
        self.frame_number = 0
        self.max_frames = 300

        # 确保输出目录存在
        os.makedirs("frames", exist_ok=True)

    def initializeGL(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glEnable(GL_DEPTH_TEST)
        self.setupShaders()
        self.setupGeometry()

    def setupShaders(self):
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
        fragment_shader = """
        #version 330 core
        in vec3 vertexColor;
        out vec4 fragColor;
        void main()
        {
            fragColor = vec4(vertexColor, 1.0);
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
        vertices = np.array([
            -0.5, -0.5, 0.0,  1.0, 0.0, 0.0,
             0.5, -0.5, 0.0,  0.0, 1.0, 0.0,
             0.0,  0.5, 0.0,  0.0, 0.0, 1.0
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)

        self.rotation += 1.0
        if self.rotation >= 360.0:
            self.rotation = 0.0

        angle = np.radians(self.rotation)
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)

        rotation_matrix = np.array([
            [cos_val, -sin_val, 0.0, 0.0],
            [sin_val,  cos_val, 0.0, 0.0],
            [0.0,      0.0,     1.0, 0.0],
            [0.0,      0.0,     0.0, 1.0]
        ], dtype=np.float32)

        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, rotation_matrix)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        self.save_frame()

        self.frame_number += 1
        if self.frame_number >= self.max_frames:
            self.timer.stop()
            print("渲染完成！可以用 ffmpeg 合成视频")
            QApplication.quit()

    def save_frame(self):
        width = self.width()
        height = self.height()
        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        image = np.flip(image, axis=0)

        filename = f"frames/frame_{self.frame_number:04d}.png"
        Image.fromarray(image).save(filename)
        print(f"保存帧：{filename}")


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("OpenGL 序列帧导出")
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
