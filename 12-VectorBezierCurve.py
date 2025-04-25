import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import *
from PySide6.QtCore import Qt, QTimer

# test: 通过 CPU 计算顶点并使用 GL_LINE_STRIP 渲染贝塞尔曲线
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0, 0.3, 0.2, 1.0);  // 红色
}
"""

def bezier_point(t, p0, p1, p2, p3):
    """计算三阶贝塞尔曲线的某个 t 点"""
    return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.program = None
        self.VAO = None
        self.vertex_count = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def initializeGL(self):
        self.program = self.create_program(vertex_shader_source, fragment_shader_source)

        # 定义三阶贝塞尔的控制点（标准化坐标）
        p0 = np.array([-0.8, -0.6], dtype=np.float32)
        p1 = np.array([-0.4,  0.8], dtype=np.float32)
        p2 = np.array([ 0.4, -0.8], dtype=np.float32)
        p3 = np.array([ 0.8,  0.6], dtype=np.float32)

        # 曲线采样点
        num_points = 200
        t_values = np.linspace(0, 1, num_points, dtype=np.float32)
        curve_points = np.array([bezier_point(t, p0, p1, p2, p3) for t in t_values], dtype=np.float32)

        self.vertex_count = len(curve_points)

        # 创建 VAO 和 VBO
        self.VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, curve_points.nbytes, curve_points, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glLineWidth(3.0)  # 设置线宽

    def paintGL(self):
        glClearColor(0.08, 0.08, 0.08, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_LINE_STRIP, 0, self.vertex_count)
        glBindVertexArray(0)
        glUseProgram(0)

    def create_shader(self, shader_type, source):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader).decode())
        return shader

    def create_program(self, vs_source, fs_source):
        vs = self.create_shader(GL_VERTEX_SHADER, vs_source)
        fs = self.create_shader(GL_FRAGMENT_SHADER, fs_source)
        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(program).decode())
        glDeleteShader(vs)
        glDeleteShader(fs)
        return program

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL 三阶贝塞尔曲线（GL_LINE_STRIP）")
        self.setFixedSize(800, 600)
        self.gl_widget = GLWidget()
        self.setCentralWidget(self.gl_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
