from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
import numpy as np
import sys
import math

# test：使用 Manimgl 的 Stroke 逻辑渲染单条 BezierCurve ，并实现简单动画效果

VERT_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPosA;
layout (location = 1) in vec2 aPosB;

uniform float t;

out vec2 pos;

void main() {
    pos = mix(aPosA, aPosB, t);
}
"""

GEOM_SHADER = """
#version 330 core
layout (points) in;
layout (line_strip, max_vertices = 64) out;

in vec2 pos[];

uniform vec2 p0;
uniform vec2 p1;
uniform vec2 p2;

uniform float t;

void main() {
    for (int i = 0; i <= 63; ++i) {
        float u = float(i) / 63.0;
        vec2 a = mix(p0, p1, u);
        vec2 b = mix(p1, p2, u);
        vec2 point = mix(a, b, u);
        gl_Position = vec4(point, 0.0, 1.0);
        EmitVertex();
    }
    EndPrimitive();
}
"""

FRAG_SHADER = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.2, 0.8, 1.0, 1.0);
}
"""

class BezierWidget(QOpenGLWidget):
    def initializeGL(self):
        self.t = 0.0
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # 控制点 A 和 B
        self.P0A = np.array([-0.8, -0.4], dtype=np.float32)
        self.P1A = np.array([ 0.0,  0.8], dtype=np.float32)
        self.P2A = np.array([ 0.8, -0.4], dtype=np.float32)

        self.P0B = np.array([-0.8,  0.4], dtype=np.float32)
        self.P1B = np.array([ 0.0, -0.8], dtype=np.float32)
        self.P2B = np.array([ 0.8,  0.4], dtype=np.float32)

        # VBO 只是用于触发 draw call，这里实际不使用顶点坐标
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        dummy = np.array([[0.0, 0.0]], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, dummy.nbytes, dummy, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        # 编译 shader
        self.shaderProgram = self.createShaderProgram()

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateScene)
        self.timer.start(16)  # ~60 FPS

    def createShaderProgram(self):
        def compile_shader(source, shader_type):
            shader = glCreateShader(shader_type)
            glShaderSource(shader, source)
            glCompileShader(shader)
            if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
                raise RuntimeError(glGetShaderInfoLog(shader).decode())
            return shader

        vs = compile_shader(VERT_SHADER, GL_VERTEX_SHADER)
        gs = compile_shader(GEOM_SHADER, GL_GEOMETRY_SHADER)
        fs = compile_shader(FRAG_SHADER, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, gs)
        glAttachShader(program, fs)
        glLinkProgram(program)
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(program).decode())
        return program

    def updateScene(self):
        self.t += 0.01
        if self.t > 1.0:
            self.t = 0.0
        self.update()

    def paintGL(self):
        glClearColor(0.05, 0.05, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shaderProgram)

        # 插值控制点
        t = self.t
        p0 = (1 - t) * self.P0A + t * self.P0B
        p1 = (1 - t) * self.P1A + t * self.P1B
        p2 = (1 - t) * self.P2A + t * self.P2B

        glUniform1f(glGetUniformLocation(self.shaderProgram, "t"), t)
        glUniform2f(glGetUniformLocation(self.shaderProgram, "p0"), p0[0], p0[1])
        glUniform2f(glGetUniformLocation(self.shaderProgram, "p1"), p1[0], p1[1])
        glUniform2f(glGetUniformLocation(self.shaderProgram, "p2"), p2[0], p2[1])

        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, 1)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("拓扑一致的贝塞尔动画")
        self.setGeometry(100, 100, 800, 600)
        self.setCentralWidget(BezierWidget())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
