import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QTimer
from OpenGL.GL import *

# test：使用 Manimgl 的 Stroke 逻辑渲染两个图形（三角形和正方形），实现二者转换的动画效果

VERT_SHADER = """
#version 330 core
layout (location = 0) in vec2 p0A;
layout (location = 1) in vec2 p1A;
layout (location = 2) in vec2 p2A;
layout (location = 3) in vec2 p0B;
layout (location = 4) in vec2 p1B;
layout (location = 5) in vec2 p2B;

uniform float t;
out vec2 cp0, cp1, cp2;

void main() {
    cp0 = mix(p0A, p0B, t);
    cp1 = mix(p1A, p1B, t);
    cp2 = mix(p2A, p2B, t);
}
"""

GEOM_SHADER = """
#version 330 core
layout (points) in;
layout (line_strip, max_vertices = 128) out;

in vec2 cp0[];
in vec2 cp1[];
in vec2 cp2[];

void main() {
    for (int i = 0; i <= 32; ++i) {
        float u = float(i) / 32.0;
        vec2 a = mix(cp0[0], cp1[0], u);
        vec2 b = mix(cp1[0], cp2[0], u);
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
    FragColor = vec4(1.0, 0.7, 0.2, 1.0);
}
"""

class BezierMorphWidget(QOpenGLWidget):
    def initializeGL(self):
        self.t = 0.0
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.triangle = self.create_triangle()
        self.square = self.create_square()

        self.total_segments = len(self.square)  # 对齐段数为 4

        # 展平成 VBO 格式，每段6个vec2点（3 from A, 3 from B）
        data = []
        for i in range(self.total_segments):
            pa = self.triangle[i]
            pb = self.square[i]
            data.extend(pa + pb)

        self.vbo_data = np.array(data, dtype=np.float32)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vbo_data.nbytes, self.vbo_data, GL_STATIC_DRAW)

        for i in range(6):
            glEnableVertexAttribArray(i)
            glVertexAttribPointer(i, 2, GL_FLOAT, GL_FALSE, 6 * 2 * 4, ctypes.c_void_p(i * 8))

        self.shader = self.create_shader_program()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_scene)
        self.timer.start(16)

    def create_triangle(self):
        # 创建 4 段，前 3 段构成等边三角形，第 4 段保持与第 3 段重复
        R = 0.4
        cx = -0.5
        angles = [np.pi / 2 + i * 2 * np.pi / 3 for i in range(3)]
        verts = [np.array([cx + R * np.cos(a), R * np.sin(a)], dtype=np.float32) for a in angles]
        segments = []
        for i in range(3):
            p0 = verts[i]
            p2 = verts[(i + 1) % 3]
            p1 = (p0 + p2) / 2 + np.array([0.0, 0.1], dtype=np.float32)  # 控制点略微外凸
            segments.append([p0, p1, p2])
        segments.append(segments[-1])  # 填补到4段
        return segments

    def create_square(self):
        # 正方形右侧，中心在 +0.5
        cx = 0.5
        s = 0.4
        verts = [
            np.array([cx - s, -s], dtype=np.float32),
            np.array([cx - s,  s], dtype=np.float32),
            np.array([cx + s,  s], dtype=np.float32),
            np.array([cx + s, -s], dtype=np.float32),
        ]
        segments = []
        for i in range(4):
            p0 = verts[i]
            p2 = verts[(i + 1) % 4]
            p1 = (p0 + p2) / 2  # 控制点 = 中点
            segments.append([p0, p1, p2])
        return segments

    def create_shader_program(self):
        def compile_shader(src, type_):
            shader = glCreateShader(type_)
            glShaderSource(shader, src)
            glCompileShader(shader)
            if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
                raise RuntimeError(glGetShaderInfoLog(shader).decode())
            return shader

        vs = compile_shader(VERT_SHADER, GL_VERTEX_SHADER)
        gs = compile_shader(GEOM_SHADER, GL_GEOMETRY_SHADER)
        fs = compile_shader(FRAG_SHADER, GL_FRAGMENT_SHADER)

        prog = glCreateProgram()
        glAttachShader(prog, vs)
        glAttachShader(prog, gs)
        glAttachShader(prog, fs)
        glLinkProgram(prog)
        if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(prog).decode())
        return prog

    def update_scene(self):
        self.t += 0.01
        if self.t > 1.0:
            self.t = 0.0
        self.update()

    def paintGL(self):
        glClearColor(0.08, 0.08, 0.08, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "t"), self.t)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.total_segments)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("贝塞尔拓扑变形：三角形 → 正方形")
        self.setGeometry(100, 100, 900, 600)
        self.setCentralWidget(BezierMorphWidget())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
