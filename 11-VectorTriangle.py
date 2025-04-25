import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import *

# test：在 Fragment Shader 中以 SDF 方式绘制抗锯齿三角形（Vertex Shader 绘制全屏四边形作为画布）

vertex_shader_src = """
#version 330 core
layout (location = 0) in vec2 aPos;
out vec2 fragPos;
void main() {
    fragPos = aPos;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

fragment_shader_src = """
#version 330 core
in vec2 fragPos;
out vec4 FragColor;

uniform vec2 V1;
uniform vec2 V2;
uniform vec2 V3;

uniform float lineWidth;
uniform vec4 lineColor;
uniform vec4 backgroundColor;

float pointLineDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    float d1 = pointLineDist(fragPos, V1, V2);
    float d2 = pointLineDist(fragPos, V2, V3);
    float d3 = pointLineDist(fragPos, V3, V1);
    float minDist = min(min(d1, d2), d3);

    float aa = fwidth(minDist);
    float alpha = smoothstep(lineWidth * 0.5 + aa, lineWidth * 0.5 - aa, minDist);

    vec4 color = mix(backgroundColor, lineColor, alpha);
    FragColor = color;
}
"""

class TriangleWidget(QOpenGLWidget):
    def initializeGL(self):
        glClearColor(0, 0, 0, 1)

        self.program = self.createShaderProgram(vertex_shader_src, fragment_shader_src)
        glUseProgram(self.program)

        # Fullscreen quad
        vertices = np.array([
            -1, -1,
             1, -1,
             1,  1,
            -1,  1
        ], dtype=np.float32)

        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindVertexArray(0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)

        # Triangle definition in NDC
        glUniform2f(glGetUniformLocation(self.program, "V1"), -0.5, -0.5)
        glUniform2f(glGetUniformLocation(self.program, "V2"), 0.5, -0.5)
        glUniform2f(glGetUniformLocation(self.program, "V3"), 0.0, 0.5)

        glUniform1f(glGetUniformLocation(self.program, "lineWidth"), 0.02)
        glUniform4f(glGetUniformLocation(self.program, "lineColor"), 1.0, 0.5, 0.0, 1.0)
        glUniform4f(glGetUniformLocation(self.program, "backgroundColor"), 0.1, 0.1, 0.1, 1.0)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def createShaderProgram(self, vert_src, frag_src):
        def compile_shader(src, shader_type):
            shader = glCreateShader(shader_type)
            glShaderSource(shader, src)
            glCompileShader(shader)
            if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
                raise RuntimeError(glGetShaderInfoLog(shader).decode())
            return shader

        vs = compile_shader(vert_src, GL_VERTEX_SHADER)
        fs = compile_shader(frag_src, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(program).decode())

        glDeleteShader(vs)
        glDeleteShader(fs)
        return program

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时矢量三角形")
        self.setGeometry(100, 100, 800, 600)
        self.setCentralWidget(TriangleWidget())

if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
