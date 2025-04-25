import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import *

VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec3 aBaseNormal;

out vec3 verts;
out vec4 v_color;
out vec3 v_base_normal;

void main() {
    verts = aPos;
    v_color = aColor;
    v_base_normal = aBaseNormal;
}
"""

GEOMETRY_SHADER_SRC = """
#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 6) out;

in vec3 verts[];
in vec4 v_color[];
in vec3 v_base_normal[];

out vec4 color;
out float fill_all;
out float orientation;
out vec2 uv_coords;

const vec2 SIMPLE_QUADRATIC[3] = vec2[3](
    vec2(0.0, 0.0),
    vec2(0.5, 0.0),
    vec2(1.0, 1.0)
);

void emit_gl_Position(vec3 p) {
    gl_Position = vec4(p, 1.0);
}

vec4 finalize_color(vec4 base_color, vec3 position, vec3 normal) {
    return base_color;
}

void emit_triangle(vec3 points[3], vec4 colors[3], vec3 unit_normal){
    orientation = sign(determinant(mat3(
        unit_normal,
        points[1] - points[0],
        points[2] - points[0]
    )));

    for(int i = 0; i < 3; i++){
        uv_coords = SIMPLE_QUADRATIC[i];
        color = finalize_color(colors[i], points[i], unit_normal);
        emit_gl_Position(points[i]);
        EmitVertex();
    }
    EndPrimitive();
}

void main() {
    if (verts[0] == verts[1]) return;
    if (vec3(v_color[0].a, v_color[1].a, v_color[2].a) == vec3(0.0, 0.0, 0.0)) return;

    vec3 base_point = v_base_normal[0];
    vec3 unit_normal = v_base_normal[1];

    fill_all = 1.0;
    emit_triangle(
        vec3[3](base_point, verts[0], verts[2]),
        vec4[3](v_color[1], v_color[0], v_color[2]),
        unit_normal
    );

    fill_all = 0.0;
    emit_triangle(
        vec3[3](verts[0], verts[1], verts[2]),
        vec4[3](v_color[0], v_color[1], v_color[2]),
        unit_normal
    );
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core

uniform bool winding;

in vec4 color;
in float fill_all;
in float orientation;
in vec2 uv_coords;

out vec4 frag_color;

void main() {
    if (color.a == 0.0) discard;
    frag_color = color;

    float a = 0.95 * frag_color.a;
    if(orientation < 0) a = -a / (1.0 - a);
    frag_color.a = a;

    if (bool(fill_all)) return;

    float x = uv_coords.x;
    float y = uv_coords.y;
    float Fxy = (y - x * x);
    if(Fxy < 0.0) discard;
}
"""

class BezierWidget(QOpenGLWidget):
    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)

        self.program = self.createShaderProgram(VERTEX_SHADER_SRC, GEOMETRY_SHADER_SRC, FRAGMENT_SHADER_SRC)
        glUseProgram(self.program)

        # 控制点设置（可以自定义成其他三角形组合）
        vertices = np.array([
            [-0.5, -0.5, 0.0],
            [ 0.0,  0.5, 0.0],
            [ 0.5, -0.5, 0.0]
        ], dtype=np.float32)

        colors = np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ], dtype=np.float32)

        base_normals = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float32)

        self.vertex_count = 3
        data = np.hstack([vertices, colors, base_normals]).astype(np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        stride = data.shape[1] * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        glUniform1i(glGetUniformLocation(self.program, "winding"), GL_TRUE)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def createShaderProgram(self, vert_src, geom_src, frag_src):
        def compile_shader(src, shader_type):
            shader = glCreateShader(shader_type)
            glShaderSource(shader, src)
            glCompileShader(shader)
            if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
                raise RuntimeError(glGetShaderInfoLog(shader).decode())
            return shader

        vs = compile_shader(vert_src, GL_VERTEX_SHADER)
        gs = compile_shader(geom_src, GL_GEOMETRY_SHADER)
        fs = compile_shader(frag_src, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        for shader in [vs, gs, fs]:
            glAttachShader(program, shader)
        glLinkProgram(program)
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(program).decode())

        for shader in [vs, gs, fs]:
            glDeleteShader(shader)
        return program

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL 贝塞尔曲线渲染")
        self.setGeometry(100, 100, 800, 600)
        self.setCentralWidget(BezierWidget())

if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
