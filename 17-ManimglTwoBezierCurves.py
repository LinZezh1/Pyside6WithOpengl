import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import *
import ctypes

# 顶点着色器源码
VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 point;
layout (location = 1) in vec4 stroke_rgba;
layout (location = 2) in float stroke_width;
layout (location = 3) in float joint_angle;
layout (location = 4) in vec3 unit_normal;

uniform float frame_scale;
uniform float is_fixed_in_frame;
uniform float scale_stroke_with_zoom;

out vec3 verts;
out vec4 v_color;
out float v_stroke_width;
out float v_joint_angle;
out vec3 v_unit_normal;

const float STROKE_WIDTH_CONVERSION = 0.01;

void main(){
    verts = point;
    v_color = stroke_rgba;
    v_stroke_width = STROKE_WIDTH_CONVERSION * stroke_width * mix(frame_scale, 1.0, scale_stroke_with_zoom);
    v_joint_angle = joint_angle;
    v_unit_normal = unit_normal;
    gl_Position = vec4(point, 1.0);
}
"""

# 几何着色器源码
GEOMETRY_SHADER_SRC = """
#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 64) out;

uniform float anti_alias_width;
uniform float flat_stroke;
uniform float pixel_size;
uniform float joint_type;
uniform float frame_scale;
uniform vec3 camera_position;
uniform float is_fixed_in_frame;

in vec3 verts[3];
in float v_joint_angle[3];
in float v_stroke_width[3];
in vec4 v_color[3];
in vec3 v_unit_normal[3];

out vec4 color;
out float dist_to_aaw;
out float half_width_to_aaw;

const int NO_JOINT = 0;
const int AUTO_JOINT = 1;
const int BEVEL_JOINT = 2;
const int MITER_JOINT = 3;

const float COS_THRESHOLD = 0.999;
const float POLYLINE_FACTOR = 100;
const int MAX_STEPS = 32;
const float MITER_COS_ANGLE_THRESHOLD = -0.8;

vec3 point_on_quadratic(float t, vec3 c0, vec3 c1, vec3 c2) {
    float s = 1.0 - t;
    return s * s * c0 + 2.0 * s * t * c1 + t * t * c2;
}

vec3 tangent_on_quadratic(float t, vec3 c1, vec3 c2) {
    return normalize(2.0 * (1.0 - t) * (c1 - verts[0]) + 2.0 * t * (c2 - c1));
}

vec3 step_to_corner(vec3 point, vec3 tangent, vec3 unit_normal, float joint_angle, bool inside_curve, bool draw_flat) {
    if (draw_flat) {
        return unit_normal;
    }
    if (inside_curve) {
        return normalize(cross(tangent, vec3(0.0, 0.0, 1.0)));
    }
    float cos_angle = cos(joint_angle);
    if (cos_angle < MITER_COS_ANGLE_THRESHOLD) {
        return normalize(cross(tangent, vec3(0.0, 0.0, 1.0)));
    }
    float sin_angle = sin(joint_angle);
    return normalize(cross(tangent, vec3(0.0, 0.0, 1.0))) / cos_angle;
}

void emit_point_with_width(
    vec3 point, vec3 tangent, float joint_angle,
    float stroke_width, vec4 point_color,
    bool inside_curve, bool draw_flat
) {
    vec3 unit_normal = draw_flat ? v_unit_normal[1] : normalize(cross(tangent, vec3(0.0, 0.0, 1.0)));
    vec3 step = step_to_corner(point, tangent, unit_normal, joint_angle, inside_curve, draw_flat);

    float hw = 0.5 * stroke_width;
    vec3 p1 = point + hw * step;
    vec3 p2 = point - hw * step;

    float aaw = anti_alias_width * pixel_size;

    color = point_color;
    half_width_to_aaw = hw / aaw;

    dist_to_aaw = -hw / aaw;
    gl_Position = vec4(p1, 1.0);
    EmitVertex();

    dist_to_aaw = hw / aaw;
    gl_Position = vec4(p2, 1.0);
    EmitVertex();
}

void main() {
    if (verts[0] == verts[1]) return;
    if (vec3(v_stroke_width[0], v_stroke_width[1], v_stroke_width[2]) == vec3(0.0, 0.0, 0.0)) return;
    if (vec3(v_color[0].a, v_color[1].a, v_color[2].a) == vec3(0.0, 0.0, 0.0)) return;

    bool draw_flat = bool(flat_stroke) || bool(is_fixed_in_frame);

    vec3 P0 = verts[0];
    vec3 P1 = verts[1];
    vec3 P2 = verts[2];
    vec3 c0 = P0;
    vec3 c1 = 2.0 * (P1 - P0);
    vec3 c2 = P0 - 2.0 * P1 + P2;

    float area = 0.5 * length(cross(P1 - P0, P2 - P0));
    int count = int(round(POLYLINE_FACTOR * sqrt(area) / frame_scale));
    int n_steps = min(2 + count, MAX_STEPS);

    for (int i = 0; i < MAX_STEPS; i++){
        if (i >= n_steps) break;
        float t = float(i) / (n_steps - 1);

        vec3 point = point_on_quadratic(t, c0, c1, c2);
        vec3 tangent = tangent_on_quadratic(t, c1, c2);

        float stroke_width = mix(v_stroke_width[0], v_stroke_width[2], t);
        vec4 current_color = mix(v_color[0], v_color[2], t);

        bool inside_curve = (i > 0 && i < n_steps - 1);
        float joint_angle;
        if (i == 0){
            joint_angle = -v_joint_angle[0];
        } else if (inside_curve){
            joint_angle = 0.0;
        } else {
            joint_angle = v_joint_angle[2];
        }

        emit_point_with_width(
            point, tangent, joint_angle,
            stroke_width, current_color,
            inside_curve, draw_flat
        );
    }
    EndPrimitive();
}
"""

# 片段着色器源码
FRAGMENT_SHADER_SRC = """
#version 330 core
in vec4 color;
in float dist_to_aaw;
in float half_width_to_aaw;

out vec4 frag_color;

void main() {
    float alpha = 1.0;
    if (abs(dist_to_aaw) > half_width_to_aaw) {
        alpha = max(0.0, (1.0 - abs(dist_to_aaw)) / (1.0 - half_width_to_aaw));
    }
    frag_color = vec4(color.rgb, color.a * alpha);
}
"""

# Joint type constants
NO_JOINT = 0
AUTO_JOINT = 1
BEVEL_JOINT = 2
MITER_JOINT = 3


class TwoBezierWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.program = None

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.program = self.createShaderProgram(VERTEX_SHADER_SRC, GEOMETRY_SHADER_SRC, FRAGMENT_SHADER_SRC)
        glUseProgram(self.program)

        # 定义两条贝塞尔曲线的控制点
        points = np.array([
            # 第一条曲线的控制点
            [-0.8, -0.5, 0.0],  # P0
            [-0.4, 0.5, 0.0],  # P1
            [0.0, -0.5, 0.0],  # P2
            # 第二条曲线的控制点
            [0.0, -0.5, 0.0],  # P0
            [0.4, 0.5, 0.0],  # P1
            [0.8, -0.5, 0.0]  # P2
        ], dtype=np.float32)

        # 为每条曲线设置颜色
        stroke_rgbas = np.array([
            # 第一条曲线的颜色
            [1.0, 0.0, 0.0, 1.0],  # 红色起点
            [1.0, 0.0, 0.0, 1.0],  # 红色控制点
            [1.0, 0.0, 0.0, 1.0],  # 红色终点
            # 第二条曲线的颜色
            [0.0, 0.0, 1.0, 1.0],  # 蓝色起点
            [0.0, 0.0, 1.0, 1.0],  # 蓝色控制点
            [0.0, 0.0, 1.0, 1.0]  # 蓝色终点
        ], dtype=np.float32)

        # 设置描边宽度
        stroke_widths = np.array([
            # 第一条曲线的宽度
            [5.0], [5.0], [5.0],
            # 第二条曲线的宽度
            [5.0], [5.0], [5.0]
        ], dtype=np.float32)

        # 设置关节角度
        joint_angles = np.array([
            # 第一条曲线的角度
            [0.0], [0.0], [0.0],
            # 第二条曲线的角度
            [0.0], [0.0], [0.0]
        ], dtype=np.float32)

        # 设置法线
        unit_normals = np.array([
            # 第一条曲线的法线
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            # 第二条曲线的法线
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # 设置顶点数量
        self.vertex_count = 6  # 两条曲线，每条3个顶点

        # 准备顶点数据
        data = np.hstack([points, stroke_rgbas, stroke_widths, joint_angles, unit_normals]).astype(np.float32)
        stride = data.itemsize * (3 + 4 + 1 + 1 + 3)

        # 创建并设置VAO/VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        # 配置顶点属性
        offset = 0
        # Location 0: point (vec3)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(0)
        offset += points.itemsize * 3

        # Location 1: stroke_rgba (vec4)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(1)
        offset += stroke_rgbas.itemsize * 4

        # Location 2: stroke_width (float)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(2)
        offset += stroke_widths.itemsize

        # Location 3: joint_angle (float)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(3)
        offset += joint_angles.itemsize

        # Location 4: unit_normal (vec3)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(4)

        # 获取uniform位置
        self.uniform_locs = {
            "frame_scale": glGetUniformLocation(self.program, "frame_scale"),
            "is_fixed_in_frame": glGetUniformLocation(self.program, "is_fixed_in_frame"),
            "scale_stroke_with_zoom": glGetUniformLocation(self.program, "scale_stroke_with_zoom"),
            "anti_alias_width": glGetUniformLocation(self.program, "anti_alias_width"),
            "flat_stroke": glGetUniformLocation(self.program, "flat_stroke"),
            "pixel_size": glGetUniformLocation(self.program, "pixel_size"),
            "joint_type": glGetUniformLocation(self.program, "joint_type"),
            "camera_position": glGetUniformLocation(self.program, "camera_position"),
        }

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        # 设置uniform值
        width, height = self.width(), self.height()
        pixel_size = 2.0 / max(height, 1)

        glUniform1f(self.uniform_locs["frame_scale"], 1.0)
        glUniform1f(self.uniform_locs["is_fixed_in_frame"], 0.0)
        glUniform1f(self.uniform_locs["scale_stroke_with_zoom"], 1.0)
        glUniform1f(self.uniform_locs["anti_alias_width"], 1.0)
        glUniform1f(self.uniform_locs["flat_stroke"], 0.0)
        glUniform1f(self.uniform_locs["pixel_size"], pixel_size)
        glUniform1f(self.uniform_locs["joint_type"], float(AUTO_JOINT))
        glUniform3f(self.uniform_locs["camera_position"], 0.0, 0.0, 3.0)

        # 绘制两条曲线
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)  # 绘制第一条曲线
        glDrawArrays(GL_TRIANGLES, 3, 3)  # 绘制第二条曲线
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def createShaderProgram(self, vert_src, geom_src, frag_src):
        def compile_shader(src, shader_type):
            shader = glCreateShader(shader_type)
            glShaderSource(shader, src)
            glCompileShader(shader)
            if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
                info_log = glGetShaderInfoLog(shader)
                if isinstance(info_log, bytes):
                    info_log = info_log.decode()
                raise RuntimeError(f"Shader compilation failed for type {shader_type}:\n{info_log}")
            return shader

        vs = compile_shader(vert_src, GL_VERTEX_SHADER)
        gs = compile_shader(geom_src, GL_GEOMETRY_SHADER)
        fs = compile_shader(frag_src, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        for shader in [vs, gs, fs]:
            glAttachShader(program, shader)
        glLinkProgram(program)
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            info_log = glGetProgramInfoLog(program)
            if isinstance(info_log, bytes):
                info_log = info_log.decode()
            raise RuntimeError(f"Shader program linking failed:\n{info_log}")

        for shader in [vs, gs, fs]:
            glDetachShader(program, shader)
            glDeleteShader(shader)
        return program


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two Bezier Curves OpenGL Rendering")
        self.setGeometry(100, 100, 800, 600)
        self.setCentralWidget(TwoBezierWidget())


if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())