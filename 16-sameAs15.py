import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import *
import ctypes

# Shader sources - containing core shader code
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
}
"""

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

void emit_gl_Position(vec3 p) {
    gl_Position = vec4(p, 1.0);
}

vec4 finalize_color(vec4 base_color, vec3 position, vec3 normal) {
    return base_color;
}

vec3 point_on_quadratic(float t, vec3 c0, vec3 c1, vec3 c2){
    float omt = 1.0 - t;
    return c0 * omt * omt + (c0 + 0.5 * c1) * 2.0 * t * omt + (c0 + c1 + c2) * t * t;
}

vec3 tangent_on_quadratic(float t, vec3 c1, vec3 c2){
    return normalize(c1 + 2.0 * c2 * t);
}

vec3 project(vec3 vect, vec3 unit_normal){
    return vect - dot(vect, unit_normal) * unit_normal;
}

vec3 rotate_vector(vec3 vect, vec3 unit_normal, float angle){
    vec3 perp = cross(unit_normal, vect);
    return cos(angle) * vect + sin(angle) * perp;
}

vec3 step_to_corner(vec3 point, vec3 tangent, vec3 unit_normal, float joint_angle, bool inside_curve, bool draw_flat){
    vec3 unit_tan = normalize(tangent);
    vec3 view_normal = draw_flat ? v_unit_normal[1] : unit_normal;
    vec3 step = normalize(cross(view_normal, unit_tan));

    if (inside_curve || int(joint_type) == NO_JOINT || joint_angle == 0.0) {
        return step;
    }

    float cos_angle = cos(joint_angle);
    float sin_angle = sin(joint_angle);

    if (abs(cos_angle) > COS_THRESHOLD) {
        return step;
    }

    float miter_factor;
    if (int(joint_type) == BEVEL_JOINT){
        miter_factor = 0.0;
    } else if (int(joint_type) == MITER_JOINT){
        miter_factor = 1.0;
    } else {
        float mcat1 = MITER_COS_ANGLE_THRESHOLD;
        float mcat2 = mix(mcat1, -1.0, 0.5);
        miter_factor = smoothstep(mcat1, mcat2, cos_angle);
    }

    if (abs(sin_angle) < 1e-6) return step;

    float shift = (cos_angle + mix(-1.0, 1.0, miter_factor)) / sin_angle;
    return normalize(step + shift * unit_tan);
}

void emit_point_with_width(
    vec3 point,
    vec3 tangent,
    float joint_angle,
    float width,
    vec4 joint_color,
    bool inside_curve,
    bool draw_flat
){
    vec3 unit_normal = draw_flat ? v_unit_normal[1] : normalize(camera_position - point);
    color = finalize_color(joint_color, point, unit_normal);
    vec3 step_dir = step_to_corner(point, tangent, unit_normal, joint_angle, inside_curve, draw_flat);
    float aaw = max(anti_alias_width * pixel_size, 1e-8);

    for (int side = -1; side <= 1; side += 2){
        float dist_from_center = side * 0.5 * (width + aaw);
        emit_gl_Position(point + dist_from_center * step_dir);
        half_width_to_aaw = 0.5 * width / aaw;
        dist_to_aaw = dist_from_center / aaw;
        EmitVertex();
    }
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

FRAGMENT_SHADER_SRC = """
#version 330 core
in float dist_to_aaw;
in float half_width_to_aaw;
in vec4 color;

out vec4 frag_color;

void main() {
    frag_color = color;
    float signed_dist_to_region = abs(dist_to_aaw) - half_width_to_aaw;
    frag_color.a *= smoothstep(0.5, -0.5, signed_dist_to_region);
    if (frag_color.a <= 0.0) {
        discard;
    }
}
"""

# Joint type constants
NO_JOINT = 0
AUTO_JOINT = 1
BEVEL_JOINT = 2
MITER_JOINT = 3


class BezierWidget(QOpenGLWidget):
    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.program = self.createShaderProgram(VERTEX_SHADER_SRC, GEOMETRY_SHADER_SRC, FRAGMENT_SHADER_SRC)
        glUseProgram(self.program)

        # Define vertex data
        points = np.array([
            [-0.7, -0.5, 0.0],  # P0
            [ 0.0,  0.8, 0.0],  # P1
            [ 0.7, -0.5, 0.0]   # P2
        ], dtype=np.float32)

        stroke_rgbas = np.array([
            [1.0, 0.0, 0.0, 1.0],  # Red at P0
            [0.0, 1.0, 0.0, 1.0],  # Green at P1 (unused)
            [0.0, 0.0, 1.0, 1.0]   # Blue at P2
        ], dtype=np.float32)

        stroke_widths = np.array([
            [5.0],   # Width at P0
            [5.0],   # Width at P1 (unused)
            [10.0]   # Width at P2
        ], dtype=np.float32)

        joint_angles = np.array([
            [0.0],  # Angle before P0
            [0.0],  # Angle at P1 (unused)
            [0.0]   # Angle after P2
        ], dtype=np.float32)

        unit_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Vertex count and data preparation
        self.vertex_count = 3
        data = np.hstack([points, stroke_rgbas, stroke_widths, joint_angles, unit_normals]).astype(np.float32)
        stride = data.itemsize * (3 + 4 + 1 + 1 + 3)

        # Create and setup VAO/VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        # Configure vertex attributes
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
        offset += stroke_widths.itemsize * 1

        # Location 3: joint_angle (float)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(3)
        offset += joint_angles.itemsize * 1

        # Location 4: unit_normal (vec3)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(4)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # Get uniform locations
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

        # Set uniform values
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

        # Draw call
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
        self.setWindowTitle("OpenGL Bezier Stroke Rendering")
        self.setGeometry(100, 100, 800, 600)
        self.setCentralWidget(BezierWidget())


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