import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget ,QVBoxLayout# Added QWidget import back
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import Qt
from OpenGL.GL import *
import ctypes
import math

# 顶点着色器 (与 Square/Circle 版本相同 - 需要 joint_angle)
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
    // gl_Position 由 GS 设置
}
"""

# 几何着色器 (与 Square 版本相同 - 处理直线段和接头)
# 注意: 这个版本的GS内部 MAX_STEPS=2 用于直线
GEOMETRY_SHADER_SRC = """
#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 64) out;

// Uniforms
uniform float anti_alias_width;
uniform float flat_stroke;
uniform float pixel_size;
uniform float joint_type; // int as float
uniform float frame_scale;
uniform vec3 camera_position;
uniform float is_fixed_in_frame;

// Inputs from VS
in vec3 verts[3];
in float v_joint_angle[3];
in float v_stroke_width[3];
in vec4 v_color[3];
in vec3 v_unit_normal[3];

// Outputs to FS
out vec4 color;
out float dist_to_aaw;
out float half_width_to_aaw;

// Joint types constants
const int NO_JOINT = 0;
const int AUTO_JOINT = 1;
const int BEVEL_JOINT = 2;
const int MITER_JOINT = 3;

// Constants
const float PI = 3.1415926535;
const float COS_THRESHOLD = 0.999;
const float MITER_COS_ANGLE_THRESHOLD = -0.8;
const int MAX_STEPS = 2; // For straight lines (P0, P2)

// --- Helper Functions ---
vec3 point_on_quadratic(float t, vec3 c0, vec3 c1, vec3 c2){
    return c0 + c1 * t + c2 * t * t; // Correct for line if P1=(P0+P2)/2
}

vec3 tangent_on_quadratic(float t, vec3 c1, vec3 c2){
    vec3 tangent = c1 + 2.0 * c2 * t; // Correct for line if P1=(P0+P2)/2
    if (length(tangent) < 1e-6) tangent = vec3(1.0, 0.0, 0.0);
    return normalize(tangent);
}

vec3 step_to_corner(vec3 point, vec3 tangent, float joint_angle, bool inside_curve, bool draw_flat, int joint_type_int) {
    vec3 step_perp;
    if (draw_flat) {
        step_perp = normalize(vec3(-tangent.y, tangent.x, 0.0));
        if (length(step_perp) < 1e-6) step_perp = vec3(0.0, 1.0, 0.0);
    } else {
        vec3 view_vec = normalize(camera_position - point);
        step_perp = cross(tangent, view_vec);
         if (length(step_perp) < 1e-6) {
             step_perp = normalize(vec3(-tangent.y, tangent.x, 0.0));
             if (length(step_perp) < 1e-6) step_perp = vec3(0.0, 1.0, 0.0);
         }
         step_perp = normalize(step_perp);
    }
    if (inside_curve || joint_type_int == NO_JOINT || joint_angle == 0.0) return step_perp;
    float cos_angle = cos(joint_angle); float sin_angle = sin(joint_angle);
    if (abs(cos_angle) > COS_THRESHOLD || abs(sin_angle) < 1e-6) return step_perp;
    float miter_factor;
    if (joint_type_int == BEVEL_JOINT){ miter_factor = 0.0; }
    else if (joint_type_int == MITER_JOINT){ miter_factor = 1.0; }
    else { float mcat1 = MITER_COS_ANGLE_THRESHOLD; float mcat2 = mix(mcat1, -1.0, 0.5); miter_factor = smoothstep(mcat1, mcat2, cos_angle); }
    float shift = (cos_angle + mix(-1.0, 1.0, miter_factor)) / sin_angle;
    return normalize(step_perp + shift * tangent);
}

void emit_point_with_width(vec3 point, vec3 tangent, float joint_angle, float width, vec4 point_color, bool inside_curve, bool draw_flat, int joint_type_int, float anti_alias_width, float pixel_size) {
    vec3 step_dir = step_to_corner(point, tangent, joint_angle, inside_curve, draw_flat, joint_type_int);
    float aaw = max(anti_alias_width * pixel_size, 1e-8);
    float hw = 0.5 * width;
    color = point_color;
    half_width_to_aaw = hw / aaw;
    for (int side = -1; side <= 1; side += 2) {
        float dist_from_center = side * 0.5 * (width + aaw);
        gl_Position = vec4(point + dist_from_center * step_dir, 1.0);
        dist_to_aaw = dist_from_center / aaw;
        EmitVertex();
    }
}

// --- GS Main ---
void main() {
    if (verts[0] == verts[2]) return; // Ignore zero-length lines
    if (vec3(v_stroke_width[0], v_stroke_width[1], v_stroke_width[2]) == vec3(0.0)) return;
    if (vec3(v_color[0].a, v_color[1].a, v_color[2].a) == vec3(0.0)) return;

    bool draw_flat = bool(flat_stroke) || bool(is_fixed_in_frame);
    int joint_type_int = int(joint_type);
    vec3 P0 = verts[0]; vec3 P1 = verts[1]; vec3 P2 = verts[2];
    vec3 c0 = P0; vec3 c1 = 2.0 * (P1 - P0); vec3 c2 = P0 - 2.0 * P1 + P2;

    // For straight lines, only need start (t=0) and end (t=1) points
    int n_steps = MAX_STEPS; // Should be 2

    for (int i = 0; i < n_steps; i++){
        float t = float(i) / float(n_steps - 1); // t=0 or t=1
        vec3 point = point_on_quadratic(t, c0, c1, c2);
        vec3 tangent = tangent_on_quadratic(t, c1, c2);
        float current_stroke_width = mix(v_stroke_width[0], v_stroke_width[2], t);
        vec4 current_color = mix(v_color[0], v_color[2], t);
        float current_joint_angle;
        // Determine joint angle based on endpoint and VBO data
        if (i == 0){ current_joint_angle = -v_joint_angle[0]; }
        else { current_joint_angle = v_joint_angle[2]; } // i == 1 for MAX_STEPS=2

        // Emit the pair of vertices for this point
        // Pass AA uniforms explicitly
        emit_point_with_width(point, tangent, current_joint_angle, current_stroke_width,
                              current_color, false, // inside_curve = false for lines
                              draw_flat, joint_type_int,
                              anti_alias_width, pixel_size);
    }
    EndPrimitive();
}
"""

# 片段着色器 (与 Square/Circle 版本相同)
FRAGMENT_SHADER_SRC = """
#version 330 core
in vec4 color;
in float dist_to_aaw;
in float half_width_to_aaw;
out vec4 frag_color;
void main() {
    frag_color = color;
    float signed_dist_to_region = abs(dist_to_aaw) - half_width_to_aaw;
    frag_color.a *= smoothstep(0.5, -0.5, signed_dist_to_region);
    if (frag_color.a <= 0.0) { discard; }
}
"""

# Joint type constants (same)
NO_JOINT, AUTO_JOINT, BEVEL_JOINT, MITER_JOINT = 0, 1, 2, 3


class HexagonWidget(QOpenGLWidget): # Renamed widget
    def __init__(self, parent=None):
        super().__init__(parent)
        self.program = None
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        self.uniform_locs = {}

    def _get_uniform_locations(self):
        """Helper to get uniform locations."""
        if not self.program: return
        self.uniform_locs = {name: glGetUniformLocation(self.program, name) for name in [
            # VS Uniforms
            "frame_scale", "is_fixed_in_frame", "scale_stroke_with_zoom",
            # GS Uniforms
            "anti_alias_width", "flat_stroke", "pixel_size", "joint_type",
            "camera_position"
            # Note: max_steps_uniform is not needed for this shader version
        ]}
        for name, loc in self.uniform_locs.items():
            if loc == -1: print(f"Warning: Uniform '{name}' not found or inactive.")

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_DEPTH_TEST)

        try:
            self.program = self.createShaderProgram(VERTEX_SHADER_SRC, GEOMETRY_SHADER_SRC, FRAGMENT_SHADER_SRC)
            glUseProgram(self.program)
            self._get_uniform_locations()
            print("Shaders compiled and linked successfully.")
        except RuntimeError as e:
            print(f"Shader Error: {e}")
            return

        # --- Define Hexagon Vertices ---
        radius = 0.7
        vertices = []
        for i in range(6):
            angle = i * math.pi / 3.0 # Angles 0, 60, 120, 180, 240, 300 degrees
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vertices.append(np.array([x, y, 0.0], dtype=np.float32))

        # --- Define Line Segments (P0, P1=(P0+P2)/2, P2) ---
        all_points = []
        for i in range(6):
            P0 = vertices[i]
            P2 = vertices[(i + 1) % 6] # Wrap around for the last segment
            P1 = (P0 + P2) / 2.0
            all_points.extend([P0, P1, P2])

        points = np.array(all_points, dtype=np.float32)
        self.vertex_count = 18 # 6 segments * 3 points

        # --- Calculate Joint Angles ---
        # For regular hexagon, CCW turn angle is 60 degrees (PI/3)
        joint_angle = math.pi / 3.0
        joint_angles_list = []
        for _ in range(6):
            joint_angles_list.extend([[joint_angle], [0.0], [joint_angle]])
        joint_angles = np.array(joint_angles_list, dtype=np.float32)
        print(f"Hexagon Joint Angle (Radians): {joint_angle}")
        print(f"Hexagon Joint Angle (Degrees): {math.degrees(joint_angle)}")

        # --- Other vertex data ---
        colors = [
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], # Red
            [1.0, 0.5, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], # Orange
            [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], # Yellow
            [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], # Green
            [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], # Blue
            [0.5, 0.0, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]  # Violet
        ]
        stroke_rgbas = np.array(colors, dtype=np.float32)
        stroke_width_val = 8.0
        stroke_widths = np.array([[stroke_width_val]] * self.vertex_count, dtype=np.float32)
        unit_normals = np.array([[0.0, 0.0, 1.0]] * self.vertex_count, dtype=np.float32)

        # --- VBO/VAO Setup (Pos, RGBA, Width, Angle, Normal) ---
        data = np.hstack([points, stroke_rgbas, stroke_widths, joint_angles, unit_normals]).astype(np.float32)
        stride = data.itemsize * (3 + 4 + 1 + 1 + 3) # 48 bytes

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW) # Use STATIC_DRAW

        offset = 0
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); offset += 12
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); offset += 16
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); offset += 4
        glEnableVertexAttribArray(3); glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); offset += 4
        glEnableVertexAttribArray(4); glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset));

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        print("Hexagon geometry created and VBO/VAO setup complete.")


    def paintGL(self):
        if not self.isValid() or not self.program: return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)

        width, height = self.width(), self.height()
        if height == 0: height = 1
        pixel_size = 2.0 / height

        # Set Uniforms
        glUniform1f(self.uniform_locs.get("frame_scale", -1), 1.0)
        glUniform1f(self.uniform_locs.get("is_fixed_in_frame", -1), 0.0)
        glUniform1f(self.uniform_locs.get("scale_stroke_with_zoom", -1), 1.0)
        glUniform1f(self.uniform_locs.get("anti_alias_width", -1), 1.5)
        glUniform1f(self.uniform_locs.get("flat_stroke", -1), 1.0) # Use flat for 2D
        glUniform1f(self.uniform_locs.get("pixel_size", -1), pixel_size)
        # --- Set Joint Type ---
        # Miter should work well for 60-degree turns (120 internal angle)
        glUniform1f(self.uniform_locs.get("joint_type", -1), float(MITER_JOINT))
        # glUniform1f(self.uniform_locs.get("joint_type", -1), float(BEVEL_JOINT))
        # glUniform1f(self.uniform_locs.get("joint_type", -1), float(AUTO_JOINT)) # Should act like Miter
        # --------------------
        glUniform3f(self.uniform_locs.get("camera_position", -1), 0.0, 0.0, 3.0)

        # Draw the hexagon (6 segments)
        glBindVertexArray(self.vao)
        if self.vertex_count > 0:
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count) # Draw all 18 vertices
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        if self.program and self.isValid():
             glUseProgram(self.program)
             pixel_size = 2.0 / max(h, 1)
             glUniform1f(self.uniform_locs.get("pixel_size", -1), pixel_size)

    def createShaderProgram(self, vert_src, geom_src, frag_src):
        # (Same implementation as previous examples)
        def compile_shader(src, shader_type):
            shader = glCreateShader(shader_type)
            glShaderSource(shader, src)
            glCompileShader(shader)
            if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
                info_log = glGetShaderInfoLog(shader)
                if isinstance(info_log, bytes): info_log = info_log.decode()
                shader_type_str = {GL_VERTEX_SHADER: "Vertex", GL_GEOMETRY_SHADER: "Geometry", GL_FRAGMENT_SHADER: "Fragment"}.get(shader_type, "Unknown")
                raise RuntimeError(f"Shader compile failure ({shader_type_str}):\n{info_log}\n--- Source ---\n{src}\n--------------")
            return shader
        vs = compile_shader(vert_src, GL_VERTEX_SHADER)
        gs = compile_shader(geom_src, GL_GEOMETRY_SHADER)
        fs = compile_shader(frag_src, GL_FRAGMENT_SHADER)
        program = glCreateProgram()
        for s in [vs, gs, fs]: glAttachShader(program, s)
        glLinkProgram(program)
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            info_log = glGetProgramInfoLog(program)
            if isinstance(info_log, bytes): info_log = info_log.decode()
            raise RuntimeError(f"Program link failure:\n{info_log}")
        for s in [vs, gs, fs]: glDetachShader(program, s); glDeleteShader(s)
        return program

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL Hexagon Stroke with Joints") # Updated title
        self.setGeometry(100, 100, 800, 800) # Square window

        # --- Use QWidget as central widget to potentially add controls later if needed ---
        central_widget = QWidget()
        # Use a simple layout for now, just containing the OpenGL widget
        layout = QVBoxLayout(central_widget) # Import QVBoxLayout if not already
        self.hexagon_widget = HexagonWidget()
        layout.addWidget(self.hexagon_widget)
        layout.setContentsMargins(0,0,0,0) # Optional: Remove margins

        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSamples(4)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)
    print("Default QSurfaceFormat set.")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    print("MainWindow shown.")
    sys.exit(app.exec())