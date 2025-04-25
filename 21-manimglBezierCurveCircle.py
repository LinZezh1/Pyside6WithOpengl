import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import *
import ctypes
import math

# 顶点着色器 (与 Square 版本相同)
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

# 几何着色器 (基于 Square 版本修改 - 增加 MAX_STEPS 并调整循环逻辑)
GEOMETRY_SHADER_SRC = """
#version 330 core
layout (triangles) in; // Input: P0, P1, P2 for one Bezier segment
// Increase max_vertices if MAX_STEPS is very large: (MAX_STEPS * 2)
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
in vec3 verts[3]; // P0, P1, P2
in float v_joint_angle[3]; // Joint angles at P0 and P2 vertices
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
// *** INCREASE MAX_STEPS for curve smoothness ***
const int MAX_STEPS = 10; // Number of subdivisions per Bezier segment

// --- Helper Functions (same as before) ---
vec3 point_on_quadratic(float t, vec3 c0, vec3 c1, vec3 c2){
    // (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
    // = P0 - 2tP0 + t^2P0 + 2tP1 - 2t^2P1 + t^2P2
    // = P0 + (2P1 - 2P0)t + (P0 - 2P1 + P2)t^2
    // = c0 + c1*t + c2*t^2
    return c0 + c1 * t + c2 * t * t;
}

vec3 tangent_on_quadratic(float t, vec3 c1, vec3 c2){
    // B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
    //      = 2(P1-P0) + 2t(P2-P1 - (P1-P0))
    //      = (2P1 - 2P0) + t * (2P2 - 4P1 + 2P0)
    //      = c1 + 2*t*c2
    vec3 tangent = c1 + 2.0 * c2 * t;
    if (length(tangent) < 1e-6) {
         tangent = vec3(1.0, 0.0, 0.0); // Fallback
    }
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

    // Apply joint logic ONLY if NOT inside the curve AND joint angle is present
    if (inside_curve || joint_type_int == NO_JOINT || joint_angle == 0.0) {
        return step_perp;
    }

    float cos_angle = cos(joint_angle);
    float sin_angle = sin(joint_angle);
    if (abs(cos_angle) > COS_THRESHOLD || abs(sin_angle) < 1e-6) return step_perp;

    float miter_factor;
    if (joint_type_int == BEVEL_JOINT){ miter_factor = 0.0; }
    else if (joint_type_int == MITER_JOINT){ miter_factor = 1.0; }
    else { // AUTO_JOINT
        float mcat1 = MITER_COS_ANGLE_THRESHOLD;
        float mcat2 = mix(mcat1, -1.0, 0.5);
        miter_factor = smoothstep(mcat1, mcat2, cos_angle);
    }
    float shift = (cos_angle + mix(-1.0, 1.0, miter_factor)) / sin_angle;
    return normalize(step_perp + shift * tangent);
}

void emit_point_with_width(
    vec3 point, vec3 tangent, float joint_angle,
    float width, vec4 point_color,
    bool inside_curve, bool draw_flat, int joint_type_int
) {
    // Get offset direction (handles joints only if not inside_curve)
    vec3 step_dir = step_to_corner(point, tangent, joint_angle, inside_curve, draw_flat, joint_type_int);
    float aaw = max(anti_alias_width * pixel_size, 1e-8);
    float hw = 0.5 * width;

    color = point_color;
    half_width_to_aaw = hw / aaw;

    for (int side = -1; side <= 1; side += 2) {
        float dist_from_center = side * 0.5 * (width + aaw);
        vec4 clip_pos = vec4(point + dist_from_center * step_dir, 1.0);
        gl_Position = clip_pos;
        dist_to_aaw = dist_from_center / aaw;
        EmitVertex();
    }
}

// --- GS Main ---
void main() {
    // Basic checks (can be simplified if P0!=P2 is guaranteed by Python)
    // if (verts[0] == verts[2]) return;
    if (vec3(v_stroke_width[0], v_stroke_width[1], v_stroke_width[2]) == vec3(0.0)) return;
    if (vec3(v_color[0].a, v_color[1].a, v_color[2].a) == vec3(0.0)) return;

    bool draw_flat = bool(flat_stroke) || bool(is_fixed_in_frame);
    int joint_type_int = int(joint_type);

    vec3 P0 = verts[0];
    vec3 P1 = verts[1]; // Actual control point P1 for curve
    vec3 P2 = verts[2];

    // Calculate quadratic bezier coefficients using the actual P1
    vec3 c0 = P0;
    vec3 c1 = 2.0 * (P1 - P0);
    vec3 c2 = P0 - 2.0 * P1 + P2;

    // *** Use MAX_STEPS to determine number of subdivisions ***
    int n_steps = MAX_STEPS;
    if (n_steps < 2) n_steps = 2; // Ensure at least start and end points

    for (int i = 0; i < n_steps; i++){
        float t = float(i) / float(n_steps - 1); // Parameter t from 0 to 1

        // Calculate point and tangent on the Bezier curve
        vec3 point = point_on_quadratic(t, c0, c1, c2);
        vec3 tangent = tangent_on_quadratic(t, c1, c2);

        // Interpolate width and color along the curve segment
        // Note: v_color[1] and v_stroke_width[1] correspond to P1, might not be intuitive
        // Simple linear interpolation between P0 and P2 values:
        float current_stroke_width = mix(v_stroke_width[0], v_stroke_width[2], t);
        vec4 current_color = mix(v_color[0], v_color[2], t);

        // *** Determine joint angle and inside_curve flag ***
        float current_joint_angle = 0.0; // Default: no joint handling for intermediate points
        bool inside_curve = false;       // Default: assume endpoint

        if (i == 0) { // First point (t=0)
            current_joint_angle = -v_joint_angle[0]; // Use start joint angle
            inside_curve = false;
        } else if (i == n_steps - 1) { // Last point (t=1)
            current_joint_angle = v_joint_angle[2]; // Use end joint angle
            inside_curve = false;
        } else { // Intermediate points (0 < t < 1)
            current_joint_angle = 0.0; // No joint turn
            inside_curve = true;      // Mark as inside the curve
        }

        // Emit the pair of vertices for this point on the stroke
        emit_point_with_width(
            point, tangent, current_joint_angle,
            current_stroke_width, current_color,
            inside_curve, // Pass inside_curve flag
            draw_flat, joint_type_int
        );
    }
    EndPrimitive(); // Finish the triangle strip for this Bezier segment
}
"""

# 片段着色器 (与 Square 版本相同)
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
    if (frag_color.a <= 0.0) {
        discard;
    }
}
"""

# Joint type constants (same)
NO_JOINT = 0
AUTO_JOINT = 1
BEVEL_JOINT = 2
MITER_JOINT = 3


class CircleWidget(QOpenGLWidget): # Renamed widget
    def __init__(self, parent=None):
        super().__init__(parent)
        self.program = None
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        self.uniform_locs = {}

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_DEPTH_TEST)

        self.program = self.createShaderProgram(VERTEX_SHADER_SRC, GEOMETRY_SHADER_SRC, FRAGMENT_SHADER_SRC)
        glUseProgram(self.program)

        # --- Circle Parameters ---
        radius = 0.7
        num_segments = 16
        delta_angle = 2.0 * math.pi / num_segments
        d_theta_half = delta_angle / 2.0
        cos_d_theta_half = math.cos(d_theta_half)

        all_points = []
        all_joint_angles = []

        print(f"Approximating circle with {num_segments} quadratic Bezier segments.")
        print(f"Angle per segment (Degrees): {math.degrees(delta_angle)}")

        # --- Calculate Bezier Control Points for each segment ---
        for i in range(num_segments):
            theta0 = i * delta_angle
            theta1 = (i + 1) * delta_angle
            mid_theta = (theta0 + theta1) / 2.0

            # Endpoints on circle
            P0 = np.array([radius * math.cos(theta0), radius * math.sin(theta0), 0.0], dtype=np.float32)
            P2 = np.array([radius * math.cos(theta1), radius * math.sin(theta1), 0.0], dtype=np.float32)

            # Calculate control point P1 (intersection of tangents)
            # Distance from origin to P1
            op1_dist = radius / cos_d_theta_half
            # P1 lies on angle bisector
            P1 = np.array([op1_dist * math.cos(mid_theta), op1_dist * math.sin(mid_theta), 0.0], dtype=np.float32)

            all_points.extend([P0, P1, P2])

            # Joint angle at each vertex on the circle is delta_angle (positive for CCW)
            joint_angle = delta_angle
            all_joint_angles.extend([[joint_angle], [0.0], [joint_angle]]) # Angle at P0, P1(dummy), P2

        points = np.array(all_points, dtype=np.float32)
        joint_angles = np.array(all_joint_angles, dtype=np.float32)
        self.vertex_count = num_segments * 3 # Total vertices

        # --- Other vertex data ---
        # Use a single color for the circle
        circle_color = [0.2, 0.6, 1.0, 1.0] # Light blue
        stroke_rgbas_list = [circle_color] * self.vertex_count
        stroke_rgbas = np.array(stroke_rgbas_list, dtype=np.float32)

        # Use constant width
        stroke_width_val = 5.0
        stroke_widths = np.array([[stroke_width_val]] * self.vertex_count, dtype=np.float32)

        # Use constant normal
        unit_normals = np.array([[0.0, 0.0, 1.0]] * self.vertex_count, dtype=np.float32)

        # --- VBO/VAO Setup (Pos, RGBA, Width, Angle, Normal) ---
        data = np.hstack([points, stroke_rgbas, stroke_widths, joint_angles, unit_normals]).astype(np.float32)
        stride = data.itemsize * (3 + 4 + 1 + 1 + 3) # 48 bytes

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        offset = 0
        # Location 0: point
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(0); offset += 12
        # Location 1: stroke_rgba
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(1); offset += 16
        # Location 2: stroke_width
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(2); offset += 4
        # Location 3: joint_angle
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(3); offset += 4
        # Location 4: unit_normal
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(4); # offset += 12

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # Get uniform locations
        self.uniform_locs = {name: glGetUniformLocation(self.program, name) for name in [
            "frame_scale", "is_fixed_in_frame", "scale_stroke_with_zoom",
            "anti_alias_width", "flat_stroke", "pixel_size", "joint_type", "camera_position"
        ]}
        for name, loc in self.uniform_locs.items():
            if loc == -1: print(f"Warning: Uniform '{name}' not found or inactive.")


    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)

        width, height = self.width(), self.height()
        pixel_size = 2.0 / max(height, 1) if height > 0 else 0.002

        # Set Uniforms
        glUniform1f(self.uniform_locs.get("frame_scale", -1), 1.0)
        glUniform1f(self.uniform_locs.get("is_fixed_in_frame", -1), 0.0)
        glUniform1f(self.uniform_locs.get("scale_stroke_with_zoom", -1), 1.0)
        glUniform1f(self.uniform_locs.get("anti_alias_width", -1), 1.5)
        glUniform1f(self.uniform_locs.get("flat_stroke", -1), 1.0) # Use flat for 2D
        glUniform1f(self.uniform_locs.get("pixel_size", -1), pixel_size)
        # --- Set Joint Type ---
        # Miter might look strange for shallow angles, Bevel or Auto might be better
        glUniform1f(self.uniform_locs.get("joint_type", -1), float(AUTO_JOINT))
        # glUniform1f(self.uniform_locs.get("joint_type", -1), float(BEVEL_JOINT))
        # glUniform1f(self.uniform_locs.get("joint_type", -1), float(MITER_JOINT))
        # --------------------
        glUniform3f(self.uniform_locs.get("camera_position", -1), 0.0, 0.0, 3.0)

        # Draw the circle (16 Bezier segments)
        glBindVertexArray(self.vao)
        # Draw all segments. Input primitive is GL_TRIANGLES (P0, P1, P2).
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        if self.program and self.uniform_locs.get("pixel_size", -1) != -1:
             glUseProgram(self.program)
             pixel_size = 2.0 / max(h, 1) if h > 0 else 0.002
             glUniform1f(self.uniform_locs["pixel_size"], pixel_size)

    # createShaderProgram function remains the same
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
        self.setWindowTitle("Manimgl BezierCurve Shader draw cirlce") # Updated title
        self.setGeometry(100, 100, 800, 800) # Make window square
        self.setCentralWidget(CircleWidget()) # Use the new widget


if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSamples(4)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())