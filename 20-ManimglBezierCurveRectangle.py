import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import *
import ctypes
import math

# 顶点着色器 (来自第一个代码块 - 处理接头角度)
VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 point;
layout (location = 1) in vec4 stroke_rgba;
layout (location = 2) in float stroke_width;
layout (location = 3) in float joint_angle; // 需要这个来处理接头
layout (location = 4) in vec3 unit_normal;

uniform float frame_scale;
uniform float is_fixed_in_frame;
uniform float scale_stroke_with_zoom;

out vec3 verts;
out vec4 v_color;
out float v_stroke_width;
out float v_joint_angle; // 传递给 GS
out vec3 v_unit_normal;

const float STROKE_WIDTH_CONVERSION = 0.01; // 调整此值以匹配期望的视觉宽度

void main(){
    verts = point;
    v_color = stroke_rgba;
    // 根据窗口缩放调整线宽（如果需要）
    v_stroke_width = STROKE_WIDTH_CONVERSION * stroke_width * mix(frame_scale, 1.0, scale_stroke_with_zoom);
    v_joint_angle = joint_angle; // 传递接头角度
    v_unit_normal = unit_normal;
    // gl_Position 由 GS 设置
}
"""

# 几何着色器 (来自第一个代码块 - 处理接头)
GEOMETRY_SHADER_SRC = """
#version 330 core
layout (triangles) in; // 输入 P0, P1, P2
layout (triangle_strip, max_vertices = 64) out; // 输出描边

// Uniforms for appearance and joint handling
uniform float anti_alias_width;
uniform float flat_stroke;
uniform float pixel_size;
uniform float joint_type; // int as float: 0=None, 1=Auto, 2=Bevel, 3=Miter
uniform float frame_scale;
uniform vec3 camera_position;
uniform float is_fixed_in_frame;

// Inputs from VS
in vec3 verts[3]; // P0, P1, P2
in float v_joint_angle[3]; // Joint angles at P0 and P2
in float v_stroke_width[3];
in vec4 v_color[3];
in vec3 v_unit_normal[3]; // Surface normal (unused in this 2D case)

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
const float COS_THRESHOLD = 0.999; // Near 1 or -1 (cos(angle) threshold for straight lines)
const float MITER_COS_ANGLE_THRESHOLD = -0.8; // Cosine angle threshold for auto miter->bevel
const int MAX_STEPS = 2; // For straight lines (P0, P2)

// --- Helper Functions ---
vec3 point_on_quadratic(float t, vec3 c0, vec3 c1, vec3 c2){
    // For line P0->P2 with P1=(P0+P2)/2, this is P0*(1-t) + P2*t
    return c0 + c1 * t + c2 * t * t;
}

vec3 tangent_on_quadratic(float t, vec3 c1, vec3 c2){
    // For line P0->P2 with P1=(P0+P2)/2, this is constant P2-P0
    vec3 tangent = c1 + 2.0 * c2 * t;
    if (length(tangent) < 1e-6) {
         tangent = vec3(1.0, 0.0, 0.0); // Fallback
    }
    return normalize(tangent);
}

// Calculates the offset direction for stroke vertices, handling joints
vec3 step_to_corner(vec3 point, vec3 tangent, float joint_angle, bool inside_curve, bool draw_flat, int joint_type_int) {
    vec3 step_perp;
    // Calculate base perpendicular direction
    if (draw_flat) {
        step_perp = normalize(vec3(-tangent.y, tangent.x, 0.0));
        if (length(step_perp) < 1e-6) step_perp = vec3(0.0, 1.0, 0.0); // Fallback
    } else {
        vec3 view_vec = normalize(camera_position - point);
        step_perp = cross(tangent, view_vec);
         if (length(step_perp) < 1e-6) {
             step_perp = normalize(vec3(-tangent.y, tangent.x, 0.0)); // Fallback
             if (length(step_perp) < 1e-6) step_perp = vec3(0.0, 1.0, 0.0);
         }
         step_perp = normalize(step_perp);
    }

    // If no joint handling needed, return perpendicular direction
    if (inside_curve || joint_type_int == NO_JOINT || joint_angle == 0.0) {
        return step_perp;
    }

    // Calculate miter/bevel adjustment
    float cos_angle = cos(joint_angle);
    float sin_angle = sin(joint_angle);

    // Avoid division by zero or extreme cases if angle is near 0 or PI
    if (abs(cos_angle) > COS_THRESHOLD || abs(sin_angle) < 1e-6) return step_perp;

    float miter_factor;
    if (joint_type_int == BEVEL_JOINT){
        miter_factor = 0.0; // Force bevel
    } else if (joint_type_int == MITER_JOINT){
        miter_factor = 1.0; // Force miter
    } else { // AUTO_JOINT
        // Smoothly transition from miter to bevel for sharp angles
        float mcat1 = MITER_COS_ANGLE_THRESHOLD; // e.g., -0.8 (around 143 deg)
        float mcat2 = mix(mcat1, -1.0, 0.5); // A slightly sharper angle threshold
        miter_factor = smoothstep(mcat1, mcat2, cos_angle); // 1.0 for wide angles, 0.0 for sharp
    }

    // Calculate shift along the tangent based on angle and miter factor
    // Mix adjusts the effective center of the miter calculation
    float shift = (cos_angle + mix(-1.0, 1.0, miter_factor)) / sin_angle;

    // Return combined direction: perpendicular + tangent shift
    return normalize(step_perp + shift * tangent);
}

// Emits two vertices for the triangle strip at a given point
void emit_point_with_width(
    vec3 point, vec3 tangent, float joint_angle,
    float width, vec4 point_color,
    bool inside_curve, bool draw_flat, int joint_type_int
) {
    vec3 step_dir = step_to_corner(point, tangent, joint_angle, inside_curve, draw_flat, joint_type_int);
    float aaw = max(anti_alias_width * pixel_size, 1e-8);
    float hw = 0.5 * width;

    color = point_color;
    half_width_to_aaw = hw / aaw;

    // Emit vertices for both sides of the stroke
    for (int side = -1; side <= 1; side += 2) {
        // Distance from center, including AA margin
        float dist_from_center = side * 0.5 * (width + aaw);
        // Calculate final vertex position
        vec4 clip_pos = vec4(point + dist_from_center * step_dir, 1.0);
        // Assuming no MVP matrix used here for simplicity (outputting directly)
        gl_Position = clip_pos;
        // Pass distance data to FS for AA calculation
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

    vec3 P0 = verts[0];
    vec3 P1 = verts[1]; // Control point
    vec3 P2 = verts[2];

    // Calculate quadratic bezier coefficients (degenerates for P1=(P0+P2)/2)
    vec3 c0 = P0;
    vec3 c1 = 2.0 * (P1 - P0);
    vec3 c2 = P0 - 2.0 * P1 + P2;

    // For straight lines, only need start (t=0) and end (t=1) points
    int n_steps = MAX_STEPS; // Should be 2

    for (int i = 0; i < n_steps; i++){
        float t = float(i) / float(n_steps - 1); // t=0 or t=1

        vec3 point = point_on_quadratic(t, c0, c1, c2);
        vec3 tangent = tangent_on_quadratic(t, c1, c2);

        // Interpolate width and color (will just be start/end values for t=0/1)
        float current_stroke_width = mix(v_stroke_width[0], v_stroke_width[2], t);
        vec4 current_color = mix(v_color[0], v_color[2], t);

        // Determine the correct joint angle for this endpoint
        // GS expects the angle of the turn AT the vertex.
        // VS provides the angle associated with P0 and P2 vertex indices.
        float current_joint_angle;
        if (i == 0){ // Start point (P0)
            // Use the angle data stored at P0's index in VBO, negate for step_to_corner convention?
            // Let's re-check step_to_corner: positive angle -> CCW turn?
            // If VBO stores CCW turn angle directly, use -angle[0] for start, +angle[2] for end.
            current_joint_angle = -v_joint_angle[0];
        } else { // End point (P2)
            current_joint_angle = v_joint_angle[2];
        }

        // Emit the pair of vertices for this point on the stroke
        emit_point_with_width(
            point, tangent, current_joint_angle,
            current_stroke_width, current_color,
            false, // inside_curve = false for lines
            draw_flat, joint_type_int
        );
    }
    EndPrimitive(); // Finish the triangle strip for this segment
}
"""

# 片段着色器 (来自第一个代码块 - 处理抗锯齿)
FRAGMENT_SHADER_SRC = """
#version 330 core
in vec4 color;
in float dist_to_aaw;      // Signed distance / AAW
in float half_width_to_aaw; // Half width / AAW

out vec4 frag_color;

void main() {
    frag_color = color;
    // Calculate signed distance to the stroke edge region
    float signed_dist_to_region = abs(dist_to_aaw) - half_width_to_aaw;
    // Apply anti-aliasing using smoothstep
    frag_color.a *= smoothstep(0.5, -0.5, signed_dist_to_region);
    // Discard fully transparent fragments
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


class SquareWidget(QOpenGLWidget): # Renamed widget for clarity
    def __init__(self, parent=None):
        super().__init__(parent)
        self.program = None
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        self.uniform_locs = {}

    # normalize_vec 函数保持不变
    def normalize_vec(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: return v
        return v / norm

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_DEPTH_TEST) # Enable if needed

        self.program = self.createShaderProgram(VERTEX_SHADER_SRC, GEOMETRY_SHADER_SRC, FRAGMENT_SHADER_SRC)
        glUseProgram(self.program)

        # --- 定义正方形顶点 (居中) ---
        scale = 0.8 # Scale factor for the square size
        s_half = 0.5 * scale # Half side length
        V0 = np.array([-s_half,  s_half, 0.0], dtype=np.float32) # Top-left
        V1 = np.array([ s_half,  s_half, 0.0], dtype=np.float32) # Top-right
        V2 = np.array([ s_half, -s_half, 0.0], dtype=np.float32) # Bottom-right
        V3 = np.array([-s_half, -s_half, 0.0], dtype=np.float32) # Bottom-left

        # --- 定义直线段控制点 (P0, P1=(P0+P2)/2, P2) ---
        # Segment 1: V0 -> V1 (Top)
        P0_0, P2_0 = V0, V1; P1_0 = (P0_0 + P2_0) / 2.0
        # Segment 2: V1 -> V2 (Right)
        P0_1, P2_1 = V1, V2; P1_1 = (P0_1 + P2_1) / 2.0
        # Segment 3: V2 -> V3 (Bottom)
        P0_2, P2_2 = V2, V3; P1_2 = (P0_2 + P2_2) / 2.0
        # Segment 4: V3 -> V0 (Left)
        P0_3, P2_3 = V3, V0; P1_3 = (P0_3 + P2_3) / 2.0

        points = np.array([
            P0_0, P1_0, P2_0, # Seg 1
            P0_1, P1_1, P2_1, # Seg 2
            P0_2, P1_2, P2_2, # Seg 3
            P0_3, P1_3, P2_3  # Seg 4
        ])
        self.vertex_count = 12 # 4 segments * 3 points

        # --- 计算 Joint Angles ---
        # 对于正方形，每个角是 90 度 (PI/2) 的转弯。
        # 顶点顺序 V0->V1->V2->V3->V0 是逆时针 (CCW)。
        # 因此，每个角点的转角是 +PI/2 弧度。
        angle = math.pi / 2.0 # 90 degrees in radians

        # VBO 数据结构: 为每个 P0 和 P2 存储其所在顶点的转角
        # GS 使用 -angle[0] 作为起点转角, angle[2] 作为终点转角
        # 例如 Seg 1 (V0->V1): angle at V0 is +PI/2, angle at V1 is +PI/2
        # VBO layout for Seg 1: [angle_V0, 0.0, angle_V1] -> [PI/2, 0, PI/2]
        joint_angles = np.array([
            [angle], [0.0], [angle], # Seg 1 (V0 angle, P1, V1 angle)
            [angle], [0.0], [angle], # Seg 2 (V1 angle, P1, V2 angle)
            [angle], [0.0], [angle], # Seg 3 (V2 angle, P1, V3 angle)
            [angle], [0.0], [angle]  # Seg 4 (V3 angle, P1, V0 angle)
        ], dtype=np.float32)
        print(f"Joint Angle (Radians): {angle}")
        print(f"Joint Angle (Degrees): {math.degrees(angle)}")

        # --- 其他顶点数据 ---
        stroke_rgbas = np.array([
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], # Seg 1 Red
            [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], # Seg 2 Green
            [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], # Seg 3 Blue
            [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]  # Seg 4 Yellow
        ], dtype=np.float32)

        stroke_widths = np.array([[10.0]] * 12, dtype=np.float32) # Example width 10.0

        unit_normals = np.array([[0.0, 0.0, 1.0]] * 12, dtype=np.float32) # Normal pointing up Z

        # --- VBO/VAO Setup (包含 angle 和 normal) ---
        # Data: pos(3) + rgba(4) + width(1) + angle(1) + normal(3) = 12 floats
        data = np.hstack([points, stroke_rgbas, stroke_widths, joint_angles, unit_normals]).astype(np.float32)
        stride = data.itemsize * (3 + 4 + 1 + 1 + 3) # 48 bytes

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        offset = 0
        # Location 0: point (vec3)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(0); offset += points.itemsize * 3
        # Location 1: stroke_rgba (vec4)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(1); offset += stroke_rgbas.itemsize * 4
        # Location 2: stroke_width (float)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(2); offset += stroke_widths.itemsize * 1
        # Location 3: joint_angle (float) - RE-ENABLED
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(3); offset += joint_angles.itemsize * 1
        # Location 4: unit_normal (vec3) - RE-ENABLED
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(4); # offset += unit_normals.itemsize * 3

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # Get uniform locations (including joint_type)
        self.uniform_locs = {name: glGetUniformLocation(self.program, name) for name in [
            "frame_scale", "is_fixed_in_frame", "scale_stroke_with_zoom", # VS
            "anti_alias_width", "flat_stroke", "pixel_size", "joint_type", "camera_position" # GS
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
        # --- Set flat_stroke = 1.0 for 2D rendering ---
        glUniform1f(self.uniform_locs.get("flat_stroke", -1), 1.0)
        # ---------------------------------------------
        glUniform1f(self.uniform_locs.get("pixel_size", -1), pixel_size)
        # --- Set Joint Type ---
        glUniform1f(self.uniform_locs.get("joint_type", -1), float(MITER_JOINT)) # Use Miter for sharp 90-degree corners
        # glUniform1f(self.uniform_locs.get("joint_type", -1), float(BEVEL_JOINT)) # Use Bevel for cut-off corners
        # glUniform1f(self.uniform_locs.get("joint_type", -1), float(AUTO_JOINT)) # Should act like Miter for 90 deg
        # --------------------
        glUniform3f(self.uniform_locs.get("camera_position", -1), 0.0, 0.0, 3.0) # Simple camera Z=3

        # Draw the 4 segments forming the square
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count) # Draw all 12 vertices (4 segments)
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        if self.program and self.uniform_locs.get("pixel_size", -1) != -1:
             glUseProgram(self.program)
             pixel_size = 2.0 / max(h, 1) if h > 0 else 0.002
             glUniform1f(self.uniform_locs["pixel_size"], pixel_size)

    # createShaderProgram function remains the same
    def createShaderProgram(self, vert_src, geom_src, frag_src):
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
        self.setWindowTitle("OpenGL Square Stroke with Joints") # Updated title
        self.setGeometry(100, 100, 800, 600)
        self.setCentralWidget(SquareWidget()) # Use the renamed widget


if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSamples(4) # Anti-aliasing
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())