import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import *
import ctypes
import math

# test：使用 Manimgl 的 Stroke 逻辑渲染三条 BezierCurve，形成封闭三角形

# 顶点着色器
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

# 几何着色器
GEOMETRY_SHADER_SRC = """
#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 64) out;

uniform float anti_alias_width;
uniform float flat_stroke;
uniform float pixel_size;
uniform float joint_type; // int as float
uniform float frame_scale;
uniform vec3 camera_position;
uniform float is_fixed_in_frame;

in vec3 verts[3]; // P0, P1, P2
in float v_joint_angle[3];
in float v_stroke_width[3];
in vec4 v_color[3];
in vec3 v_unit_normal[3]; // 表面法线

out vec4 color;
out float dist_to_aaw;
out float half_width_to_aaw;

// Joint types
const int NO_JOINT = 0;
const int AUTO_JOINT = 1;
const int BEVEL_JOINT = 2;
const int MITER_JOINT = 3;

// Constants
const float COS_THRESHOLD = 0.999; // Near 1 or -1
const float POLYLINE_FACTOR = 100; // 对直线影响不大，但保留
const int MAX_STEPS = 2;         // 直线只需要起点和终点
const float MITER_COS_ANGLE_THRESHOLD = -0.8;

// --- Helper Functions ---
vec3 point_on_quadratic(float t, vec3 c0, vec3 c1, vec3 c2){
    // 对直线 P0->P2, 当 P1=(P0+P2)/2 时, 结果等同于线性插值 P0*(1-t) + P2*t
    return c0 + c1 * t + c2 * t * t;
}

vec3 tangent_on_quadratic(float t, vec3 c1, vec3 c2){
    // 对直线 P0->P2, c1 = P2-P0, c2 = 0, 导数 B'(t) = c1 = P2-P0 (常数向量)
    vec3 tangent = c1 + 2.0 * c2 * t;
    if (length(tangent) < 1e-6) {
        // 如果退化 (P0=P1=P2), 提供备用切线
         tangent = vec3(1.0, 0.0, 0.0);
    }
    return normalize(tangent);
}

vec3 step_to_corner(vec3 point, vec3 tangent, float joint_angle, bool inside_curve, bool draw_flat, int joint_type_int) {
    vec3 step_perp;
    if (draw_flat) {
        step_perp = normalize(vec3(-tangent.y, tangent.x, 0.0));
    } else {
        vec3 view_vec = normalize(camera_position - point);
        step_perp = normalize(cross(tangent, view_vec));
    }

    if (inside_curve || joint_type_int == NO_JOINT || joint_angle == 0.0) {
        return step_perp;
    }

    float cos_angle = cos(joint_angle);
    float sin_angle = sin(joint_angle);
    if (abs(cos_angle) > COS_THRESHOLD) return step_perp;
    if (abs(sin_angle) < 1e-6) return step_perp;

    float miter_factor;
    if (joint_type_int == BEVEL_JOINT){
        miter_factor = 0.0;
    } else if (joint_type_int == MITER_JOINT){
        miter_factor = 1.0;
    } else { // AUTO_JOINT
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
    if (verts[0] == verts[2]) return; // 忽略零长度直线 (P0 == P2)
    if (vec3(v_stroke_width[0], v_stroke_width[1], v_stroke_width[2]) == vec3(0.0)) return;
    if (vec3(v_color[0].a, v_color[1].a, v_color[2].a) == vec3(0.0)) return;

    bool draw_flat = bool(flat_stroke) || bool(is_fixed_in_frame);
    int joint_type_int = int(joint_type);

    vec3 P0 = verts[0];
    vec3 P1 = verts[1]; // 控制点 P1, 对于直线 P1=(P0+P2)/2
    vec3 P2 = verts[2];

    vec3 c0 = P0;
    vec3 c1 = 2.0 * (P1 - P0); // 对直线 c1 = P2 - P0
    vec3 c2 = P0 - 2.0 * P1 + P2; // 对直线 c2 = 0

    // 直线不需要很多步，只需起点和终点
    int n_steps = MAX_STEPS; // 使用 MAX_STEPS 常量 (设为 2)

    for (int i = 0; i < n_steps; i++){
        // if (i >= n_steps) break; // 不需要这行，因为 n_steps=2
        float t = float(i) / float(n_steps - 1); // t = 0 或 1

        vec3 point = point_on_quadratic(t, c0, c1, c2); // 计算起点(t=0)或终点(t=1)
        vec3 tangent = tangent_on_quadratic(t, c1, c2); // 切线是常数 (P2-P0)

        float current_stroke_width = mix(v_stroke_width[0], v_stroke_width[2], t); // 使用起点或终点宽度
        vec4 current_color = mix(v_color[0], v_color[2], t); // 使用起点或终点颜色

        bool inside_curve = false; // 直线没有内部点
        float current_joint_angle;
        if (i == 0){ // 起点
            current_joint_angle = -v_joint_angle[0]; // 使用 VBO 中索引 0 的角度
        } else { // 终点 (i == 1)
            current_joint_angle = v_joint_angle[2]; // 使用 VBO 中索引 2 的角度
        }

        emit_point_with_width(
            point, tangent, current_joint_angle,
            current_stroke_width, current_color,
            inside_curve, draw_flat, joint_type_int
        );
    }
    EndPrimitive();
}
"""

# 片段着色器
FRAGMENT_SHADER_SRC = """
#version 330 core
in vec4 color;
in float dist_to_aaw;      // Signed distance / AAW
in float half_width_to_aaw; // Half width / AAW

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


class TriangleWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.program = None
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        self.uniform_locs = {}

    def normalize_vec(self, v):
        """简单的 NumPy 向量归一化"""
        norm = np.linalg.norm(v)
        if norm == 0:
           return v
        return v / norm

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.program = self.createShaderProgram(VERTEX_SHADER_SRC, GEOMETRY_SHADER_SRC, FRAGMENT_SHADER_SRC)
        glUseProgram(self.program)

        # --- 定义等边三角形顶点 ---
        scale = 0.8 # 缩放因子，使三角形适应窗口
        s = 1.0 * scale # 边长
        h = s * math.sqrt(3.0) / 2.0 # 高度
        y_top = h / 2.0
        y_bottom = -h / 2.0
        x_side = s / 2.0

        V0 = np.array([0.0,    y_top, 0.0], dtype=np.float32)
        V1 = np.array([-x_side, y_bottom, 0.0], dtype=np.float32)
        V2 = np.array([ x_side, y_bottom, 0.0], dtype=np.float32)

        # --- 定义三条直线段的控制点 (P0, P1=(P0+P2)/2, P2) ---
        P0_0, P2_0 = V0, V1
        P1_0 = (P0_0 + P2_0) / 2.0
        P0_1, P2_1 = V1, V2
        P1_1 = (P0_1 + P2_1) / 2.0
        P0_2, P2_2 = V2, V0
        P1_2 = (P0_2 + P2_2) / 2.0

        points = np.array([
            P0_0, P1_0, P2_0, # Segment 1 (V0 -> V1)
            P0_1, P1_1, P2_1, # Segment 2 (V1 -> V2)
            P0_2, P1_2, P2_2  # Segment 3 (V2 -> V0)
        ])

        # --- 计算 Joint Angles ---
        # 对于等边三角形，每个顶点的转向角是 120 度 (2*pi/3)
        # 需要确定符号。从 V0->V1->V2->V0 是逆时针。
        # 在 V1: T1_end (V1-V0), T2_start (V2-V1). cross(T1, T2).z > 0? -> angle = 2pi/3
        # 在 V2: T2_end (V2-V1), T3_start (V0-V2). cross(T2, T3).z > 0? -> angle = 2pi/3
        # 在 V0: T3_end (V0-V2), T1_start (V1-V0). cross(T3, T1).z > 0? -> angle = 2pi/3
        # 假设逆时针旋转为正角度
        angle = 2.0 * math.pi / 3.0
        # angle_V0 = angle
        # angle_V1 = angle
        # angle_V2 = angle
        # 按照 GS 的使用方式 (-angle_start, 0, angle_end)
        # Seg 1 (0,1,2): V0(start), V1(end) -> Needs -angle_V0, angle_V1
        # Seg 2 (3,4,5): V1(start), V2(end) -> Needs -angle_V1, angle_V2
        # Seg 3 (6,7,8): V2(start), V0(end) -> Needs -angle_V2, angle_V0
        joint_angles = np.array([
            [-angle], [0.0], [angle], # Seg 1 angles at V0, (P1), V1
            [-angle], [0.0], [angle], # Seg 2 angles at V1, (P1), V2
            [-angle], [0.0], [angle]  # Seg 3 angles at V2, (P1), V0
        ], dtype=np.float32)
        print(f"Joint Angle (Radians): {angle}")
        print(f"Joint Angle (Degrees): {math.degrees(angle)}")

        # --- 其他顶点数据 ---
        stroke_rgbas = np.array([
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], # Seg 1 Red
            [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], # Seg 2 Green
            [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]  # Seg 3 Blue
        ], dtype=np.float32)

        stroke_widths = np.array([[5.0]] * 9, dtype=np.float32) # Constant width 5.0

        unit_normals = np.array([[0.0, 0.0, 1.0]] * 9, dtype=np.float32) # All normals point up Z

        # --- VBO/VAO Setup ---
        self.vertex_count = 9 # 3 segments * 3 points
        data = np.hstack([points, stroke_rgbas, stroke_widths, joint_angles, unit_normals]).astype(np.float32)
        stride = data.itemsize * (3 + 4 + 1 + 1 + 3) # 48 bytes

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        offset = 0
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(0); offset += points.itemsize * 3
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(1); offset += stroke_rgbas.itemsize * 4
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(2); offset += stroke_widths.itemsize * 1
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(3); offset += joint_angles.itemsize * 1
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)); glEnableVertexAttribArray(4);

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
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        width, height = self.width(), self.height()
        pixel_size = 2.0 / max(height, 1) if height > 0 else 0.002

        glUniform1f(self.uniform_locs.get("frame_scale", -1), 1.0)
        glUniform1f(self.uniform_locs.get("is_fixed_in_frame", -1), 0.0)
        glUniform1f(self.uniform_locs.get("scale_stroke_with_zoom", -1), 1.0)
        glUniform1f(self.uniform_locs.get("anti_alias_width", -1), 1.5)
        # --- 设置为 flat rendering ---
        glUniform1f(self.uniform_locs.get("flat_stroke", -1), 1.0) # True
        # --------------------------
        glUniform1f(self.uniform_locs.get("pixel_size", -1), pixel_size)
        # --- 选择接头类型 ---
        glUniform1f(self.uniform_locs.get("joint_type", -1), float(MITER_JOINT)) # Miter 适合尖角
        # glUniform1f(self.uniform_locs.get("joint_type", -1), float(BEVEL_JOINT)) # Bevel 会切掉角
        # glUniform1f(self.uniform_locs.get("joint_type", -1), float(AUTO_JOINT))
        # --------------------
        glUniform3f(self.uniform_locs.get("camera_position", -1), 0.0, 0.0, 3.0)

        # 绘制三条直线段
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)  # 绘制 Segment 1 (V0 -> V1)
        glDrawArrays(GL_TRIANGLES, 3, 3)  # 绘制 Segment 2 (V1 -> V2)
        glDrawArrays(GL_TRIANGLES, 6, 3)  # 绘制 Segment 3 (V2 -> V0)
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        if self.program and self.uniform_locs.get("pixel_size", -1) != -1:
             glUseProgram(self.program)
             pixel_size = 2.0 / max(h, 1) if h > 0 else 0.002
             glUniform1f(self.uniform_locs["pixel_size"], pixel_size)

    def createShaderProgram(self, vert_src, geom_src, frag_src):
        # (着色器编译链接函数 - 保持不变)
        def compile_shader(src, shader_type):
            shader = glCreateShader(shader_type)
            glShaderSource(shader, src)
            glCompileShader(shader)
            if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
                info_log = glGetShaderInfoLog(shader)
                if isinstance(info_log, bytes): info_log = info_log.decode()
                raise RuntimeError(f"Shader compile fail {shader_type}:\n{info_log}")
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
            raise RuntimeError(f"Program link fail:\n{info_log}")
        for s in [vs, gs, fs]: glDetachShader(program, s); glDeleteShader(s)
        return program

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL Equilateral Triangle Stroke") # 更新标题
        self.setGeometry(100, 100, 800, 600)
        # 使用新的 Widget 类
        self.setCentralWidget(TriangleWidget())


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