import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import *
import ctypes
import math # 用于计算角度

# test：使用 Manimgl 的 Stroke 逻辑渲染两条贝塞尔曲线（自定义控制点）

# 顶点着色器源码 (移除 gl_Position)
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

const float STROKE_WIDTH_CONVERSION = 0.01; // 可根据需要调整

void main(){
    verts = point;
    v_color = stroke_rgba;
    v_stroke_width = STROKE_WIDTH_CONVERSION * stroke_width * mix(frame_scale, 1.0, scale_stroke_with_zoom);
    v_joint_angle = joint_angle;
    v_unit_normal = unit_normal;
    // gl_Position 不应在这里设置，由 GS 处理最终顶点位置
}
"""

# 几何着色器源码 (修正和恢复逻辑)
GEOMETRY_SHADER_SRC = """
#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 64) out;

uniform float anti_alias_width;
uniform float flat_stroke;
uniform float pixel_size;
uniform float joint_type; // int as float
uniform float frame_scale;
uniform vec3 camera_position; // 即使 flat 也可能需要 (虽然这个版本简化了)
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
const float POLYLINE_FACTOR = 100;
const int MAX_STEPS = 32;
const float MITER_COS_ANGLE_THRESHOLD = -0.8; // For AUTO_JOINT miter limit

// --- Helper Functions ---
vec3 point_on_quadratic(float t, vec3 c0, vec3 c1, vec3 c2){
    // 使用系数计算: B(t) = c0 + c1*t + c2*t^2
    return c0 + c1 * t + c2 * t * t;
}

vec3 tangent_on_quadratic(float t, vec3 c1, vec3 c2){
    // 导数 B'(t) = c1 + 2 * c2 * t, 然后归一化
    // 处理 t=0 或 t=1 时可能出现的零向量切线 (如果 P0=P1 或 P1=P2)
    vec3 tangent = c1 + 2.0 * c2 * t;
    if (length(tangent) < 1e-6) {
        // 如果切线是零向量，尝试使用控制点方向作为备用
        if (t < 0.5) tangent = verts[1] - verts[0]; // P1-P0
        else tangent = verts[2] - verts[1]; // P2-P1
        if (length(tangent) < 1e-6) tangent = vec3(1.0, 0.0, 0.0); // 最终备用
    }
    return normalize(tangent);
}

vec3 step_to_corner(vec3 point, vec3 tangent, float joint_angle, bool inside_curve, bool draw_flat, int joint_type_int) {
    // 计算基础的、垂直于切线的步进方向
    vec3 step_perp;
    if (draw_flat) {
        // 2D 情况: 直接计算 XY 平面内的垂直向量
        step_perp = normalize(vec3(-tangent.y, tangent.x, 0.0));
    } else {
        // 3D 情况: 需要视图向量来确定步进方向 (简化: 假设 Z 轴为视图方向)
        vec3 view_vec = vec3(0.0, 0.0, 1.0); // normalize(camera_position - point);
        step_perp = normalize(cross(tangent, view_vec));
    }

    // 如果在曲线内部、无接头或角度为零，则直接使用垂直方向
    if (inside_curve || joint_type_int == NO_JOINT || joint_angle == 0.0) {
        return step_perp;
    }

    // --- 接头处理 (简化版 Miter/Bevel) ---
    float cos_angle = cos(joint_angle);
    float sin_angle = sin(joint_angle);

    // 角度太小，无需处理
    if (abs(cos_angle) > COS_THRESHOLD) return step_perp;
    // 防止除零
    if (abs(sin_angle) < 1e-6) return step_perp;

    float miter_factor;
    if (joint_type_int == BEVEL_JOINT){
        miter_factor = 0.0;
    } else if (joint_type_int == MITER_JOINT){
        miter_factor = 1.0;
    } else { // AUTO_JOINT
        float mcat1 = MITER_COS_ANGLE_THRESHOLD;
        float mcat2 = mix(mcat1, -1.0, 0.5); // 平滑过渡区域
        miter_factor = smoothstep(mcat1, mcat2, cos_angle);
    }

    // 计算沿切线方向的偏移量
    float shift = (cos_angle + mix(-1.0, 1.0, miter_factor)) / sin_angle;
    // 最终步进方向 = 垂直方向 + 切线方向偏移 * 切线
    // 需要归一化，否则 miter 接头会过长
    return normalize(step_perp + shift * tangent);
}


void emit_point_with_width(
    vec3 point, vec3 tangent, float joint_angle,
    float width, vec4 point_color,
    bool inside_curve, bool draw_flat, int joint_type_int
) {
    // 计算到描边边缘的步进方向 (包含接头处理)
    vec3 step_dir = step_to_corner(point, tangent, joint_angle, inside_curve, draw_flat, joint_type_int);

    // 抗锯齿宽度 (AAW)
    float aaw = max(anti_alias_width * pixel_size, 1e-8);
    float hw = 0.5 * width; // 半描边宽度

    // 设置颜色
    color = point_color; // 使用插值后的颜色
    // 设置用于抗锯齿的变量 (恢复 smoothstep 兼容格式)
    half_width_to_aaw = hw / aaw;

    // 发射描边两侧的顶点
    for (int side = -1; side <= 1; side += 2) { // side = -1, +1
        float dist_from_center = side * 0.5 * (width + aaw);
        gl_Position = vec4(point + dist_from_center * step_dir, 1.0);
        dist_to_aaw = dist_from_center / aaw; // 带符号距离 / AAW
        EmitVertex();
    }
}

// --- GS Main ---
void main() {
    // 基础检查
    if (verts[0] == verts[1]) return; // 忽略退化曲线
    if (vec3(v_stroke_width[0], v_stroke_width[1], v_stroke_width[2]) == vec3(0.0)) return; // 零宽度
    if (vec3(v_color[0].a, v_color[1].a, v_color[2].a) == vec3(0.0)) return; // 全透明

    bool draw_flat = bool(flat_stroke) || bool(is_fixed_in_frame);
    int joint_type_int = int(joint_type);

    // 控制点 P0, P1, P2
    vec3 P0 = verts[0];
    vec3 P1 = verts[1];
    vec3 P2 = verts[2];

    // 计算贝塞尔系数 c0, c1, c2
    vec3 c0 = P0;
    vec3 c1 = 2.0 * (P1 - P0);
    vec3 c2 = P0 - 2.0 * P1 + P2;

    // 估算细分步数
    float area = 0.5 * length(cross(P1 - P0, P2 - P0));
    int count = int(round(POLYLINE_FACTOR * sqrt(area) / frame_scale));
    int n_steps = min(2 + count, MAX_STEPS);

    // 沿曲线生成顶点
    for (int i = 0; i < MAX_STEPS; i++){
        if (i >= n_steps) break;
        float t = float(i) / float(n_steps - 1); // 参数 t [0, 1]

        // 计算点和切线
        vec3 point = point_on_quadratic(t, c0, c1, c2);
        vec3 tangent = tangent_on_quadratic(t, c1, c2);

        // 插值计算宽度和颜色 (在 P0 和 P2 之间插值)
        float current_stroke_width = mix(v_stroke_width[0], v_stroke_width[2], t);
        vec4 current_color = mix(v_color[0], v_color[2], t);

        // 确定 joint_angle (仅在端点使用传入值)
        bool inside_curve = (i > 0 && i < n_steps - 1);
        float current_joint_angle;
        if (i == 0){
            current_joint_angle = -v_joint_angle[0]; // GS 使用 VBO 中索引 0 的角度
        } else if (inside_curve){
            current_joint_angle = 0.0;
        } else { // i == n_steps - 1
            current_joint_angle = v_joint_angle[2]; // GS 使用 VBO 中索引 2 的角度
        }

        // 发射顶点对
        emit_point_with_width(
            point, tangent, current_joint_angle,
            current_stroke_width, current_color,
            inside_curve, draw_flat, joint_type_int
        );
    }
    EndPrimitive(); // 结束当前曲线的 triangle_strip
}
"""

# 片段着色器源码 (恢复 smoothstep 抗锯齿)
FRAGMENT_SHADER_SRC = """
#version 330 core
in vec4 color;
in float dist_to_aaw;      // Signed distance / AAW
in float half_width_to_aaw; // Half width / AAW

out vec4 frag_color;

void main() {
    frag_color = color;

    // 计算符号距离场值 (SDF)
    // 内部为负, 边缘为0, 外部为正
    float signed_dist_to_region = abs(dist_to_aaw) - half_width_to_aaw;

    // 使用 smoothstep 实现抗锯齿
    // 当 signed_dist_to_region 从 -0.5 变化到 0.5 时, alpha 从 1 平滑过渡到 0
    frag_color.a *= smoothstep(0.5, -0.5, signed_dist_to_region);

    // 丢弃完全透明的片段
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


class TwoBezierWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.program = None
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        self.uniform_locs = {}

    def compute_tangent(self, P0, P1, P2, t):
        """Helper function to compute tangent vector in Python"""
        c1 = 2.0 * (P1 - P0)
        c2 = P0 - 2.0 * P1 + P2
        tangent = c1 + 2.0 * c2 * t
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            # Handle degenerate cases (similar to shader)
            if t < 0.5: tangent = P1 - P0
            else: tangent = P2 - P1
            norm = np.linalg.norm(tangent)
            if norm < 1e-6: return np.array([1.0, 0.0, 0.0], dtype=np.float32) # Fallback
        return tangent / norm

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.program = self.createShaderProgram(VERTEX_SHADER_SRC, GEOMETRY_SHADER_SRC, FRAGMENT_SHADER_SRC)
        glUseProgram(self.program)

        # --- 定义控制点 ---
        P0_0 = np.array([-0.8, -0.5, 0.0], dtype=np.float32) # Curve 1 Start
        P1_0 = np.array([-0.4,  0.5, 0.0], dtype=np.float32) # Curve 1 Control
        P2_0 = np.array([ 0.0, -0.5, 0.0], dtype=np.float32) # Curve 1 End == Curve 2 Start
        P0_1 = P2_0                                         # Curve 2 Start
        P1_1 = np.array([ 0.4,  0.5, 0.0], dtype=np.float32) # Curve 2 Control
        P2_1 = np.array([ 0.8, -0.5, 0.0], dtype=np.float32) # Curve 2 End

        points = np.array([ P0_0, P1_0, P2_0, P0_1, P1_1, P2_1 ])

        # --- 计算连接点 (P2_0 / P0_1) 的 Joint Angle ---
        tangent1_end = self.compute_tangent(P0_0, P1_0, P2_0, 1.0)
        tangent2_start = self.compute_tangent(P0_1, P1_1, P2_1, 0.0)

        # 计算点积和角度
        dot_product = np.dot(tangent1_end, tangent2_start)
        dot_product = np.clip(dot_product, -1.0, 1.0) # Clamp for numerical stability
        angle = np.arccos(dot_product)

        # 判断角度符号 (使用叉积的 Z 分量)
        # cross_product = np.cross(tangent1_end, tangent2_start)
        # if cross_product[2] < 0: # 根据右手定则和坐标系调整符号
        #     angle = -angle
        # 注意：着色器内部对角度的使用方式可能需要调整符号，
        # GS 使用 -v_joint_angle[0] 和 v_joint_angle[2]。
        # 我们需要在 P2_0 (索引2) 和 P0_1 (索引3) 处设置角度。
        # 假设 angle 是从 tangent1 旋转到 tangent2 的角度。
        # GS 在处理 curve 1 时，需要 joint_angle[2] 代表离开 P2_0 的角度。
        # GS 在处理 curve 2 时，需要 -joint_angle[3] 代表进入 P0_1 的角度。
        # 如果 angle 是从 t1 到 t2 的，那么在 P2_0 处的离开角度是 angle, 在 P0_1 处的进入角度是 -angle。
        joint_angle_at_connection = angle
        joint_angle_for_P2_0 = joint_angle_at_connection
        joint_angle_for_P0_1 = -joint_angle_at_connection # Shader 使用 -v_joint_angle[0]

        print(f"Tangent 1 End: {tangent1_end}")
        print(f"Tangent 2 Start: {tangent2_start}")
        print(f"Dot Product: {dot_product}")
        print(f"Calculated Joint Angle (Radians): {joint_angle_at_connection}")
        print(f"Angle (Degrees): {math.degrees(joint_angle_at_connection)}")


        # --- 其他顶点数据 ---
        stroke_rgbas = np.array([
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], # Curve 1 (Red)
            [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]  # Curve 2 (Blue)
        ], dtype=np.float32)

        stroke_widths = np.array([
            [8.0], [8.0], [8.0], # Curve 1 width
            [8.0], [8.0], [8.0]  # Curve 2 width
        ], dtype=np.float32)

        # 设置 Joint Angles (注意索引 2 和 3)
        joint_angles = np.array([
            [0.0],                             # P0_0 start angle (usually 0)
            [0.0],                             # P1_0 (not used for joint)
            [joint_angle_for_P2_0],            # P2_0 end angle
            [joint_angle_for_P0_1],            # P0_1 start angle (note sign for shader)
            [0.0],                             # P1_1 (not used for joint)
            [0.0]                              # P2_1 end angle (usually 0)
        ], dtype=np.float32)

        unit_normals = np.array([
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # --- VBO/VAO Setup ---
        self.vertex_count = 6
        data = np.hstack([points, stroke_rgbas, stroke_widths, joint_angles, unit_normals]).astype(np.float32)
        stride = data.itemsize * (3 + 4 + 1 + 1 + 3) # 48 bytes

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        offset = 0
        # Loc 0: point
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(0)
        offset += points.itemsize * 3
        # Loc 1: stroke_rgba
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(1)
        offset += stroke_rgbas.itemsize * 4
        # Loc 2: stroke_width
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(2)
        offset += stroke_widths.itemsize * 1 # Corrected itemsize usage
        # Loc 3: joint_angle
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(3)
        offset += joint_angles.itemsize * 1 # Corrected itemsize usage
        # Loc 4: unit_normal
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
        # Check for invalid locations (-1)
        for name, loc in self.uniform_locs.items():
            if loc == -1:
                print(f"Warning: Uniform '{name}' not found or inactive.")


    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        # 设置uniform值
        width, height = self.width(), self.height()
        # 更稳健的 pixel_size 计算
        pixel_size = 2.0 / max(height, 1) if height > 0 else 0.002

        glUniform1f(self.uniform_locs.get("frame_scale", -1), 1.0)
        glUniform1f(self.uniform_locs.get("is_fixed_in_frame", -1), 0.0) # False
        glUniform1f(self.uniform_locs.get("scale_stroke_with_zoom", -1), 1.0) # True
        glUniform1f(self.uniform_locs.get("anti_alias_width", -1), 1.5) # 稍微增加抗锯齿宽度
        # --- 设置为 flat rendering ---
        glUniform1f(self.uniform_locs.get("flat_stroke", -1), 1.0) # True
        # --------------------------
        glUniform1f(self.uniform_locs.get("pixel_size", -1), pixel_size)
        # --- 选择接头类型 ---
        glUniform1f(self.uniform_locs.get("joint_type", -1), float(MITER_JOINT)) # 尝试 MITER 接头
        # glUniform1f(self.uniform_locs.get("joint_type", -1), float(BEVEL_JOINT)) # 或尝试 BEVEL 接头
        # glUniform1f(self.uniform_locs.get("joint_type", -1), float(AUTO_JOINT)) # 或使用 AUTO
        # --------------------
        # camera_position 在 flat_stroke=1.0 时理论上不严格需要，但最好设置
        glUniform3f(self.uniform_locs.get("camera_position", -1), 0.0, 0.0, 3.0)

        # 绘制两条曲线 (分开绘制，但 joint_angle 已设置)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)  # 绘制第一条曲线 (顶点 0, 1, 2)
        glDrawArrays(GL_TRIANGLES, 3, 3)  # 绘制第二条曲线 (顶点 3, 4, 5)
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        # 可选: 在这里更新 pixel_size uniform
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
    # (主窗口类 - 保持不变)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Connected Bezier Curves (Flat)") # 更新标题
        self.setGeometry(100, 100, 800, 600)
        self.setCentralWidget(TwoBezierWidget())


if __name__ == "__main__":
    # (主程序入口 - 保持不变)
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSamples(4) # 启用 MSAA
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())