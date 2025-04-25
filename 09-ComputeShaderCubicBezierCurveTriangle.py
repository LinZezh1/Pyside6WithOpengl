import sys
import numpy as np
import math # 需要导入 math 模块进行三角函数计算
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat, QOpenGLContext, QVector2D, QVector4D

# 依赖检查 (保持不变)
try:
    from OpenGL.GL import *
    required_gl_funcs = ['glDispatchCompute', 'GL_COMPUTE_SHADER', 'glBindImageTexture',
                         'glTexStorage2D', 'glMemoryBarrier', 'GL_SHADER_IMAGE_ACCESS_BARRIER_BIT',
                         'glGenVertexArrays', 'glBindVertexArray', 'glGenBuffers', 'glBindBuffer',
                         'glBufferData', 'glVertexAttribPointer', 'glEnableVertexAttribArray',
                         'glUseProgram', 'glActiveTexture', 'glBindTexture', 'glUniform1i',
                         'glDrawElements', 'GL_RGBA8', 'glUniform1f', 'glUniform2f', 'glUniform4f']
    missing_funcs = [func for func in required_gl_funcs if not hasattr(OpenGL.GL, func)]
    if missing_funcs:
         print(f"警告: PyOpenGL 可能缺少必要函数: {', '.join(missing_funcs)}")
         print("这通常表明 OpenGL 驱动程序 (需要 4.3+) 或 PyOpenGL 安装存在问题。")
except ImportError:
    print("错误: 未安装 PyOpenGL。请安装: pip install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print("错误: 未安装 NumPy。请安装: pip install numpy")
    sys.exit(1)

# == Shader Sources ==

# -- 计算着色器 (Compute Shader) --
# 绘制一个抗锯齿的等边三角形轮廓到纹理
COMPUTE_SHADER_SRC = """
#version 430 core

// 输出纹理 (image)
layout (rgba8, binding = 0) uniform writeonly image2D destTex;

// 输入 Uniforms (三角形顶点和样式)
uniform vec2 V1; // 三角形顶点 1
uniform vec2 V2; // 三角形顶点 2
uniform vec2 V3; // 三角形顶点 3

uniform vec2 textureSize;     // 纹理尺寸 (宽度, 高度)
uniform float lineWidth;       // 线条宽度 (以像素为单位)
uniform vec4 curveColor;        // 线条颜色 (复用之前的变量名)
uniform vec4 backgroundColor;   // 背景颜色

// 本地工作组大小
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// 计算点 p 到线段 ab 的最近距离的平方
// (来自 Inigo Quilez - https://iquilezles.org/articles/distfunctions2d/)
float distSqPointLineSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    // 处理 a == b 的情况，避免除以零
    float baba = dot(ba, ba);
    if (baba < 1e-8) return dot(pa, pa); // 如果线段长度接近零，返回点到a的距离平方
    float h = clamp(dot(pa, ba) / baba, 0.0, 1.0);
    return dot(pa - ba * h, pa - ba * h);
}

void main() {
    // 获取像素坐标
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(textureSize);

    // 边界检查
    if (pixelCoords.x >= size.x || pixelCoords.y >= size.y) {
        return;
    }

    // 计算像素中心在归一化纹理坐标系 (0.0 到 1.0) 的位置
    vec2 uv = (vec2(pixelCoords) + 0.5) / textureSize;

    // 计算像素中心到三角形三条边的最短距离
    float d1_sq = distSqPointLineSegment(uv, V1, V2);
    float d2_sq = distSqPointLineSegment(uv, V2, V3);
    float d3_sq = distSqPointLineSegment(uv, V3, V1);

    // 找到到三条边的最小距离
    float minDist = sqrt(min(min(d1_sq, d2_sq), d3_sq));

    // --- 抗锯齿计算 ---
    // 将线条宽度从像素单位转换为归一化坐标单位 (基于纹理高度)
    float halfWidthNorm = (lineWidth * 0.5) / textureSize.y;
    // 抗锯齿过渡带宽度，通常为 1 个像素对应的归一化宽度
    float aa_width = 1.0 / textureSize.y;

    // 使用 smoothstep 实现平滑过渡
    float alpha = 1.0 - smoothstep(halfWidthNorm - aa_width, halfWidthNorm + aa_width, minDist);

    // 根据 alpha 混合线条颜色和背景颜色
    vec4 finalColor = mix(backgroundColor, curveColor, alpha);

    // 写入目标纹理
    imageStore(destTex, pixelCoords, finalColor);
}
"""

# -- 顶点/片段着色器 (Render Shaders) --
# (保持不变)
RENDER_VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""
RENDER_FRAGMENT_SHADER_SRC = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D computeOutputTexture;
void main() {
    FragColor = texture(computeOutputTexture, TexCoord);
}
"""


class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.compute_program = None
        self.render_program = None
        self.texture_id = None
        self.quad_vao = None
        self.quad_vbo = None
        self.quad_ebo = None
        self.texture_uniform_location_render = -1
        self.texture_width = 512
        self.texture_height = 512

        # 计算着色器的 Uniform 位置 (更新为顶点)
        self.v1_loc = -1
        self.v2_loc = -1
        self.v3_loc = -1
        # 保留其他 uniforms
        self.tex_size_loc = -1
        self.line_width_loc = -1
        self.curve_color_loc = -1 # 名字复用，表示线条颜色
        self.bg_color_loc = -1

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)

    def initializeGL(self):
        context = QOpenGLContext.currentContext()
        version_profile = context.format()
        print(f"OpenGL 版本: {version_profile.majorVersion()}.{version_profile.minorVersion()} Profile: {'Core' if version_profile.profile() == QSurfaceFormat.CoreProfile else 'Compatibility'}")

        if version_profile.majorVersion() < 4 or (version_profile.majorVersion() == 4 and version_profile.minorVersion() < 3):
            print("\n错误: 需要 OpenGL 4.3+")
            QApplication.instance().quit()
            return

        glClearColor(0.1, 0.1, 0.1, 1.0)

        print("编译着色器...")
        self.compute_program = self.compileShader(COMPUTE_SHADER_SRC, GL_COMPUTE_SHADER, "Compute")
        self.render_program = self.compileShader(RENDER_VERTEX_SHADER_SRC, GL_VERTEX_SHADER, "Render Vertex",
                                                  RENDER_FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER, "Render Fragment")

        if not self.compute_program or not self.render_program:
             print("着色器编译失败。正在退出。")
             QApplication.instance().quit()
             return

        # 获取渲染着色器的 uniform 位置
        self.texture_uniform_location_render = glGetUniformLocation(self.render_program, "computeOutputTexture")
        if self.texture_uniform_location_render == -1:
            print("警告: 在渲染着色器中未找到 'computeOutputTexture' uniform。")

        # 获取计算着色器的 uniform 位置
        self.v1_loc = glGetUniformLocation(self.compute_program, "V1")
        self.v2_loc = glGetUniformLocation(self.compute_program, "V2")
        self.v3_loc = glGetUniformLocation(self.compute_program, "V3")
        self.tex_size_loc = glGetUniformLocation(self.compute_program, "textureSize")
        self.line_width_loc = glGetUniformLocation(self.compute_program, "lineWidth")
        self.curve_color_loc = glGetUniformLocation(self.compute_program, "curveColor")
        self.bg_color_loc = glGetUniformLocation(self.compute_program, "backgroundColor")

        # 检查 Uniform 获取情况
        uniform_locations = {
            "V1": self.v1_loc, "V2": self.v2_loc, "V3": self.v3_loc,
            "textureSize": self.tex_size_loc, "lineWidth": self.line_width_loc,
            "curveColor": self.curve_color_loc, "backgroundColor": self.bg_color_loc
        }
        for name, loc in uniform_locations.items():
            if loc == -1:
                print(f"警告: 在计算着色器中未找到 '{name}' uniform。")

        print("设置纹理...")
        self.setupTexture()
        print("设置四边形...")
        self.setupQuad()

        print("初始化完成。启动定时器...")
        self.timer.start(16)

    def compileShader(self, src1, type1, name1, src2=None, type2=None, name2=None):
        # (编译逻辑保持不变)
        shader1 = glCreateShader(type1)
        glShaderSource(shader1, src1)
        glCompileShader(shader1)
        if not glGetShaderiv(shader1, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(shader1).decode() if glGetShaderInfoLog(shader1) else "无错误日志"
            print(f"{name1} 着色器编译错误:\n", log)
            glDeleteShader(shader1)
            return None

        program = glCreateProgram()
        glAttachShader(program, shader1)

        shader2 = None
        if src2 and type2 and name2:
            shader2 = glCreateShader(type2)
            glShaderSource(shader2, src2)
            glCompileShader(shader2)
            if not glGetShaderiv(shader2, GL_COMPILE_STATUS):
                log = glGetShaderInfoLog(shader2).decode() if glGetShaderInfoLog(shader2) else "无错误日志"
                print(f"{name2} 着色器编译错误:\n", log)
                glDeleteShader(shader1)
                glDeleteShader(shader2)
                glDeleteProgram(program)
                return None
            glAttachShader(program, shader2)

        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            log = glGetProgramInfoLog(program).decode() if glGetProgramInfoLog(program) else "无错误日志"
            print("着色器程序链接错误:\n", log)
            glDeleteShader(shader1)
            if shader2: glDeleteShader(shader2)
            glDeleteProgram(program)
            return None

        glDetachShader(program, shader1)
        glDeleteShader(shader1)
        if shader2:
             glDetachShader(program, shader2)
             glDeleteShader(shader2)

        print(f"{name1}{' & ' + name2 if name2 else ''} 着色器编译链接成功。")
        return program

    def setupTexture(self):
        # (设置纹理逻辑保持不变)
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, self.texture_width, self.texture_height)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        print(f"纹理已创建 (ID: {self.texture_id}) 尺寸 {self.texture_width}x{self.texture_height}")

    def setupQuad(self):
        # (设置四边形逻辑保持不变)
        quad_vertices = np.array([
            -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0
        ], dtype=np.float32).reshape((4, 4))
        quad_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        self.quad_ebo = glGenBuffers(1)
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quad_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)
        stride = 4 * sizeof(GLfloat)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * sizeof(GLfloat)))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        print("四边形 VAO/VBO/EBO 设置完成。")

    def resizeGL(self, width, height):
        # (保持不变)
        if height == 0: height = 1
        glViewport(0, 0, width, height)

    def paintGL(self):
        if not self.compute_program or not self.render_program or not self.texture_id or not self.quad_vao:
            return # 未初始化完成

        # --- 计算阶段 ---
        glUseProgram(self.compute_program)

        # --- 计算等边三角形顶点 (在 0-1 归一化坐标系) ---
        center_x, center_y = 0.5, 0.5
        radius = 0.4 # 三角形外接圆半径
        v = []
        for i in range(3):
            # 角度: 90度(上), 210度(左下), 330度(右下) -> 对应弧度
            angle_deg = 90.0 - i * 120.0 # 调整起始角度和方向使顶点在常规位置
            angle_rad = math.radians(angle_deg)
            vx = center_x + radius * math.cos(angle_rad)
            vy = center_y + radius * math.sin(angle_rad)
            v.append(QVector2D(vx, vy))
        v1, v2, v3 = v[0], v[1], v[2] # 获取三个顶点

        line_width_px = 3.0 # 线条宽度 (像素)
        line_color = QVector4D(0.9, 0.9, 0.1, 1.0) # 黄色线条
        bg_color = QVector4D(0.1, 0.1, 0.2, 1.0)   # 深蓝紫色背景

        # 设置计算着色器的 Uniform 变量
        if self.v1_loc != -1: glUniform2f(self.v1_loc, v1.x(), v1.y())
        if self.v2_loc != -1: glUniform2f(self.v2_loc, v2.x(), v2.y())
        if self.v3_loc != -1: glUniform2f(self.v3_loc, v3.x(), v3.y())
        if self.tex_size_loc != -1: glUniform2f(self.tex_size_loc, float(self.texture_width), float(self.texture_height))
        if self.line_width_loc != -1: glUniform1f(self.line_width_loc, line_width_px)
        if self.curve_color_loc != -1: glUniform4f(self.curve_color_loc, line_color.x(), line_color.y(), line_color.z(), line_color.w())
        if self.bg_color_loc != -1: glUniform4f(self.bg_color_loc, bg_color.x(), bg_color.y(), bg_color.z(), bg_color.w())

        # 绑定纹理以供写入
        glBindImageTexture(0, self.texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8)

        # 调度计算着色器
        local_size_x = 8
        local_size_y = 8
        num_groups_x = (self.texture_width + local_size_x - 1) // local_size_x
        num_groups_y = (self.texture_height + local_size_y - 1) // local_size_y
        glDispatchCompute(num_groups_x, num_groups_y, 1)

        # 关键屏障 - 确保计算写入完成
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # --- 渲染阶段 ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.render_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        if self.texture_uniform_location_render != -1:
            glUniform1i(self.texture_uniform_location_render, 0)
        glBindVertexArray(self.quad_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    def cleanupGL(self):
        # (清理逻辑保持不变)
        print("清理 GL 资源...")
        if self.context().isValid():
             self.makeCurrent()
             if hasattr(self, 'quad_ebo') and self.quad_ebo: glDeleteBuffers(1, [self.quad_ebo]); self.quad_ebo = None
             if hasattr(self, 'quad_vbo') and self.quad_vbo: glDeleteBuffers(1, [self.quad_vbo]); self.quad_vbo = None
             if hasattr(self, 'quad_vao') and self.quad_vao: glDeleteVertexArrays(1, [self.quad_vao]); self.quad_vao = None
             if hasattr(self, 'texture_id') and self.texture_id: glDeleteTextures(1, [self.texture_id]); self.texture_id = None
             if hasattr(self, 'compute_program') and self.compute_program: glDeleteProgram(self.compute_program); self.compute_program = None
             if hasattr(self, 'render_program') and self.render_program: glDeleteProgram(self.render_program); self.render_program = None
             self.doneCurrent()
             print("清理尝试完成。")
        else:
             print("OpenGL 上下文无效，跳过清理。")


class MainWindow(QMainWindow):
    # (保持不变)
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("PySide6 计算着色器绘制三角形轮廓")
        self.setGeometry(100, 100, 550, 550) # 调整窗口大小
        self.gl_widget = OpenGLWidget()
        self.setCentralWidget(self.gl_widget)

    def closeEvent(self, event):
        print("主窗口关闭...")
        self.gl_widget.cleanupGL()
        super().closeEvent(event)


if __name__ == "__main__":
    # (保持不变)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    gl_format = QSurfaceFormat()
    print("请求 OpenGL 4.3 Core Profile...")
    gl_format.setVersion(4, 3)
    gl_format.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(gl_format)
    app = QApplication(sys.argv)
    print("创建主窗口...")
    window = MainWindow()
    window.show()
    print("启动应用程序事件循环...")
    sys.exit(app.exec())