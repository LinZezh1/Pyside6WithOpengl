import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
# 导入 Qt, QTimer
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat, QOpenGLContext, QVector2D, QVector4D

# test：使用 Computer Shader 实现 Cubic Bezier Curve 的抗锯齿效果
try:
    from OpenGL.GL import *
    # 检查必要的 OpenGL 函数
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

# -- 计算着だし (Compute Shader) --
# 绘制一条抗锯齿的贝塞尔曲线到纹理
# *** 注意：这里是修正了 P3 类型的版本 ***
COMPUTE_SHADER_SRC = """
#version 430 core

// 输出纹理 (image)
layout (rgba8, binding = 0) uniform writeonly image2D destTex;

// 输入 Uniforms (曲线参数)
uniform vec2 P0; // 控制点 0 (开始点)
uniform vec2 P1; // 控制点 1
uniform vec2 P2; // 控制点 2
// uniform vec3 P3; // 控制点 3 (结束点) - 这是错误的声明，注释掉或删除
uniform vec2 P3; // 控制点 3 (结束点) -- 使用这个正确的 vec2 声明

uniform vec2 textureSize; // 纹理尺寸 (宽度, 高度)
uniform float lineWidth;   // 线条宽度 (以像素为单位)
uniform vec4 curveColor;    // 曲线颜色
uniform vec4 backgroundColor; // 背景颜色

// 本地工作组大小
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// 三阶贝塞尔曲线函数
vec2 bezier(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    float mt = 1.0 - t;
    float mt2 = mt * mt;
    float mt3 = mt2 * mt;
    // 现在 P0, P1, P2, P3 都是 vec2，类型匹配，可以正确相加
    return mt3*P0 + 3.0*mt2*t*P1 + 3.0*mt*t2*P2 + t3*P3;
}

// 计算点 p 到线段 ab 的最近距离的平方
float distSqPointLineSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return dot(pa - ba * h, pa - ba * h);
}


// 近似计算点 p 到贝塞尔曲线的距离
float distanceToBezier(vec2 p) {
    int num_samples = 100; // 采样点数量
    float min_dist_sq = 1e20; // 距离平方初始值

    vec2 prev_point = bezier(0.0);
    for(int i = 1; i <= num_samples; ++i) {
        float t = float(i) / float(num_samples);
        vec2 current_point = bezier(t);
        min_dist_sq = min(min_dist_sq, distSqPointLineSegment(p, prev_point, current_point));
        prev_point = current_point;
    }
    return sqrt(min_dist_sq); // 返回实际距离
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

    // 计算像素中心到贝塞尔曲线的近似距离 (使用归一化坐标)
    float dist = distanceToBezier(uv);

    // --- 抗锯齿计算 ---
    float halfWidthNorm = (lineWidth * 0.5) / textureSize.y;
    float aa_width = 1.0 / textureSize.y; // 抗锯齿过渡带宽度 (1 像素)
    float alpha = 1.0 - smoothstep(halfWidthNorm - aa_width, halfWidthNorm + aa_width, dist);

    // 根据 alpha 混合曲线和背景颜色
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
        self.texture_width = 512 # 固定纹理尺寸
        self.texture_height = 512

        # 计算着色器的 Uniform 位置
        self.p0_loc = -1
        self.p1_loc = -1
        self.p2_loc = -1
        self.p3_loc = -1
        self.tex_size_loc = -1
        self.line_width_loc = -1
        self.curve_color_loc = -1
        self.bg_color_loc = -1

        # 定时器仍然需要，以便定期调用 paintGL 来绘制
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update) # update() 调用 paintGL()

    def initializeGL(self):
        context = QOpenGLContext.currentContext()
        version_profile = context.format()
        print(f"OpenGL 版本: {version_profile.majorVersion()}.{version_profile.minorVersion()} Profile: {'Core' if version_profile.profile() == QSurfaceFormat.CoreProfile else 'Compatibility'}")

        if version_profile.majorVersion() < 4 or (version_profile.majorVersion() == 4 and version_profile.minorVersion() < 3):
            print("\n错误: 需要 OpenGL 4.3+")
            QApplication.instance().quit()
            return

        glClearColor(0.1, 0.1, 0.1, 1.0) # 设置屏幕清除颜色

        print("编译着色器...")
        self.compute_program = self.compileShader(COMPUTE_SHADER_SRC, GL_COMPUTE_SHADER, "Compute")
        self.render_program = self.compileShader(RENDER_VERTEX_SHADER_SRC, GL_VERTEX_SHADER, "Render Vertex",
                                                  RENDER_FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER, "Render Fragment")

        if not self.compute_program or not self.render_program:
             print("着色器编译失败。正在退出。")
             QApplication.instance().quit()
             return

        # 获取 Uniform 位置 (渲染着色器)
        self.texture_uniform_location_render = glGetUniformLocation(self.render_program, "computeOutputTexture")
        if self.texture_uniform_location_render == -1:
            print("警告: 在渲染着色器中未找到 'computeOutputTexture' uniform。")

        # 获取 Uniform 位置 (计算着色器)
        self.p0_loc = glGetUniformLocation(self.compute_program, "P0")
        self.p1_loc = glGetUniformLocation(self.compute_program, "P1")
        self.p2_loc = glGetUniformLocation(self.compute_program, "P2")
        self.p3_loc = glGetUniformLocation(self.compute_program, "P3") # 修正后为 vec2
        self.tex_size_loc = glGetUniformLocation(self.compute_program, "textureSize")
        self.line_width_loc = glGetUniformLocation(self.compute_program, "lineWidth")
        self.curve_color_loc = glGetUniformLocation(self.compute_program, "curveColor")
        self.bg_color_loc = glGetUniformLocation(self.compute_program, "backgroundColor")

        # 检查 Uniform 获取情况
        uniform_locations = {
            "P0": self.p0_loc, "P1": self.p1_loc, "P2": self.p2_loc, "P3": self.p3_loc,
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
        self.timer.start(16) # ~60 FPS 更新 paintGL

    def compileShader(self, src1, type1, name1, src2=None, type2=None, name2=None):
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
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        # 定义纹理存储 (不可变) - GL_RGBA8 格式匹配计算着色器布局
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, self.texture_width, self.texture_height)
        # 设置纹理参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        print(f"纹理已创建 (ID: {self.texture_id}) 尺寸 {self.texture_width}x{self.texture_height}")

    def setupQuad(self):
        # 定义顶点数据 (位置 + 纹理坐标)
        quad_vertices = np.array([
            # 位置        纹理坐标
            -1.0, -1.0,   0.0, 0.0, # 左下
             1.0, -1.0,   1.0, 0.0, # 右下
             1.0,  1.0,   1.0, 1.0, # 右上
            -1.0,  1.0,   0.0, 1.0  # 左上
        ], dtype=np.float32)
        # 定义索引数据 (两个三角形组成四边形)
        quad_indices = np.array([
            0, 1, 2, # 第一个三角形
            2, 3, 0  # 第二个三角形
        ], dtype=np.uint32)

        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        self.quad_ebo = glGenBuffers(1) # 元素缓冲对象

        glBindVertexArray(self.quad_vao)
        # 上传顶点数据
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        # 上传索引数据
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quad_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)
        # 配置顶点属性指针
        stride = 4 * sizeof(GLfloat) # 每个顶点4个float (pos.x, pos.y, tex.s, tex.t)
        # 位置属性 (location = 0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # 纹理坐标属性 (location = 1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * sizeof(GLfloat)))
        glEnableVertexAttribArray(1)
        # 解绑 VAO（VAO 会记录 EBO 的绑定状态）
        glBindVertexArray(0)
        # 解绑其他缓冲区
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0) # EBO的绑定记录在VAO中，解绑VAO后这里解绑是可选的，但保持一致性
        print("四边形 VAO/VBO/EBO 设置完成。")

    def resizeGL(self, width, height):
        if height == 0: height = 1
        glViewport(0, 0, width, height)

    def paintGL(self):
        if not self.compute_program or not self.render_program or not self.texture_id or not self.quad_vao:
            return # 未初始化完成

        # --- 计算阶段 ---
        glUseProgram(self.compute_program)

        # 设置计算着色器的 Uniform 变量
        # 贝塞尔曲线控制点 (归一化坐标 0.0 到 1.0)
        p0 = QVector2D(0.1, 0.1)
        p1 = QVector2D(0.9, 0.1)
        p2 = QVector2D(0.1, 0.9)
        p3 = QVector2D(0.9, 0.9)
        line_width_px = 4.0 # 线条宽度 (像素)
        curve_col = QVector4D(1.0, 1.0, 1.0, 1.0) # 白色曲线
        bg_col = QVector4D(0.2, 0.2, 0.2, 1.0)    # 深灰色背景

        # 使用 glUniform* 函数传递数据
        if self.p0_loc != -1: glUniform2f(self.p0_loc, p0.x(), p0.y())
        if self.p1_loc != -1: glUniform2f(self.p1_loc, p1.x(), p1.y())
        if self.p2_loc != -1: glUniform2f(self.p2_loc, p2.x(), p2.y())
        if self.p3_loc != -1: glUniform2f(self.p3_loc, p3.x(), p3.y())
        if self.tex_size_loc != -1: glUniform2f(self.tex_size_loc, float(self.texture_width), float(self.texture_height))
        if self.line_width_loc != -1: glUniform1f(self.line_width_loc, line_width_px)
        if self.curve_color_loc != -1: glUniform4f(self.curve_color_loc, curve_col.x(), curve_col.y(), curve_col.z(), curve_col.w())
        if self.bg_color_loc != -1: glUniform4f(self.bg_color_loc, bg_col.x(), bg_col.y(), bg_col.z(), bg_col.w())

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
        # 清除屏幕背景色
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.render_program)

        # 绑定纹理以供读取/采样
        glActiveTexture(GL_TEXTURE0) # 激活纹理单元 0
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        if self.texture_uniform_location_render != -1:
            glUniform1i(self.texture_uniform_location_render, 0) # 告知采样器使用纹理单元 0

        # 绑定并绘制四边形
        glBindVertexArray(self.quad_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None) # 使用 EBO 中的索引

        # 解绑资源
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0) # 从纹理单元 0 解绑
        glUseProgram(0)

    def cleanupGL(self):
        print("清理 GL 资源...")
        # 确保在删除 OpenGL 对象前使上下文成为当前状态
        if self.context().isValid():
             self.makeCurrent()
             # 安全地删除资源，检查变量是否存在
             if hasattr(self, 'quad_ebo') and self.quad_ebo:
                  print(f"删除 EBO: {self.quad_ebo}")
                  glDeleteBuffers(1, [self.quad_ebo]); self.quad_ebo = None
             if hasattr(self, 'quad_vbo') and self.quad_vbo:
                  print(f"删除 VBO: {self.quad_vbo}")
                  glDeleteBuffers(1, [self.quad_vbo]); self.quad_vbo = None
             if hasattr(self, 'quad_vao') and self.quad_vao:
                  print(f"删除 VAO: {self.quad_vao}")
                  glDeleteVertexArrays(1, [self.quad_vao]); self.quad_vao = None
             if hasattr(self, 'texture_id') and self.texture_id:
                  print(f"删除纹理: {self.texture_id}")
                  glDeleteTextures(1, [self.texture_id]); self.texture_id = None
             if hasattr(self, 'compute_program') and self.compute_program:
                  print(f"删除计算着色器程序: {self.compute_program}")
                  glDeleteProgram(self.compute_program); self.compute_program = None
             if hasattr(self, 'render_program') and self.render_program:
                  print(f"删除渲染着色器程序: {self.render_program}")
                  glDeleteProgram(self.render_program); self.render_program = None
             self.doneCurrent() # 释放上下文
             print("清理尝试完成。")
        else:
             print("OpenGL 上下文无效，跳过清理。")


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("PySide6 计算着色器绘制贝塞尔曲线")
        self.setGeometry(100, 100, 550, 550) # 调整窗口大小以适合纹理
        self.gl_widget = OpenGLWidget()
        self.setCentralWidget(self.gl_widget)

    # 重写 closeEvent 以确保 OpenGL 资源被清理
    def closeEvent(self, event):
        print("主窗口关闭...")
        self.gl_widget.cleanupGL()
        super().closeEvent(event)


if __name__ == "__main__":
    # 启用高 DPI 缩放
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # 请求 OpenGL 4.3 Core Profile
    gl_format = QSurfaceFormat()
    print("请求 OpenGL 4.3 Core Profile...")
    gl_format.setVersion(4, 3)
    gl_format.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(gl_format) # 在创建 QApplication 前设置

    app = QApplication(sys.argv)

    print("创建主窗口...")
    window = MainWindow()
    window.show()
    print("启动应用程序事件循环...")
    sys.exit(app.exec())