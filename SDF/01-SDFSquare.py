# 文件名: sdf_square_hollow_pyside6.py
import sys
import numpy as np
from OpenGL.GL import *
# from OpenGL.GLUT import * # 移除 GLUT 导入
import ctypes
import math

# --- 导入 PySide6 相关模块 ---
from PySide6.QtWidgets import QApplication, QWidget # QWidget 用于主窗口基类(虽然我们直接用 QOpenGLWidget)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QTimer, QSize # QTimer 用于驱动更新
from PySide6.QtGui import QSurfaceFormat, QKeyEvent, Qt # QSurfaceFormat 设置OpenGL版本

# --- 形状参数 (全局或移入类中) ---
# 将这些设为类的成员变量通常更好，但为简化，暂时保留全局
square_size = 0.6   # 正方形 "半径"
outline_thickness = 0.05 # 边框的粗细
square_color = np.array([0.9, 0.7, 0.2, 1.0], dtype=np.float32) # 黄色

# --- Shader Code ---
# (顶点和片段着色器代码保持不变)
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;

uniform vec2 u_resolution;
uniform float u_square_size;
uniform float u_outline_thickness;
uniform vec4 u_color;

// 正方形 SDF (L-infinity norm based)
float sdSquare(vec2 p, float s) {
    return max(abs(p.x), abs(p.y)) - s;
}

void main()
{
    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;
    vec2 p = uv;
    float dist = sdSquare(p, u_square_size);
    float abs_dist = abs(dist);
    float half_thickness = u_outline_thickness * 0.5;
    float pixelWidth = 2.0 / u_resolution.y;
    float alpha = smoothstep(half_thickness + pixelWidth, half_thickness - pixelWidth, abs_dist);
    FragColor = vec4(u_color.rgb, u_color.a * alpha);
}
"""

# --- OpenGL Helper Functions ---
# (这些可以保持独立，或者作为类的静态/普通方法)
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        glDeleteShader(shader)
        raise RuntimeError(f"Shader compilation error:\n{error}")
    return shader

def create_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        glDeleteProgram(program)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        raise RuntimeError(f"Program linking error:\n{error}")
    glDetachShader(program, vertex_shader)
    glDetachShader(program, fragment_shader)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

def get_uniform_locations(program):
    locs = {}
    locs['resolution'] = glGetUniformLocation(program, "u_resolution")
    locs['square_size'] = glGetUniformLocation(program, "u_square_size")
    locs['outline_thickness'] = glGetUniformLocation(program, "u_outline_thickness")
    locs['color'] = glGetUniformLocation(program, "u_color")
    for name, loc in locs.items():
        if loc == -1: print(f"Warning: Uniform '{name}' not found.")
    return locs

# --- PySide6 QOpenGLWidget 子类 ---
class SDFWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.program = None
        self.vao = None
        self.vbo = None
        self.uniform_locs = {}

        # 将形状参数移入类实例变量
        self.square_size = 0.6
        self.outline_thickness = 0.05
        self.square_color = np.array([0.9, 0.7, 0.2, 1.0], dtype=np.float32) # 黄色

        # 设置定时器以定期更新 (对于静态图像不是必须，但对动画或平滑交互有用)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update) # 连接 timeout 信号到 update() 插槽
        self.timer.start(16) # 大约 60 FPS (1000ms / 60fps ≈ 16.67ms)

    def minimumSizeHint(self) -> QSize:
        # 提供一个最小尺寸提示
        return QSize(100, 100)

    def sizeHint(self) -> QSize:
        # 提供一个建议的初始尺寸
        return QSize(800, 600)

    # -- QOpenGLWidget 必须实现的三个核心方法 --

    def initializeGL(self):
        """OpenGL 初始化，只调用一次"""
        print("Initializing OpenGL...")
        print(f"Vendor: {glGetString(GL_VENDOR).decode()}")
        print(f"Renderer: {glGetString(GL_RENDERER).decode()}")
        print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
        print(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}")

        try:
            print("Compiling shaders...")
            self.program = create_program(vertex_shader_source, fragment_shader_source)
            self.uniform_locs = get_uniform_locations(self.program)
            print("Shaders linked.")
        except RuntimeError as e:
            print(e)
            QApplication.quit() # 编译失败则退出
            return

        # 创建 VAO 和 VBO
        quad_vertices = np.array([-1.0,-1.0, 1.0,-1.0, 1.0,1.0, -1.0,-1.0, 1.0,1.0, -1.0,1.0], dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * ctypes.sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # 设置 OpenGL 状态
        glClearColor(0.15, 0.15, 0.15, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        print("Initialization complete.")

    def resizeGL(self, w: int, h: int):
        """窗口大小改变时调用"""
        print(f"Resizing to {w}x{h}")
        # h_safe = max(1, h) # 避免除以零
        glViewport(0, 0, w, h)
        # 注意：我们不再需要全局变量 window_width/height
        # 可以在 paintGL 中直接使用 self.width() 和 self.height()

    def paintGL(self):
        """渲染窗口内容时调用"""
        if not self.program or not self.vao:
            return # 避免在初始化完成前绘制

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        # 更新 Uniforms
        # 使用 self.width() 和 self.height() 获取当前尺寸
        current_width = self.width()
        current_height = max(1, self.height()) # 确保不为零
        glUniform2f(self.uniform_locs['resolution'], float(current_width), float(current_height))
        glUniform1f(self.uniform_locs['square_size'], self.square_size)
        glUniform1f(self.uniform_locs['outline_thickness'], self.outline_thickness)
        glUniform4fv(self.uniform_locs['color'], 1, self.square_color)

        # 绘制
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

    # -- 事件处理 --
    def keyPressEvent(self, event: QKeyEvent):
        """处理键盘按下事件"""
        needs_update = True
        key = event.key() # 获取按键代码

        if key == Qt.Key_Escape:
            print("Exiting...")
            self.close() # 关闭窗口会触发 QApplication 退出
            return
        # --- 控制大小 ---
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.square_size *= 1.1
            print(f"Square size: {self.square_size:.3f}")
        elif key == Qt.Key_Minus or key == Qt.Key_Underscore:
            self.square_size /= 1.1
            print(f"Square size: {self.square_size:.3f}")
        # --- 控制边框粗细 ---
        elif key == Qt.Key_BracketRight or key == Qt.Key_BraceRight:
            self.outline_thickness *= 1.2
            print(f"Outline thickness: {self.outline_thickness:.4f}")
        elif key == Qt.Key_BracketLeft or key == Qt.Key_BraceLeft:
            self.outline_thickness /= 1.2
            self.outline_thickness = max(0.0001, self.outline_thickness)
            print(f"Outline thickness: {self.outline_thickness:.4f}")
        else:
            needs_update = False
            super().keyPressEvent(event) # 将其他按键事件传递给基类处理

        # if needs_update:
        #     self.update() # 请求重绘 (定时器已经在做了，这里可以省略)
        #     pass

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        print("Cleaning up OpenGL resources...")
        # PySide6 的 QOpenGLWidget 会自动管理上下文，
        # 但手动删除程序、VAO、VBO 是好习惯
        # 需要先 makeCurrent() 才能调用 glDelete*
        self.makeCurrent()
        if self.program: glDeleteProgram(self.program)
        if self.vao: glDeleteVertexArrays(1, [self.vao])
        if self.vbo: glDeleteBuffers(1, [self.vbo])
        self.doneCurrent()
        print("Cleanup finished.")
        super().closeEvent(event)


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # --- 设置 OpenGL 版本和配置 ---
    # 必须在创建 QOpenGLWidget 之前设置
    format = QSurfaceFormat()
    format.setVersion(3, 3) # 请求 OpenGL 3.3
    format.setProfile(QSurfaceFormat.CoreProfile) # 请求核心配置
    # format.setSamples(4) # 可选：启用多重采样抗锯齿 (MSAA)
    QSurfaceFormat.setDefaultFormat(format)
    # ---

    # 创建并显示窗口
    window = SDFWidget()
    window.setWindowTitle("PySide6 OpenGL SDF Hollow Square")
    window.resize(window.sizeHint()) # 使用建议的尺寸
    window.show()

    # 运行 Qt 应用程序事件循环
    sys.exit(app.exec())