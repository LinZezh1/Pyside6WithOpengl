import sys
import numpy as np
import math
import freetype # <--- 导入 FreeType 库
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QElapsedTimer # 需要 QElapsedTimer
from PySide6.QtGui import QSurfaceFormat, QOpenGLContext, QVector2D, QVector4D

# --- 配置 ---
# *** 修改为你本地的字体文件路径 ***
FONT_PATH = "NotoSansCJKsc-Regular.otf" # 例如 "C:/Windows/Fonts/msyh.ttc" 或下载的 Noto 字体路径
TARGET_CHAR = "好"
FONT_PIXEL_SIZE = 128 # 渲染字体到位图的像素大小
TEXTURE_WIDTH = 256 # 输出纹理宽度
TEXTURE_HEIGHT = 256 # 输出纹理高度

# test：使用 Computer Shader 结合 FreeType 绘制中文字符
try:
    from OpenGL.GL import *
    # 检查必要的 OpenGL 函数
    required_gl_funcs = ['glDispatchCompute', 'GL_COMPUTE_SHADER', 'glBindImageTexture',
                         'glTexStorage2D', 'glMemoryBarrier', 'GL_SHADER_IMAGE_ACCESS_BARRIER_BIT',
                         'glGenVertexArrays', 'glBindVertexArray', 'glGenBuffers', 'glBindBuffer',
                         'glBufferData', 'glVertexAttribPointer', 'glEnableVertexAttribArray',
                         'glUseProgram', 'glActiveTexture', 'glBindTexture', 'glUniform1i',
                         'glDrawElements', 'GL_RGBA8', 'glUniform1f', 'glUniform2f', 'glUniform4f',
                         'glTexImage2D', 'GL_RED', 'GL_R8', 'GL_UNSIGNED_BYTE',
                         'glPixelStorei', 'GL_UNPACK_ALIGNMENT']
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
try:
    import freetype
except ImportError:
    print(f"错误: 未安装 freetype-py。请安装: pip install freetype-py")
    print(f"并且需要一个字体文件 (如 NotoSansCJKsc-Regular.otf) 在路径: {FONT_PATH}")
    sys.exit(1)

# == Shader Sources ==

# -- 计算着色器 (Compute Shader) --
# 读取预渲染的字形纹理 (灰度/alpha), 应用效果并写入输出纹理
COMPUTE_SHADER_SRC = """
#version 430 core

// 输出纹理 (RGBA)
layout (rgba8, binding = 0) uniform writeonly image2D destTex;

// 输入 Uniforms
uniform sampler2D fontTexture;     // 预渲染的字形纹理 (单通道)
uniform vec2 textureSize;         // 输出纹理尺寸
uniform float time;              // 时间，用于动画效果
uniform vec4 charColor;           // 字符基础颜色
uniform vec4 backgroundColor;     // 背景颜色

// 本地工作组大小
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    // 获取像素坐标
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(textureSize);

    // 边界检查
    if (pixelCoords.x >= size.x || pixelCoords.y >= size.y) {
        return;
    }

    // 计算像素对应的原始 UV 坐标 (Y 从下到上 0->1)
    // *** 确保变量名为 uv ***
    vec2 uv = vec2(pixelCoords) / textureSize;

    // 计算用于采样 FreeType 纹理的翻转 UV 坐标 (Y 从上到下 0->1)
    vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);

    // 使用翻转后的 UV 采样字体纹理
    float alpha = texture(fontTexture, flipped_uv).r;

    // --- 应用效果 (示例：颜色脉冲) ---
    vec3 effectColor = charColor.rgb;
    if (alpha > 0.05) { // 只对字符部分应用效果
        // *** 确保这里使用的是已定义的 uv.y ***
        // 基于时间和 *原始* 纵坐标的简单亮度脉冲
        effectColor *= (0.75 + 0.25 * sin(time * 4.0 + uv.y * 15.0)); // 使用 uv.y
    }

    // --- 混合颜色 ---
    // 使用字形的 alpha 值混合效果颜色和背景色
    vec4 finalColor = mix(backgroundColor, vec4(effectColor, 1.0), alpha);

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
uniform sampler2D computeOutputTexture; // 采样由 Compute Shader 生成的最终纹理
void main() {
    FragColor = texture(computeOutputTexture, TexCoord);
}
"""


class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.compute_program = None
        self.render_program = None
        self.texture_id = None      # Compute shader 的输出纹理 (RGBA)
        self.font_texture_id = None # 存储 FreeType 渲染字形的纹理 (单通道)
        self.quad_vao = None
        self.quad_vbo = None
        self.quad_ebo = None

        # Uniform 位置
        self.texture_uniform_location_render = -1
        self.font_tex_loc_compute = -1
        self.tex_size_loc_compute = -1
        self.time_loc_compute = -1
        self.char_color_loc_compute = -1
        self.bg_color_loc_compute = -1

        # 纹理尺寸
        self.texture_width = TEXTURE_WIDTH
        self.texture_height = TEXTURE_HEIGHT

        # FreeType 渲染的字形位图数据
        self.char_bitmap_data = None
        self.char_bitmap_width = 0
        self.char_bitmap_rows = 0

        # 计时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.start_time = QElapsedTimer()

    def initializeGL(self):
        context = QOpenGLContext.currentContext()
        version_profile = context.format()
        print(f"OpenGL 版本: {version_profile.majorVersion()}.{version_profile.minorVersion()} Profile: {'Core' if version_profile.profile() == QSurfaceFormat.CoreProfile else 'Compatibility'}")
        if version_profile.majorVersion() < 4 or (version_profile.majorVersion() == 4 and version_profile.minorVersion() < 3):
            print("\n错误: 需要 OpenGL 4.3+")
            QApplication.instance().quit()
            return
        glClearColor(0.1, 0.1, 0.1, 1.0)

        # 1. 使用 FreeType 加载并渲染字形到位图
        if not self.loadAndRenderGlyph():
            QApplication.instance().quit()
            return

        # 2. 编译着色器
        print("编译着色器...")
        self.compute_program = self.compileShader(COMPUTE_SHADER_SRC, GL_COMPUTE_SHADER, "Compute")
        self.render_program = self.compileShader(RENDER_VERTEX_SHADER_SRC, GL_VERTEX_SHADER, "Render Vertex",
                                                  RENDER_FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER, "Render Fragment")
        if not self.compute_program or not self.render_program:
             print("着色器编译失败。正在退出。")
             QApplication.instance().quit()
             return

        # 3. 获取 Uniform 位置
        print("获取 Uniform 位置...")
        self.texture_uniform_location_render = glGetUniformLocation(self.render_program, "computeOutputTexture")
        if self.texture_uniform_location_render == -1: print("警告: Render shader 'computeOutputTexture' uniform 未找到。")
        self.font_tex_loc_compute = glGetUniformLocation(self.compute_program, "fontTexture")
        self.tex_size_loc_compute = glGetUniformLocation(self.compute_program, "textureSize")
        self.time_loc_compute = glGetUniformLocation(self.compute_program, "time")
        self.char_color_loc_compute = glGetUniformLocation(self.compute_program, "charColor")
        self.bg_color_loc_compute = glGetUniformLocation(self.compute_program, "backgroundColor")
        uniform_locs_compute = {"fontTexture":self.font_tex_loc_compute, "textureSize":self.tex_size_loc_compute,
                                "time":self.time_loc_compute, "charColor":self.char_color_loc_compute,
                                "backgroundColor":self.bg_color_loc_compute}
        for name, loc in uniform_locs_compute.items():
            if loc == -1: print(f"警告: Compute shader '{name}' uniform 未找到。")

        # 4. 设置 OpenGL 纹理和几何体
        print("设置 OpenGL 资源...")
        self.setupFontTexture()
        self.setupOutputTexture()
        self.setupQuad()

        print("初始化完成。启动定时器...")
        self.timer.start(16)
        self.start_time.start()

    def loadAndRenderGlyph(self):
        """使用 FreeType 加载字体并渲染目标字符到位图"""
        print(f"尝试加载字体: {FONT_PATH}")
        try:
            face = freetype.Face(FONT_PATH)
            print(f"字体加载成功: {face.family_name.decode()} {face.style_name.decode()}")
        except freetype.FT_Exception as e:
            print(f"错误: 无法加载字体文件 '{FONT_PATH}'. 请检查路径是否正确。错误信息: {e}")
            return False

        try:
            face.set_pixel_sizes(0, FONT_PIXEL_SIZE)
            face.load_char(TARGET_CHAR, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
            glyph_bitmap = face.glyph.bitmap

            self.char_bitmap_width = glyph_bitmap.width
            self.char_bitmap_rows = glyph_bitmap.rows
            buffer_data = glyph_bitmap.buffer

            print(f"字形 '{TARGET_CHAR}' 渲染成功: 尺寸={self.char_bitmap_width}x{self.char_bitmap_rows}, Pitch={glyph_bitmap.pitch}")

            if glyph_bitmap.pixel_mode == freetype.FT_PIXEL_MODE_GRAY:
                 self.char_bitmap_data = np.zeros((self.char_bitmap_rows, self.char_bitmap_width), dtype=np.uint8)
                 # 逐行复制数据，考虑 pitch
                 for r in range(self.char_bitmap_rows):
                     start_index = r * glyph_bitmap.pitch
                     # 从 FreeType buffer 中提取当前行的数据 (作为列表或类似序列)
                     row_list_or_sequence = buffer_data[start_index : start_index + self.char_bitmap_width]
                     # *** FIX: 使用 np.array() 而不是 np.frombuffer() 来处理列表 ***
                     try:
                         self.char_bitmap_data[r, :] = np.array(row_list_or_sequence, dtype=np.uint8)
                     except ValueError as ve:
                         print(f"错误: 转换第 {r} 行数据时出错。数据片段长度可能不匹配宽度。")
                         print(f"    预期宽度: {self.char_bitmap_width}, 获取的数据长度: {len(row_list_or_sequence)}")
                         raise ve # 重新抛出错误
                     # *** FIX END ***
                 print("FreeType 位图数据已成功转换为 NumPy 数组。")
                 return True
            else:
                 print(f"错误: 不支持的 FreeType 像素模式: {glyph_bitmap.pixel_mode}")
                 return False

        except freetype.FT_Exception as e:
            print(f"错误: 处理字体或字形 '{TARGET_CHAR}' 时出错: {e}")
            return False
        except Exception as e:
            print(f"错误: 转换 FreeType 位图时发生未知错误: {e}") # 这里会捕获并打印 TypeError
            return False


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
                glDeleteShader(shader1); glDeleteShader(shader2); glDeleteProgram(program)
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
        glDetachShader(program, shader1); glDeleteShader(shader1)
        if shader2: glDetachShader(program, shader2); glDeleteShader(shader2)
        print(f"{name1}{' & ' + name2 if name2 else ''} 着色器编译链接成功。")
        return program


    def setupFontTexture(self):
        """创建并上传 FreeType 渲染的字形位图到 OpenGL 纹理"""
        if self.char_bitmap_data is None:
            print("错误: 字形位图数据未加载，无法创建字体纹理。")
            return

        self.font_texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.font_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1) # 重要: 设置为 1
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, # 使用 GL_R8 内部格式
                     self.char_bitmap_width, self.char_bitmap_rows, 0,
                     GL_RED, GL_UNSIGNED_BYTE, self.char_bitmap_data)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4) # 恢复默认对齐
        glBindTexture(GL_TEXTURE_2D, 0)
        print(f"字体纹理已创建并上传 (ID: {self.font_texture_id})，源尺寸: {self.char_bitmap_width}x{self.char_bitmap_rows}")

    def setupOutputTexture(self):
        """创建计算着色器的输出纹理 (RGBA)"""
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, self.texture_width, self.texture_height)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST) # 输出纹理通常用 NEAREST
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        print(f"计算着色器输出纹理已创建 (ID: {self.texture_id}) 尺寸 {self.texture_width}x{self.texture_height}")

    def setupQuad(self):
        # (设置四边形逻辑保持不变)
        quad_vertices = np.array([ -1.0,-1.0,0.0,0.0, 1.0,-1.0,1.0,0.0, 1.0,1.0,1.0,1.0, -1.0,1.0,0.0,1.0 ], dtype=np.float32)
        quad_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        self.quad_vao = glGenVertexArrays(1); self.quad_vbo = glGenBuffers(1); self.quad_ebo = glGenBuffers(1)
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo); glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quad_ebo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)
        stride = 4 * sizeof(GLfloat)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * sizeof(GLfloat))); glEnableVertexAttribArray(1)
        glBindVertexArray(0); glBindBuffer(GL_ARRAY_BUFFER, 0); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        print("四边形 VAO/VBO/EBO 设置完成。")


    def resizeGL(self, width, height):
        # (保持不变)
        if height == 0: height = 1
        glViewport(0, 0, width, height)

    def paintGL(self):
        if not self.compute_program or not self.render_program or not self.texture_id or not self.font_texture_id or not self.quad_vao:
            return # 未初始化完成

        current_time_sec = self.start_time.elapsed() / 1000.0

        # --- 计算阶段 ---
        glUseProgram(self.compute_program)

        # 设置 Uniforms
        char_col = QVector4D(0.9, 0.9, 0.9, 1.0) # 浅灰色字符
        bg_col = QVector4D(0.1, 0.1, 0.15, 1.0)  # 深灰蓝背景

        if self.tex_size_loc_compute != -1: glUniform2f(self.tex_size_loc_compute, float(self.texture_width), float(self.texture_height))
        if self.time_loc_compute != -1: glUniform1f(self.time_loc_compute, current_time_sec)
        if self.char_color_loc_compute != -1: glUniform4f(self.char_color_loc_compute, char_col.x(), char_col.y(), char_col.z(), char_col.w())
        if self.bg_color_loc_compute != -1: glUniform4f(self.bg_color_loc_compute, bg_col.x(), bg_col.y(), bg_col.z(), bg_col.w())

        # 绑定输入字体纹理到纹理单元 1
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.font_texture_id)
        if self.font_tex_loc_compute != -1:
            glUniform1i(self.font_tex_loc_compute, 1) # 告知 sampler 使用纹理单元 1

        # 绑定输出纹理到图像单元 0 以供写入
        glBindImageTexture(0, self.texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8)

        # 调度计算着色器
        local_size_x = 8; local_size_y = 8
        num_groups_x = (self.texture_width + local_size_x - 1) // local_size_x
        num_groups_y = (self.texture_height + local_size_y - 1) // local_size_y
        glDispatchCompute(num_groups_x, num_groups_y, 1)

        # 关键屏障
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # --- 渲染阶段 ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.render_program)

        # 绑定由计算着色器生成的最终纹理到纹理单元 0 以供渲染
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        if self.texture_uniform_location_render != -1:
            glUniform1i(self.texture_uniform_location_render, 0)

        # 绘制四边形
        glBindVertexArray(self.quad_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # 解绑
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0) # 从纹理单元 0 解绑
        glUseProgram(0)

    def cleanupGL(self):
        # (清理逻辑保持不变，已包含 font_texture_id)
        print("清理 GL 资源...")
        if self.context().isValid():
             self.makeCurrent()
             if hasattr(self, 'quad_ebo') and self.quad_ebo: glDeleteBuffers(1, [self.quad_ebo]); self.quad_ebo = None
             if hasattr(self, 'quad_vbo') and self.quad_vbo: glDeleteBuffers(1, [self.quad_vbo]); self.quad_vbo = None
             if hasattr(self, 'quad_vao') and self.quad_vao: glDeleteVertexArrays(1, [self.quad_vao]); self.quad_vao = None
             if hasattr(self, 'texture_id') and self.texture_id: print(f"删除输出纹理: {self.texture_id}"); glDeleteTextures(1, [self.texture_id]); self.texture_id = None
             if hasattr(self, 'font_texture_id') and self.font_texture_id: print(f"删除字体纹理: {self.font_texture_id}"); glDeleteTextures(1, [self.font_texture_id]); self.font_texture_id = None
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
        self.setWindowTitle(f"PySide6 FreeType + ComputerShader 创建中文字符")
        self.setGeometry(100, 100, TEXTURE_WIDTH + 50, TEXTURE_HEIGHT + 50)
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