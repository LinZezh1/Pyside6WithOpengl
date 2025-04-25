import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import *
import numpy as np

# 使用 Shader 绘制简单几何图形
class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shape_type = 0  # 0=圆形, 1=正方形, 2=三角形

    def initializeGL(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        self.setupShaders()
        self.setupGeometry()

    def setupShaders(self):
        vertex_shader = """
        #version 330 core
        const vec2 pos[4] = vec2[](
            vec2(-1.0, -1.0),
            vec2( 1.0, -1.0),
            vec2( 1.0,  1.0),
            vec2(-1.0,  1.0)
        );
        void main() {
            gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
        }
        """

        fragment_shader = """
        #version 330 core
        out vec4 outColor;
        uniform int shapeType;

        void main() {
            vec2 uv = gl_FragCoord.xy / vec2(800, 600); // 窗口大小
            vec2 center = vec2(0.5, 0.5);
            vec2 pos = uv - center;
            float d = length(pos);

            if (shapeType == 0) {
                // 圆形
                if (d < 0.4)
                    outColor = vec4(1.0, 0.0, 0.0, 1.0);
                else
                    outColor = vec4(0.2, 0.3, 0.3, 1.0);
            } else if (shapeType == 1) {
                // 正方形
                if (abs(pos.x) < 0.4 && abs(pos.y) < 0.4)
                    outColor = vec4(0.0, 0.0, 1.0, 1.0);
                else
                    outColor = vec4(0.2, 0.3, 0.3, 1.0);
            } else if (shapeType == 2) {
                // 三角形 (上顶点0.0,0.4，左右-0.4,-0.4)
                vec2 p0 = vec2(0.0, 0.4);
                vec2 p1 = vec2(-0.4, -0.4);
                vec2 p2 = vec2(0.4, -0.4);

                // 计算重心坐标法判断点是否在三角形内
                float area = abs((p1.x-p0.x)*(p2.y-p0.y) - (p2.x-p0.x)*(p1.y-p0.y));
                float a = abs((p1.x-pos.x)*(p2.y-pos.y) - (p2.x-pos.x)*(p1.y-pos.y)) / area;
                float b = abs((p2.x-pos.x)*(p0.y-pos.y) - (p0.x-pos.x)*(p2.y-pos.y)) / area;
                float c = abs((p0.x-pos.x)*(p1.y-pos.y) - (p1.x-pos.x)*(p0.y-pos.y)) / area;

                if (a >= 0.0 && b >= 0.0 && c >= 0.0 && (a + b + c) <= 1.0 + 0.01)
                    outColor = vec4(0.0, 1.0, 0.0, 1.0);
                else
                    outColor = vec4(0.2, 0.3, 0.3, 1.0);
            }
        }
        """

        self.shader = self.compileShaders(vertex_shader, fragment_shader)

    def compileShaders(self, vertex_src, fragment_src):
        vertexShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexShader, vertex_src)
        glCompileShader(vertexShader)
        if not glGetShaderiv(vertexShader, GL_COMPILE_STATUS):
            print(glGetShaderInfoLog(vertexShader))

        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragmentShader, fragment_src)
        glCompileShader(fragmentShader)
        if not glGetShaderiv(fragmentShader, GL_COMPILE_STATUS):
            print(glGetShaderInfoLog(fragmentShader))

        program = glCreateProgram()
        glAttachShader(program, vertexShader)
        glAttachShader(program, fragmentShader)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            print(glGetProgramInfoLog(program))

        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)

        return program

    def setupGeometry(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "shapeType"), self.shape_type)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)

    def setShape(self, shape):
        self.shape_type = shape
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("纯 Shader 图形绘制")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.gl_widget = OpenGLWidget()
        layout.addWidget(self.gl_widget)

        button_circle = QPushButton("圆形")
        button_circle.clicked.connect(lambda: self.gl_widget.setShape(0))
        layout.addWidget(button_circle)

        button_square = QPushButton("正方形")
        button_square.clicked.connect(lambda: self.gl_widget.setShape(1))
        layout.addWidget(button_square)

        button_triangle = QPushButton("三角形")
        button_triangle.clicked.connect(lambda: self.gl_widget.setShape(2))
        layout.addWidget(button_triangle)

        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3)
    gl_format.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(gl_format)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
