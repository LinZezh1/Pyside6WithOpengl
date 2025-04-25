import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt # Needed for messages
from PySide6.QtGui import QSurfaceFormat, QOpenGLContext # QOpenGLContext for version check

# Check if PyOpenGL is available, otherwise provide a helpful message
try:
    from OpenGL.GL import *
    # Check if compute shader related functions are available (might indicate version support)
    if not hasattr(OpenGL.GL, 'glDispatchCompute') or not hasattr(OpenGL.GL, 'GL_COMPUTE_SHADER'):
         print("WARNING: PyOpenGL seems installed, but might lack Compute Shader support (check OpenGL drivers and PyOpenGL version).")
except ImportError:
    print("ERROR: PyOpenGL is not installed. Please install it: pip install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)

# Check if numpy is available
try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy is not installed. Please install it: pip install numpy")
    sys.exit(1)

# test：Opengl Computer shader 功能测试（未成功显示目标效果）
class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.compute_shader_program = None
        self.ssbo = None # Shader Storage Buffer Object ID
        self.buffer_size = 32 # Number of elements (integers) in the buffer
        self.compute_ran = False # Flag to ensure compute runs only once

    def initializeGL(self):
        # Check actual OpenGL Version obtained
        context = QOpenGLContext.currentContext()
        version_profile = context.format()
        print(f"OpenGL Version Initialized: {version_profile.majorVersion()}.{version_profile.minorVersion()}")
        print(f"OpenGL Profile: {'Core' if version_profile.profile() == QSurfaceFormat.CoreProfile else 'Compatibility'}")

        # Check if Compute Shaders are supported (basic check)
        if version_profile.majorVersion() < 4 or (version_profile.majorVersion() == 4 and version_profile.minorVersion() < 3):
            print("\nCRITICAL ERROR: OpenGL version 4.3 or higher is required for Compute Shaders.")
            print("Your system reported version {}.{}".format(version_profile.majorVersion(), version_profile.minorVersion()))
            print("Please ensure your graphics drivers are up-to-date and support OpenGL 4.3+.")
            # Optionally, close the application or show a message box
            QApplication.instance().quit() # Quit the application cleanly
            return # Stop initialization

        glClearColor(0.1, 0.1, 0.2, 1.0) # Darker background

        print("Setting up Compute Shader...")
        self.setupComputeShader()
        print("Setting up Compute Buffer (SSBO)...")
        self.setupComputeBuffer()
        print("Running Compute Shader...")
        self.runCompute()
        print("Compute Shader finished.")
        self.compute_ran = True

    def setupComputeShader(self):
        # Requires OpenGL 4.3+
        compute_shader_source = """
        #version 430 core

        // Define the size of the work group (e.g., 16 invocations per group)
        // Should match glDispatchCompute's relation to total size
        layout (local_size_x = 8, local_size_y = 1, local_size_z = 1) in;

        // Define the Shader Storage Buffer (SSBO)
        // 'binding = 0' must match glBindBufferBase in Python
        layout(std430, binding = 0) buffer DataBuffer {
            int data[]; // Array of integers
        };

        void main() {
            // Get the unique index for this shader instance across all work groups
            uint index = gl_GlobalInvocationID.x;

            // Simple compute task: write index squared to the buffer
            // Add a check to prevent out-of-bounds access if buffer_size is not a multiple of local_size_x
            // (Though glDispatchCompute should handle this)
            if (index < data.length()) { // data.length() gives buffer size in elements
                 data[index] = int(index * index);
                 // Example of atomic operation (less useful here, but common in compute)
                 // atomicAdd(data[0], 1); // Atomically increment the first element
            }
        }
        """
        self.compute_shader_program = self.compileComputeShader(compute_shader_source)

    def compileComputeShader(self, compute_source):
        computeShader = glCreateShader(GL_COMPUTE_SHADER)
        glShaderSource(computeShader, compute_source)
        glCompileShader(computeShader)
        if not glGetShaderiv(computeShader, GL_COMPILE_STATUS):
            error_log = glGetShaderInfoLog(computeShader).decode()
            print("Compute Shader compilation error:\n", error_log)
            glDeleteShader(computeShader) # Clean up
            return None # Indicate failure

        program = glCreateProgram()
        glAttachShader(program, computeShader)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            error_log = glGetProgramInfoLog(program).decode()
            print("Shader Program linking error:\n", error_log)
            glDeleteShader(computeShader) # Clean up attached shader
            glDeleteProgram(program) # Clean up program
            return None # Indicate failure

        # Detach and delete the shader now that it's linked
        glDetachShader(program, computeShader)
        glDeleteShader(computeShader)

        print("Compute Shader compiled and linked successfully.")
        return program

    def setupComputeBuffer(self):
        # Create initial data (e.g., all zeros) on the CPU
        # Use int32 as it matches 'int' in GLSL (usually 32-bit)
        initial_data = np.zeros(self.buffer_size, dtype=np.int32)
        print(f"Initial buffer data (first 10 elements): {initial_data[:10]}")

        # Generate buffer ID
        self.ssbo = glGenBuffers(1)

        # Bind the buffer to the Shader Storage Buffer target
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo)

        # Allocate buffer memory on the GPU and upload initial data
        # GL_DYNAMIC_DRAW is a hint, could be GL_STATIC_DRAW or GL_STREAM_DRAW too
        glBufferData(GL_SHADER_STORAGE_BUFFER, initial_data.nbytes, initial_data, GL_DYNAMIC_DRAW)

        # Unbind buffer from the generic target (optional, good practice)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        print(f"SSBO created (ID: {self.ssbo}) with size {initial_data.nbytes} bytes for {self.buffer_size} int32 elements.")


    def runCompute(self):
        if not self.compute_shader_program or not self.ssbo:
            print("Cannot run compute shader: Program or SSBO not initialized.")
            return

        # Activate the compute shader program
        glUseProgram(self.compute_shader_program)

        # Bind the SSBO to the correct binding point (index 0, matches shader layout)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo)

        # Determine the number of work groups needed
        # Work group size is defined in the shader (local_size_x = 8)
        local_size_x = 8 # Must match shader!
        num_groups_x = (self.buffer_size + local_size_x - 1) // local_size_x # Ceiling division
        print(f"Dispatching compute shader with {num_groups_x} work groups (local size: {local_size_x})...")

        # Dispatch the compute shader
        glDispatchCompute(num_groups_x, 1, 1) # Dispatch groups in X dimension only

        # Ensure Compute Shader writes are finished before reading back
        # GL_SHADER_STORAGE_BARRIER_BIT ensures memory writes to SSBOs are visible
        # to subsequent operations that access the same memory.
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        print("Compute dispatch finished, memory barrier set.")

        # --- Read back results ---
        # Bind the buffer to make it the current target for buffer operations
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo)

        # Map the buffer data to CPU memory (alternative to getBufferSubData)
        # ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
        # if ptr:
        #     # Read data directly from pointer (requires careful handling, maybe ctypes)
        #     # Example: result_data_map = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32)), shape=(self.buffer_size,))
        #     # print(f"Data read via glMapBuffer (first 10): {result_data_map[:10]}")
        #     # Make sure to unmap!
        #     glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        # else:
        #      print("Error mapping buffer!")

        # Or use glGetBufferSubData (often simpler for full buffer read)
        result_data = np.empty(self.buffer_size, dtype=np.int32)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, result_data.nbytes, result_data)

        # Unbind the buffer again
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        print(f"Result buffer data (first 10 elements): {result_data[:10]}")
        # Verify a few values
        print(f"Expected value at index 5: {5*5}, Got: {result_data[5]}")
        print(f"Expected value at index {self.buffer_size-1}: {(self.buffer_size-1)**2}, Got: {result_data[self.buffer_size-1]}")


    def resizeGL(self, width, height):
        # Viewport update might not be strictly necessary if not drawing,
        # but good practice to keep it.
        if height == 0: height = 1
        glViewport(0, 0, width, height)

    def paintGL(self):
        # We only clear the screen. The compute task runs once in initializeGL.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # If you wanted to run the compute shader repeatedly, you'd call
        # self.runCompute() here, but reading back every frame is SLOW.
        # Usually, compute results are used by subsequent rendering passes on the GPU.
        # if not self.compute_ran: # Ensure it only runs once if called from paintGL
        #      self.runCompute()
        #      self.compute_ran = True


    # Clean up GPU resources when the widget is about to be destroyed
    # This is important but might not always be called reliably on exit depending
    # on how the application terminates. Proper cleanup requires context management.
    def cleanupGL(self):
         print("Cleaning up GL resources...")
         if self.ssbo:
              glDeleteBuffers(1, [self.ssbo])
              self.ssbo = None
              print("Deleted SSBO.")
         if self.compute_shader_program:
              glDeleteProgram(self.compute_shader_program)
              self.compute_shader_program = None
              print("Deleted Compute Shader Program.")

    # Override closeEvent for the widget if necessary, or handle in MainWindow
    # def closeEvent(self, event):
    #     self.cleanupGL() # Attempt cleanup
    #     super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("PySide6 OpenGL Compute Shader Test")
        self.setGeometry(100, 100, 600, 400) # Smaller window is fine

        self.gl_widget = OpenGLWidget()
        self.setCentralWidget(self.gl_widget)

    # Ensure cleanup is called when the main window closes
    def closeEvent(self, event):
        print("MainWindow closing...")
        # Access the widget and call its cleanup method
        # Make sure the OpenGL context is current for cleanup calls
        self.gl_widget.makeCurrent()
        self.gl_widget.cleanupGL()
        self.gl_widget.doneCurrent()
        print("Cleanup finished.")
        super().closeEvent(event)


if __name__ == "__main__":
    gl_format = QSurfaceFormat()
    # Request OpenGL 4.3 Core Profile for Compute Shaders
    print("Requesting OpenGL 4.3 Core Profile...")
    gl_format.setVersion(4, 3)
    gl_format.setProfile(QSurfaceFormat.CoreProfile)
    # Enable debug context (optional, requires drivers supporting it)
    # gl_format.setOption(QSurfaceFormat.DebugContext)
    QSurfaceFormat.setDefaultFormat(gl_format)

    app = QApplication(sys.argv)

    # Check if the requested format could be obtained *after* app creation
    # Although default is set, the actual context creation happens later.
    # The check inside initializeGL is more definitive.

    print("Creating MainWindow...")
    window = MainWindow()
    window.show()
    print("Starting application event loop...")
    sys.exit(app.exec())