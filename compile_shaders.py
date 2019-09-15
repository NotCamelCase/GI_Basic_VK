# Compile SPIR-V shaders in current folder using glslangValidator from Vulkan SDK

import os
import subprocess

shaderExts = [ ".vert", ".frag", ".tesc", ".tese", ".geom", ".comp" ]
shadersToCompile = [f for f in os.listdir(os.getcwd()) if os.path.splitext(f)[1] in shaderExts]
for shader in shadersToCompile:
    shaderName = shader.split(".")[0] + "_" + shader.split(".")[1]
    compileCmd = "glslangValidator.exe {shaderFile} -V -o {outputFile}.spv".format(shaderFile=shader, outputFile=shaderName)
    subprocess.call(compileCmd)