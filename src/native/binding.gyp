{
  "targets": [
    {
      "target_name": "llmdev_native",
      "sources": [
        "src/addon.cc",
        "src/kernels.cu"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "/usr/local/cuda/include"
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
      "cflags_cc": ["-std=c++17", "-O3", "-fPIC"],
      "libraries": [
        "-L/usr/local/cuda/lib64",
        "-L/usr/local/cuda/lib64/stubs",
        "-lcudart"
      ],
      "rules": [
        {
          "rule_name": "cuda_kernels",
          "extension": "cu",
          "message": "nvcc <(RULE_INPUT_PATH)",
          "process_outputs_as_sources": 1,
          "outputs": ["<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o"],
          "action": [
            "nvcc",
            "-c", "<(RULE_INPUT_PATH)",
            "-o", "<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o",
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-Xcompiler", "-fPIC",
            "-gencode", "arch=compute_120,code=sm_120",
            "-gencode", "arch=compute_120,code=compute_120"
          ]
        }
      ]
    }
  ]
}
