module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/THUDM/CogVideo app",
        ]
      }
    },
    {
      method: "fs.copy",
      params: {
        src: "cogstudio.py",
        dest: "app/inference/gradio_composite_demo/cogstudio.py"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "../../env",
        path: "app/inference/gradio_composite_demo",
        message: [
          "uv pip install -r requirements.txt",
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "app/env",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
          // xformers: true
        }
      }
    }
  ]
}
