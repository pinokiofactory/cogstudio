module.exports = {
  run: [
    // Edit this step to customize the git repository to use
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
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "../../env",                // Edit this to customize the venv folder path
        path: "app/inference/gradio_composite_demo",                // Edit this to customize the path to start the shell from
        message: [
          "pip install -r requirements.txt",
        ]
      }
    },
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "app/env",
        message: [
          "pip install -r requirements.txt",
        ]
      }
    },
    // Delete this step if your project does not use torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",                // Edit this to customize the venv folder path
          path: "app",                // Edit this to customize the path to start the shell from
          // xformers: true   // uncomment this line if your project requires xformers
        }
      }
    },
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    }
  ]
}
