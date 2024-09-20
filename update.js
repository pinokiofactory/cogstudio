module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      message: "git pull",
      path: "app"
    }
  }, {
    method: "fs.copy",
    params: {
      src: "cogstudio.py",
      dest: "app/inference/gradio_composite_demo/cogstudio.py"
    }
  }]
}
