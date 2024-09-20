module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "fs.rm",
    params: {
      path: "app"
    }
  }, {
    method: "script.start",
    params: {
      path: "install.js"
    }
  }]
}
