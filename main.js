const canvas = document.getElementById("canvas")
const clearButton = document.getElementById("clear-button")
const classifyButton = document.getElementById("classify-button")
const modelStatus = document.getElementById("model-status")
const digit = document.getElementById("digit")
const score = document.getElementById("score")
const ctx = canvas.getContext("2d")

canvas.width = 28
canvas.height = 28
ctx.lineWidth = 3
ctx.lineCap = 'round'

var x, y;
var drawing;

async function loadModel() {
  const model = await tf.loadLayersModel("model/model.json")
  return model
}

function updatePos(event) {
  x = (event.clientX - canvas.offsetLeft) * (canvas.width * 1.0 / canvas.offsetWidth)
  y = (event.clientY - canvas.offsetTop) * (canvas.height * 1.0 / canvas.offsetHeight)
}

clearButton.onclick = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height)
}

canvas.addEventListener("mousemove", event => {
  if (drawing) {
    ctx.beginPath()
    ctx.moveTo(x, y)
    updatePos(event)
    ctx.lineTo(x, y)
    ctx.stroke()
  }
})

canvas.addEventListener("mousedown", event => {
  drawing = true
  updatePos(event)
})

canvas.addEventListener("mouseup", () => {
  drawing = false
})

canvas.addEventListener("mouseout", () => {
  drawing = false
})

loadModel().then(model => {
  modelStatus.innerHTML = "Model Loaded"
  classifyButton.disabled = false

  classifyButton.onclick = () => {
    var imageData = new Float32Array(
      ctx.getImageData(0, 0, canvas.width, canvas.height).data.filter((_, index) => index % 4 == 3)
    ).map(val => val * 1.0 / 255.0)

    model.predict(tf.tensor(imageData).reshape([-1, 28, 28, 1])).data().then(result => {
      var curDigit = -1
      var curScore = 0

      result.forEach((value, index) => {
        if (value > curScore) {
          curDigit = index
          curScore = value
        }
      })

      console.log(curDigit)

      digit.innerHTML = curDigit
      score.innerHTML = curScore
    })
  }
})
