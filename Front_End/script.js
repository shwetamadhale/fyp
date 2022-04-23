// (function () {
// define relevant variables
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext('2d');
var dragging = false;
var pos = { x: 0, y: 0 };


// define event listeners for both desktop and mobile

// nontouch
canvas.addEventListener('mousedown',  engage);
canvas.addEventListener('mousedown',  setPosition);
canvas.addEventListener('mousemove',  draw);
canvas.addEventListener('mouseup', disengage);

// touch
canvas.addEventListener('touchstart', engage);
canvas.addEventListener('touchmove', setPosition);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', disengage);

// detect if it is a touch device
function isTouchDevice() {
  return (
    ('ontouchstart' in window) ||
    (navigator.maxTouchPoints > 0) ||
    (navigator.msMaxTouchPoints > 0)
  );
}

// define basic functions to detect click / release

function engage() {
  dragging = true;
};

function disengage() {
  dragging = false;
};

// get the new position given a mouse / touch event
function setPosition(e) {

  if (isTouchDevice()) {
  	var touch = e.touches[0];
  	pos.x = touch.clientX - ctx.canvas.offsetLeft;
  	pos.y = touch.clientY - ctx.canvas.offsetTop;
  } else {
  
	  pos.x = e.clientX - ctx.canvas.offsetLeft;
  	pos.y = e.clientY - ctx.canvas.offsetTop;
  }
}

// draws a line in a canvas if mouse is pressed
function draw(e) {
  
  e.preventDefault();
  e.stopPropagation();

  // to draw the user needs to be engaged (dragging = True)
  if (dragging) {

    // begin drawing
    ctx.beginPath();
  
    // attributes of the line
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    // get current position, move to new position, create line from current to new
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);

    // draw
    ctx.stroke();
  }
}


// clear canvas
function erase() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}




// -----------------------------------------------------------------------------

//integrating  CANVAS  with CNN MODEL


//loading the model

//the base url of website in which our 
//web app is deployed is obtained from window.location.origin
//the json file is loaded using async function

var base_url = window.location.origin;
let model;
(async function(){  
    console.log("model loading...");  
    model = await tf.loadLayersModel("https://raw.githubusercontent.com/Meenakshiee/tfjs-model/main/model.json")
    console.log("model loaded..");
})();

//preprocessing model

/*
the digit sketched is passed as image to model
so as to predict the value of it
*/

function preprocessCanvas(image) { 
   
    //resizing the input image to target size of (1, 28, 28) 
    //tf.browser.fromPixels() method, to create a tensor that will flow into the first layer of the model
    //tf.image.resizeNearestNeighbor() function resizes a batch of 3D images to a new shape
    //tf.mean() function is used to compute the mean of elements across the dimensions of the tensor
    //tf.toFloat() function casts the array to type float
    //The tensor.div() function is used to divide the array or tensor by the maximum RGB value(255)
    let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([28, 28]).mean(2).expandDims(2).expandDims().toFloat(); 
    console.log(tensor.shape); 
    return tensor.div(255.0);
}

//Prediction
//canvas.toDataURL() : returns 
//image in format specified default png
//than send to preprocess function
//await makes program wait until mmodel prediction
//displayLabel to display result
document.getElementById('predict_button').addEventListener("click",async function(){     
    var imageData = canvas.toDataURL();    
    let tensor = preprocessCanvas(canvas); 
    console.log(tensor)   
    let predictions = await model.predict(tensor).data();  
    console.log(predictions)  
    let results = Array.from(predictions);    
    displayLabel(results);    
    console.log(results);
});


//output
function displayLabel(data) { 
    var max = data[0];    
    var maxIndex = 0;     
    for (var i = 1; i < data.length; i++) {        
      if (data[i] > max) {            
        maxIndex = i;            
        max = data[i];        
      }
    }
document.getElementById('results').innerHTML = maxIndex;  
// document.getElementById('confidence').innerHTML = "Confidence: "+(max*100).toFixed(2) + "%";
}





// --------------------------------------------------------------------- website
// // defines the model inference functino
// async function predictModel(){
    
//   // gets image data
//   imageData = getData();
  
//   // converts from a canvas data object to a tensor
//   image = tf.browser.fromPixels(imageData)
  
//   // pre-process image
//   image = tf.image.resizeBilinear(image, [28,28]).sum(2).expandDims(0).expandDims(-1)
  
//   // gets model prediction
//   y = model.predict(image).dataSync();
  
//   // replaces the text in the result tag by the model prediction
//   document.getElementById('result').innerHTML = "Prediction: " + y.argMax(1).dataSync();
// }

// // defines a TF model load function
// async function loadModel(){	
  	
//   // loads the model
//   model = await tf.loadLayersModel('https://raw.githubusercontent.com/Meenakshiee/tfjs-model/main/model.json');    
  
//   // warm start the model. speeds up the first inference
//   model.predict(tf.zeros([1, 28, 28, 1]))
  
//   // return model
//   return model
// }

// // gets an image tensor from a canvas
// function getData(){
//   return ctx.getImageData(0, 0, canvas.width, canvas.height);
// }

// -------------------------------------------------------------




















// const canvas = document.getElementById('main-canvas');
// const smallCanvas = document.getElementById('small-canvas');
// const displayBox = document.getElementById('prediction');

// const inputBox = canvas.getContext('2d');
// const smBox = smallCanvas.getContext('2d');

// let isDrawing = false;
// let model;


// /* Loads trained model */
// async function init() {
//   model = await tf.loadModel('https://raw.githubusercontent.com/Meenakshiee/tfjs-model/main/model.json');
// }

// canvas.addEventListener('mousedown', event => {
//   isDrawing = true;

//   inputBox.strokeStyle = 'white';
//   inputBox.lineWidth = '3';
//   inputBox.lineJoin = inputBox.lineCap = 'round';
//   inputBox.beginPath();
// });

// canvas.addEventListener('mousemove', event => {
//   if (isDrawing) drawStroke(event.clientX, event.clientY);
// });

// canvas.addEventListener('mouseup', event => {
//   isDrawing = false;
//   updateDisplay(predict());
// });

// /* Draws on canvas */
// function drawStroke(clientX, clientY) {
//   // get mouse coordinates on canvas
//   const rect = canvas.getBoundingClientRect();
//   const x = clientX - rect.left;
//   const y = clientY - rect.top;

//   // draw
//   inputBox.lineTo(x, y);
//   inputBox.stroke();
//   inputBox.moveTo(x, y);
// }

// /* Makes predictions */
// function predict() {
//   let values = getPixelData();
//   let predictions = model.predict(values).dataSync();

//   return predictions;
// }

// /* Returns pixel data from canvas after applying transformations */
// function getPixelData() {
//   smBox.drawImage(inputBox.canvas, 0, 0, smallCanvas.width, smallCanvas.height);
//   const imgData = smBox.getImageData(0, 0, smallCanvas.width, smallCanvas.height);

//   // preserve and normalize values from red channel only
//   let values = [];
//   for (let i = 0; i < imgData.data.length; i += 4) {
//     values.push(imgData.data[i] / 255);
//   }
//   values = tf.reshape(values, [1, 28, 28, 1]);
//   return values;
// }

// /* Displays predictions on screen */
// function updateDisplay(predictions) {
//   // Find index of best prediction, which corresponds to the predicted value
//   const bestPred = predictions.indexOf(Math.max(...predictions));
//   displayBox.innerText = bestPred;
// }

// document.getElementById('erase').addEventListener('click', erase);

// /* Clears canvas */
// function erase() {
//   inputBox.fillStyle = 'black';
//   inputBox.fillRect(0, 0, canvas.width, canvas.height);
//   displayBox.innerText = '';
// }

// erase();
// init();
